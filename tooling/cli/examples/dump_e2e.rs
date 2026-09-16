//! End-to-end parity dump (f2z-pcs stage B): one seeded SHA-256 statement
//! proved through the whole scheme — `Prepared::new/witness/commit/prove/
//! verify` for the canonical timings, and the same flow by hand through the
//! public pieces (`prove_spartan_piop`, `VirtualStatement`,
//! `BitZProver::prove_virtual`) so the terminal claim and the claim on `h`
//! can be written out too. The two flows must agree on root, narg, hints.
//!
//! `dump_e2e <sha256-compression|sha256-chain> <blocks> <seed> <dir> [--sampled]`
//! (`--sampled`: the sampled-prime scheme, `PreparedSampled`, 100-bit prime)
//!
//! Files: `meta.txt`, `public.bin`, `inputs.bin` (LSB-first bits), `f.bin`,
//! `h.bin` (packed bits), `spartan.bin` (the feasibility doc's canonical
//! form, terminal claim included), `claim_h.bin` (u64 rows ‖ rows ‖ u64
//! cols ‖ cols ‖ target, Fq residues 16 LE bytes), `narg.bin`, `hints.bin`,
//! and `products.bin` (Az ‖ Bz ‖ Cz) up to 64 blocks.
use std::io::Write;

use bitz_cli::end_to_end::{CircuitStatement, Prepared, PreparedSampled};
use circuit::{
    constraints::ConstraintGenerator,
    matrix_transpose::MTransposeGenerator,
    sha256::INITIAL_STATE,
    witgen::{PackedWitness, ProductWitgen},
};
use common::{BitZParams, LinearClaim, Shape, VirtualMap, VirtualStatement};
use field::{F128, Fq, FqDefault, Q100, RUNTIME, gf128::smallest_generator};
use num_traits::ConstZero;
use pcs::{HashKind, LigeritoProfile, Pcs};
use poly::ScaledMleEvaluationClaim;
use prover::{BitZProver, VirtualWitness};
use spartan::{
    PreparedConstraintMatrices, R1csProductMles, SpartanPiopProof, bigint_to_fq,
    build_assignment_mle, build_product_mles, prove_spartan_piop, verify_spartan_proof,
};
use transcript::{Encoding, PublicTranscript, build_prover, build_verifier};
use verifier::BitZVerifier;

#[path = "common/sha256.rs"]
mod sha256;
use sha256::{Sha256Circuit, Sha256Statement};

const E2E_SESSION: &[u8] = b"bitz/circuit-e2e/v1";
const WINDOW: u32 = 8;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn seeded_statement(circuit: Sha256Circuit, blocks: usize, seed: u64) -> Sha256Statement {
    let mut state = seed;
    let mut digest = INITIAL_STATE;
    let blocks = (0..blocks)
        .map(|_| {
            let mut bytes = [0u8; 64];
            for chunk in bytes.chunks_exact_mut(8) {
                chunk.copy_from_slice(&splitmix64(&mut state).to_le_bytes());
            }
            sha2::compress256(&mut digest, &[bytes.into()]);
            std::array::from_fn(|i| u32::from_be_bytes(std::array::from_fn(|j| bytes[4 * i + j])))
        })
        .collect();
    Sha256Statement {
        circuit,
        blocks,
        initial_state: INITIAL_STATE,
        digest,
    }
}

fn shape_for(bits: usize) -> Shape {
    let padded = bits.checked_next_power_of_two().expect("witness too large");
    let log_bits = (padded.ilog2() as usize).max(common::shape::MIN_LOG_BITS);
    Shape::for_log_bits(log_bits).expect("shape")
}

fn pack(witness: &PackedWitness, shape: Shape) -> Vec<F128> {
    let mut packed: Vec<_> = witness
        .words()
        .chunks(2)
        .map(|words| F128::new(words[0], words.get(1).copied().unwrap_or(0)))
        .collect();
    packed.resize(1 << shape.log_packed_len(), F128::ZERO);
    packed
}

fn opening_claim<const Q: u128>(
    params: &BitZParams<Q>,
    terminal: &ScaledMleEvaluationClaim<Fq<Q>>,
) -> LinearClaim<Fq<Q>> {
    let shape = params.shape();
    let mut point = terminal.point().to_vec();
    point.resize(shape.log_bits(), Fq::<Q>::ZERO);
    let rows = poly::eq_table(&point[..shape.log_rows()])
        .into_iter()
        .map(|weight| terminal.scale() * weight)
        .collect();
    let columns = poly::eq_table(&point[shape.log_rows()..]);
    LinearClaim::new(params, rows, columns, terminal.value()).expect("claim dimensions")
}

fn fq_bytes<const Q: u128>(value: &Fq<Q>) -> [u8; 16] {
    value.encode().as_ref().try_into().expect("16 bytes")
}

fn spartan_bytes<const Q: u128>(proof: &SpartanPiopProof<Fq<Q>>, terminal: &ScaledMleEvaluationClaim<Fq<Q>>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for round in &proof.outer.sumcheck.round_polynomials {
        for c in round {
            bytes.extend_from_slice(&fq_bytes(c));
        }
    }
    for v in [&proof.outer.az_mle_claim, &proof.outer.bz_mle_claim, &proof.outer.cz_mle_claim] {
        bytes.extend_from_slice(&fq_bytes(v));
    }
    for round in &proof.inner.round_polynomials {
        for c in round {
            bytes.extend_from_slice(&fq_bytes(c));
        }
    }
    for c in terminal.point() {
        bytes.extend_from_slice(&fq_bytes(c));
    }
    bytes.extend_from_slice(&fq_bytes(&terminal.scale()));
    bytes.extend_from_slice(&fq_bytes(&terminal.value()));
    bytes
}

fn packed_bits(bits: &PackedWitness) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.bit_len().div_ceil(8));
    for (i, word) in bits.words().iter().enumerate() {
        let remaining = bits.bit_len() - 64 * i;
        out.extend_from_slice(&word.to_le_bytes()[..remaining.min(64).div_ceil(8)]);
    }
    out
}

fn bools_to_bytes(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

fn claim_bytes<const Q: u128>(claim: &LinearClaim<Fq<Q>>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for weights in [claim.row_weights(), claim.column_weights()] {
        bytes.extend_from_slice(&(weights.len() as u64).to_le_bytes());
        for w in weights {
            bytes.extend_from_slice(&fq_bytes(w));
        }
    }
    bytes.extend_from_slice(&fq_bytes(&claim.target()));
    bytes
}

/// The sampled-prime scheme on the same statement: `PreparedSampled` end to
/// end (its proof carries the terminal claim and the prime), the same files
/// with `kind=e2e-sampled`, `prime_bits`, `prime`, the integer digest as
/// `constraint_digest`, and `q` the prime.
fn dump_sampled(
    circuit: Sha256Circuit,
    blocks: usize,
    seed: u64,
    out: &std::path::Path,
    args0: &str,
) {
    const PRIME_BITS: u32 = 100;
    let statement = seeded_statement(circuit, blocks, seed);
    let inputs = statement.input();
    let started = std::time::Instant::now();
    let prepared = PreparedSampled::new(statement.clone(), PRIME_BITS).expect("prepared");
    let t_setup = started.elapsed();
    let started = std::time::Instant::now();
    let witness = prepared.witness(&inputs).expect("witness");
    let t_witness = started.elapsed();
    let started = std::time::Instant::now();
    let data = prepared.commit(&witness).expect("commit");
    let t_commit = started.elapsed();
    let started = std::time::Instant::now();
    let proof = prepared.prove(witness, &data).expect("prove");
    let t_prove = started.elapsed();
    let started = std::time::Instant::now();
    prepared.verify(&proof).expect("verify");
    let t_verify = started.elapsed();
    assert_eq!(proof.prime, prepared.prime_for(proof.root));

    // f and h for the files, by hand.
    let mut witgen = ProductWitgen::with_inputs(&inputs);
    statement.synthesize(&mut witgen, &inputs).expect("witgen");
    let (f, h, exact) = witgen.into_parts();
    let params = BitZParams::<RUNTIME>::new(prepared.claim_shape(), smallest_generator()).expect("params");
    let claim = opening_claim(&params, &proof.terminal);
    let spartan_bin = spartan_bytes(&proof.spartan, &proof.terminal);
    let public = statement.public_bytes();

    std::fs::create_dir_all(out).unwrap();
    let mut files: Vec<(&str, Vec<u8>)> = vec![
        ("public.bin", public.clone()),
        ("inputs.bin", bools_to_bytes(&inputs)),
        ("f.bin", packed_bits(&f)),
        ("h.bin", packed_bits(&h)),
        ("spartan.bin", spartan_bin.clone()),
        ("claim_h.bin", claim_bytes(&claim)),
        ("narg.bin", proof.opening.narg_string.clone()),
        ("hints.bin", proof.opening.hints.clone()),
    ];
    if blocks <= 64 {
        let products = spartan::build_product_mles_in::<RUNTIME>(&exact, prepared.matrices().matrices().a.row_count()).unwrap();
        let mut bytes = Vec::new();
        for table in [&products.az, &products.bz, &products.cz] {
            for v in table.iter() {
                bytes.extend_from_slice(&fq_bytes(v));
            }
        }
        files.push(("products.bin", bytes));
    }
    for (name, bytes) in &files {
        std::fs::write(out.join(name), bytes).unwrap();
    }
    let meta = [
        ("kind", "e2e-sampled".to_string()),
        ("circuit", args0.to_string()),
        ("blocks", blocks.to_string()),
        ("seed", seed.to_string()),
        ("session", String::from_utf8_lossy(bitz_cli::end_to_end::SESSION_SAMPLED).into()),
        ("instance", String::from_utf8_lossy(statement.domain()).into()),
        ("input_bits", inputs.len().to_string()),
        ("root", hex(&proof.root.0)),
        ("prime_bits", PRIME_BITS.to_string()),
        ("prime", proof.prime.to_string()),
        ("q", proof.prime.to_string()),
        ("claim_t", prepared.claim_shape().log_rows().to_string()),
        ("claim_s", prepared.claim_shape().log_columns().to_string()),
        ("committed_t", prepared.committed_shape().log_rows().to_string()),
        ("committed_s", prepared.committed_shape().log_columns().to_string()),
        ("generator", hex(&params.generator().to_bytes())),
        ("constraint_digest", hex(prepared.matrices().digest())),
        ("map_digest", hex(&prepared.map().digest())),
        ("rows", prepared.matrices().matrices().a.row_count().to_string()),
        ("h_len", prepared.map().h_len().to_string()),
        ("f_len", prepared.map().f_len().to_string()),
        ("map_nnz", prepared.map().nonzero_count().to_string()),
        ("num_row_vars", prepared.matrices().num_row_vars().to_string()),
        ("num_column_vars", prepared.matrices().num_column_vars().to_string()),
        ("spartan_len", spartan_bin.len().to_string()),
        ("narg_len", proof.opening.narg_string.len().to_string()),
        ("hints_len", proof.opening.hints.len().to_string()),
        ("setup_ms", format!("{:.3}", ms(t_setup))),
        ("witness_ms", format!("{:.3}", ms(t_witness))),
        ("commit_ms", format!("{:.3}", ms(t_commit))),
        ("prove_ms", format!("{:.3}", ms(t_prove))),
        ("verify_ms", format!("{:.3}", ms(t_verify))),
    ];
    let mut text = String::new();
    for (k, v) in &meta {
        text.push_str(&format!("{k}={v}\n"));
    }
    std::fs::write(out.join("meta.txt"), &text).unwrap();
    print!("{text}");
    let _ = std::io::stdout().flush();
}

fn products_bytes(products: &R1csProductMles<FqDefault>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for table in [&products.az, &products.bz, &products.cz] {
        for v in table.iter() {
            bytes.extend_from_slice(&fq_bytes(v));
        }
    }
    bytes
}

/// `Prepared::bind`, object-safe over both transcript halves.
trait BindTarget {
    fn bind(&mut self, public: &[u8], root: &[u8; 32], params: &BitZParams<Q100>, pcs: &Pcs, map_digest: &[u8; 32]);
}

impl<T: PublicTranscript> BindTarget for T {
    fn bind(&mut self, public: &[u8], root: &[u8; 32], params: &BitZParams<Q100>, pcs: &Pcs, map_digest: &[u8; 32]) {
        self.public_message(&(public.len() as u64));
        self.public_message(public);
        self.public_message(root);
        self.public_message(params);
        self.public_message(pcs);
        self.public_message(map_digest);
    }
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let circuit = match args[0].as_str() {
        "sha256-compression" => Sha256Circuit::Compression,
        "sha256-chain" => Sha256Circuit::Chain,
        other => panic!("unknown circuit {other}"),
    };
    let blocks: usize = args[1].parse().expect("blocks");
    let seed: u64 = args[2].parse().expect("seed");
    let out = std::path::PathBuf::from(&args[3]);
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});
    if args.iter().any(|a| a == "--sampled") {
        dump_sampled(circuit, blocks, seed, &out, &args[0]);
        return;
    }

    let statement = seeded_statement(circuit, blocks, seed);
    let inputs = statement.input();

    // The canonical flow, for the timings and as the reference proof.
    let started = std::time::Instant::now();
    let prepared = Prepared::new(statement.clone()).expect("prepared");
    let t_setup = started.elapsed();
    let started = std::time::Instant::now();
    let witness = prepared.witness(&inputs).expect("witness");
    let t_witness = started.elapsed();
    let started = std::time::Instant::now();
    let data = prepared.commit(&witness).expect("commit");
    let t_commit = started.elapsed();
    let started = std::time::Instant::now();
    let reference = prepared.prove(witness, &data).expect("prove");
    let t_prove = started.elapsed();
    let started = std::time::Instant::now();
    prepared.verify(&reference).expect("verify");
    let t_verify = started.elapsed();
    drop(prepared);
    drop(data);

    // The same flow by hand, for the terminal claim and the claim on h.
    let mut constraints = ConstraintGenerator::new(statement.input_bits());
    let symbolic: Vec<_> = (0..statement.input_bits()).map(|i| constraints.input(i)).collect();
    statement.synthesize(&mut constraints, &symbolic).expect("synthesize");
    let matrices = PreparedConstraintMatrices::new(constraints.into_matrices().map_coefficients(|c| bigint_to_fq(&c))).unwrap();
    let mut generator = MTransposeGenerator::new(statement.input_bits());
    let map_inputs = generator.take_inputs();
    statement.synthesize(&mut generator, &map_inputs).expect("synthesize map");
    let map = generator.finish();
    let claim_shape = shape_for(map.h_len());
    let committed_shape = shape_for(map.f_len() - 1);
    let params = BitZParams::<Q100>::new(claim_shape, smallest_generator()).expect("params");
    let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3).expect("pcs");
    let mut witgen = ProductWitgen::with_inputs(&inputs);
    statement.synthesize(&mut witgen, &inputs).expect("witgen");
    let (f, h, exact) = witgen.into_parts();
    let products = build_product_mles(&exact, matrices.matrices().a.row_count()).unwrap();
    let assignment = build_assignment_mle::<FqDefault>(&h, map.h_len()).unwrap();
    let committed = pack(&f, committed_shape);
    let virtual_bits = pack(&h, claim_shape);
    let (root, data) = pcs.commit(&committed).expect("commit");
    assert_eq!(root, reference.root, "by-hand commit differs from Prepared's");
    let public = statement.public_bytes();
    let map_digest = map.digest();

    let mut prover = build_prover(E2E_SESSION, statement.domain());
    prover.bind(&public, &root.0, &params, &pcs, &map_digest);
    let (spartan, terminal) = prove_spartan_piop(&mut prover, &matrices, &products, &assignment).expect("spartan");
    let claim = opening_claim(&params, &terminal);
    let vstatement = VirtualStatement::new(params, committed_shape, &map, &claim).expect("virtual statement");
    BitZProver::new(params, WINDOW)
        .prove_virtual(
            &vstatement,
            &pcs,
            &data,
            VirtualWitness {
                committed_bits: committed.clone(),
                virtual_bits: &virtual_bits,
            },
            &mut prover,
        )
        .expect("prove_virtual");
    let opening = prover.finish();
    assert_eq!(opening.narg_string, reference.opening.narg_string, "by-hand narg differs from Prepared's");
    assert_eq!(opening.hints, reference.opening.hints, "by-hand hints differ from Prepared's");
    assert_eq!(spartan, reference.spartan);

    // Verify by hand too (recomputes the terminal claim through evaluate_batched).
    let mut verifier = build_verifier(E2E_SESSION, statement.domain(), &opening);
    verifier.bind(&public, &root.0, &params, &pcs, &map_digest);
    let checked = verify_spartan_proof(&mut verifier, &matrices, &spartan).expect("spartan verify");
    assert_eq!(checked.point(), terminal.point());
    assert_eq!(checked.scale(), terminal.scale());
    assert_eq!(checked.value(), terminal.value());
    BitZVerifier::new(params, WINDOW)
        .verify_virtual(&vstatement, &pcs, root, verifier)
        .expect("verify_virtual");

    let spartan_bin = spartan_bytes(&spartan, &terminal);
    let mut claim_bytes = Vec::new();
    for weights in [claim.row_weights(), claim.column_weights()] {
        claim_bytes.extend_from_slice(&(weights.len() as u64).to_le_bytes());
        for w in weights {
            claim_bytes.extend_from_slice(&fq_bytes(w));
        }
    }
    claim_bytes.extend_from_slice(&fq_bytes(&claim.target()));

    std::fs::create_dir_all(&out).unwrap();
    let mut files: Vec<(&str, Vec<u8>)> = vec![
        ("public.bin", public.clone()),
        ("inputs.bin", bools_to_bytes(&inputs)),
        ("f.bin", packed_bits(&f)),
        ("h.bin", packed_bits(&h)),
        ("spartan.bin", spartan_bin.clone()),
        ("claim_h.bin", claim_bytes),
        ("narg.bin", opening.narg_string.clone()),
        ("hints.bin", opening.hints.clone()),
    ];
    if blocks <= 64 {
        files.push(("products.bin", products_bytes(&products)));
    }
    for (name, bytes) in &files {
        std::fs::write(out.join(name), bytes).unwrap();
    }
    let meta = [
        ("kind", "e2e-full".to_string()),
        ("circuit", args[0].clone()),
        ("blocks", blocks.to_string()),
        ("seed", seed.to_string()),
        ("session", String::from_utf8_lossy(E2E_SESSION).into()),
        ("instance", String::from_utf8_lossy(statement.domain()).into()),
        ("input_bits", inputs.len().to_string()),
        ("root", hex(&root.0)),
        ("claim_t", claim_shape.log_rows().to_string()),
        ("claim_s", claim_shape.log_columns().to_string()),
        ("committed_t", committed_shape.log_rows().to_string()),
        ("committed_s", committed_shape.log_columns().to_string()),
        ("q", Q100.to_string()),
        ("generator", hex(&params.generator().to_bytes())),
        ("constraint_digest", hex(matrices.digest())),
        ("map_digest", hex(&map_digest)),
        ("rows", matrices.matrices().a.row_count().to_string()),
        ("h_len", map.h_len().to_string()),
        ("f_len", map.f_len().to_string()),
        ("map_nnz", map.nonzero_count().to_string()),
        ("num_row_vars", matrices.num_row_vars().to_string()),
        ("num_column_vars", matrices.num_column_vars().to_string()),
        ("spartan_len", spartan_bin.len().to_string()),
        ("narg_len", opening.narg_string.len().to_string()),
        ("hints_len", opening.hints.len().to_string()),
        ("setup_ms", format!("{:.3}", ms(t_setup))),
        ("witness_ms", format!("{:.3}", ms(t_witness))),
        ("commit_ms", format!("{:.3}", ms(t_commit))),
        ("prove_ms", format!("{:.3}", ms(t_prove))),
        ("verify_ms", format!("{:.3}", ms(t_verify))),
    ];
    let mut text = String::new();
    for (k, v) in &meta {
        text.push_str(&format!("{k}={v}\n"));
    }
    std::fs::write(out.join("meta.txt"), &text).unwrap();
    print!("{text}");
    let _ = std::io::stdout().flush();
}
