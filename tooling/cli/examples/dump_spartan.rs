//! Spartan-alone parity dump (f2z-pcs stage A): the PIOP's out-of-band proof
//! and terminal claim in the feasibility doc's canonical byte form, together
//! with everything our side needs to replay the same transcript prefix.
//!
//! `dump_spartan abc <dir>` — their `spartan/tests/sha256_piop.rs` fixture:
//!   the abc compression through `compression_circuit`, session
//!   `spartan/piop/sha256-compression/v1`, instance `abc-single-compression`.
//! `dump_spartan <sha256-compression|sha256-chain> <blocks> <seed> <dir>` —
//!   the e2e statement (seeded like `e2e_probe`), the e2e transcript prefix
//!   (`Prepared::bind`), then `prove_spartan_piop`; also the `opening_claim`
//!   on `h` the e2e derives from the terminal claim.
//!
//! Files: `meta.txt`, `inputs.bin` (input bits, LSB-first), `public.bin`
//! (e2e), `f.bin`/`h.bin` (packed bits), `products.bin` (Az ‖ Bz ‖ Cz padded,
//! 16 LE bytes each), `spartan.bin`, `claim_h.bin` (e2e: u64 rows ‖ rows ‖
//! u64 cols ‖ cols ‖ target). `next_challenge` in `meta.txt` is one `F128`
//! squeezed after the PIOP, pinning the sponge state.
use std::io::Write;

use bitz_cli::end_to_end::CircuitStatement;
use circuit::{
    constraints::ConstraintGenerator,
    matrix_transpose::MTransposeGenerator,
    sha256::{ABC_BLOCK, ABC_DIGEST, COMPRESSION_HINT_BITS, COMPRESSION_INPUT_BITS, INITIAL_STATE, compression_circuit},
    witgen::{PackedWitness, ProductWitgen},
};
use common::{BitZParams, LinearClaim, Shape, VirtualMap};
use field::{F128, FqDefault, Q100, gf128::smallest_generator};
use num_traits::ConstZero;
use pcs::{HashKind, LigeritoProfile, Pcs};
use poly::ScaledMleEvaluationClaim;
use spartan::{
    PreparedConstraintMatrices, R1csProductMles, SpartanPiopProof, bigint_to_fq,
    build_assignment_mle, build_product_mles, prove_spartan_piop, verify_spartan_proof,
};
use transcript::{Encoding, ProverState, PublicTranscript, build_prover, build_verifier};

#[path = "common/sha256.rs"]
mod sha256;
use sha256::{Sha256Circuit, Sha256Statement};

const E2E_SESSION: &[u8] = b"bitz/circuit-e2e/v1";

/// `Prepared::bind` (`end_to_end.rs:200-208`), object-safe.
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

fn seeded_statement(circuit: Sha256Circuit, blocks: usize, seed: u64) -> Sha256Statement {
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

/// `end_to_end.rs::pack`.
fn pack(witness: &PackedWitness, shape: Shape) -> Vec<F128> {
    let mut packed: Vec<_> = witness
        .words()
        .chunks(2)
        .map(|words| F128::new(words[0], words.get(1).copied().unwrap_or(0)))
        .collect();
    packed.resize(1 << shape.log_packed_len(), F128::ZERO);
    packed
}

/// `end_to_end.rs::opening_claim`.
fn opening_claim(
    params: &BitZParams<Q100>,
    terminal: &ScaledMleEvaluationClaim<FqDefault>,
) -> LinearClaim<FqDefault> {
    let shape = params.shape();
    assert!(terminal.point().len() <= shape.log_bits());
    let mut point = terminal.point().to_vec();
    point.resize(shape.log_bits(), FqDefault::ZERO);
    let rows = poly::eq_table(&point[..shape.log_rows()])
        .into_iter()
        .map(|weight| terminal.scale() * weight)
        .collect();
    let columns = poly::eq_table(&point[shape.log_rows()..]);
    LinearClaim::new(params, rows, columns, terminal.value()).expect("claim dimensions")
}

fn fq_bytes(value: &FqDefault) -> [u8; 16] {
    value.encode().as_ref().try_into().expect("16 bytes")
}

fn spartan_bytes(proof: &SpartanPiopProof<FqDefault>, terminal: &ScaledMleEvaluationClaim<FqDefault>) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut put = |v: &FqDefault| bytes.extend_from_slice(&fq_bytes(v));
    for round in &proof.outer.sumcheck.round_polynomials {
        for c in round {
            put(c);
        }
    }
    for v in [&proof.outer.az_mle_claim, &proof.outer.bz_mle_claim, &proof.outer.cz_mle_claim] {
        put(v);
    }
    for round in &proof.inner.round_polynomials {
        for c in round {
            put(c);
        }
    }
    for c in terminal.point() {
        put(c);
    }
    put(&terminal.scale());
    put(&terminal.value());
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

fn products_bytes(products: &R1csProductMles<FqDefault>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for table in [&products.az, &products.bz, &products.cz] {
        for v in table.iter() {
            bytes.extend_from_slice(&fq_bytes(v));
        }
    }
    bytes
}

struct Dump {
    meta: Vec<(String, String)>,
    files: Vec<(&'static str, Vec<u8>)>,
}

impl Dump {
    fn write(&self, out: &std::path::Path) {
        std::fs::create_dir_all(out).unwrap();
        for (name, bytes) in &self.files {
            std::fs::write(out.join(name), bytes).unwrap();
        }
        let mut meta = String::new();
        for (k, v) in &self.meta {
            meta.push_str(&format!("{k}={v}\n"));
        }
        std::fs::write(out.join("meta.txt"), &meta).unwrap();
        print!("{meta}");
    }
}

/// Runs the PIOP on a transcript already carrying its prefix; verifies it on
/// a verifier transcript built by `verifier`; returns the bytes and the
/// pinned next challenge.
fn prove_and_check(
    mut prover: ProverState,
    verifier: impl FnOnce(&transcript::Proof) -> transcript::VerifierState<'_>,
    matrices: &PreparedConstraintMatrices<FqDefault>,
    products: &R1csProductMles<FqDefault>,
    assignment: &poly::DenseMultilinearExtension<FqDefault>,
) -> (SpartanPiopProof<FqDefault>, ScaledMleEvaluationClaim<FqDefault>, Vec<u8>, F128, f64, f64) {
    let started = std::time::Instant::now();
    let (proof, terminal) = prove_spartan_piop(&mut prover, matrices, products, assignment).expect("spartan");
    let prove_ms = started.elapsed().as_secs_f64() * 1e3;
    let next: F128 = prover.verifier_message();
    let transcript_proof = prover.finish();
    assert!(transcript_proof.narg_string.is_empty() && transcript_proof.hints.is_empty());
    let mut v = verifier(&transcript_proof);
    let started = std::time::Instant::now();
    let checked = verify_spartan_proof(&mut v, matrices, &proof).expect("spartan verify");
    let verify_ms = started.elapsed().as_secs_f64() * 1e3;
    assert_eq!(checked.point(), terminal.point());
    assert_eq!(checked.scale(), terminal.scale());
    assert_eq!(checked.value(), terminal.value());
    let next_v: F128 = v.verifier_message();
    assert_eq!(next_v, next);
    v.check_eof().expect("eof");
    let bytes = spartan_bytes(&proof, &terminal);
    (proof, terminal, bytes, next, prove_ms, verify_ms)
}

/// Replays the proof through the verifier's pieces and records every squeezed
/// challenge (`tau ‖ r_x ‖ rho ‖ r_y`), and the prover's dense `D` table.
fn challenges_and_table(
    mut verifier: transcript::VerifierState<'_>,
    matrices: &PreparedConstraintMatrices<FqDefault>,
    proof: &SpartanPiopProof<FqDefault>,
) -> (Vec<u8>, Vec<u8>) {
    verifier.public_message(matrices.digest());
    let tau: Vec<FqDefault> = (0..matrices.num_row_vars()).map(|_| verifier.squeeze::<FqDefault>()).collect();
    let outer = proof.outer.verify(&mut verifier, FqDefault::ZERO, &tau).expect("outer");
    let rho = verifier.squeeze::<FqDefault>();
    let inner_claim = outer.az_mle_claim + rho * outer.bz_mle_claim + rho * rho * outer.cz_mle_claim;
    let (r_y, _) = proof.inner.verify(&mut verifier, inner_claim, matrices.num_column_vars()).expect("inner");
    let mut bytes = Vec::new();
    for v in tau.iter().chain(outer.eval_points.iter()).chain(std::iter::once(&rho)).chain(r_y.iter()) {
        bytes.extend_from_slice(&fq_bytes(v));
    }
    let table = matrices.bind_and_batch(&outer.eval_points, rho).expect("bind");
    let mut d = Vec::new();
    for v in table.iter() {
        d.extend_from_slice(&fq_bytes(v));
    }
    (bytes, d)
}

fn abc(out: &std::path::Path) {
    const SESSION: &[u8] = b"spartan/piop/sha256-compression/v1";
    const INSTANCE: &[u8] = b"abc-single-compression";
    let inputs: Vec<bool> = (0..COMPRESSION_INPUT_BITS)
        .map(|index| {
            let (word, bit) = if index < 512 {
                (ABC_BLOCK[index / 32], index % 32)
            } else {
                (INITIAL_STATE[(index - 512) / 32], (index - 512) % 32)
            };
            word >> bit & 1 == 1
        })
        .collect();
    let boxed: Box<[bool; COMPRESSION_INPUT_BITS]> = inputs.clone().into_boxed_slice().try_into().unwrap();
    let mut witgen = ProductWitgen::with_inputs_and_capacity(boxed.as_ref(), COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS);
    let digest = compression_circuit(&mut witgen, boxed.as_ref());
    let words: [u32; 8] = std::array::from_fn(|w| (0..32).fold(0u32, |v, b| v | (u32::from(digest[w * 32 + b]) << b)));
    assert_eq!(words, ABC_DIGEST);
    let (f, h, exact) = witgen.into_parts();
    let mut generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
    let symbolic = generator.boxed_inputs::<COMPRESSION_INPUT_BITS>();
    let _ = compression_circuit(&mut generator, symbolic.as_ref());
    let integer = generator.into_matrices();
    let matrices = PreparedConstraintMatrices::new(integer.map_coefficients(|c| bigint_to_fq(&c))).unwrap();
    let products = build_product_mles(&exact, matrices.matrices().a.row_count()).unwrap();
    let assignment = build_assignment_mle::<FqDefault>(&h, matrices.matrices().a.column_count()).unwrap();

    let prover = build_prover(SESSION, INSTANCE);
    let (proof, terminal, bytes, next, prove_ms, verify_ms) = prove_and_check(
        prover,
        |p| build_verifier(SESSION, INSTANCE, p),
        &matrices,
        &products,
        &assignment,
    );
    let empty = transcript::Proof::default();
    let (challenges, d_table) = challenges_and_table(build_verifier(SESSION, INSTANCE, &empty), &matrices, &proof);
    let dump = Dump {
        meta: vec![
            ("kind".into(), "abc".into()),
            ("session".into(), String::from_utf8_lossy(SESSION).into()),
            ("instance".into(), String::from_utf8_lossy(INSTANCE).into()),
            ("input_bits".into(), inputs.len().to_string()),
            ("rows".into(), matrices.matrices().a.row_count().to_string()),
            ("h_len".into(), h.bit_len().to_string()),
            ("f_len".into(), (f.bit_len() + 1).to_string()),
            ("num_row_vars".into(), matrices.num_row_vars().to_string()),
            ("num_column_vars".into(), matrices.num_column_vars().to_string()),
            ("constraint_digest".into(), hex(matrices.digest())),
            ("outer_rounds".into(), proof.outer.sumcheck.round_polynomials.len().to_string()),
            ("inner_rounds".into(), proof.inner.round_polynomials.len().to_string()),
            ("terminal_scale".into(), hex(&fq_bytes(&terminal.scale()))),
            ("next_challenge".into(), hex(&next.to_bytes())),
            ("spartan_len".into(), bytes.len().to_string()),
            ("prove_ms".into(), format!("{prove_ms:.3}")),
            ("verify_ms".into(), format!("{verify_ms:.3}")),
        ],
        files: vec![
            ("inputs.bin", bools_to_bytes(&inputs)),
            ("f.bin", packed_bits(&f)),
            ("h.bin", packed_bits(&h)),
            ("products.bin", products_bytes(&products)),
            ("spartan.bin", bytes),
            ("challenges.bin", challenges),
            ("d.bin", d_table),
        ],
    };
    dump.write(out);
}

fn e2e(circuit: Sha256Circuit, blocks: usize, seed: u64, out: &std::path::Path) {
    let statement = seeded_statement(circuit, blocks, seed);
    let inputs = statement.input();
    // Prepared::new, step by step.
    let mut constraints = ConstraintGenerator::new(statement.input_bits());
    let symbolic: Vec<_> = (0..statement.input_bits()).map(|i| constraints.input(i)).collect();
    statement.synthesize(&mut constraints, &symbolic).expect("synthesize");
    let matrices = PreparedConstraintMatrices::new(constraints.into_matrices().map_coefficients(|c| bigint_to_fq(&c))).unwrap();
    let mut generator = MTransposeGenerator::new(statement.input_bits());
    let map_inputs = generator.take_inputs();
    statement.synthesize(&mut generator, &map_inputs).expect("synthesize map");
    let map = generator.finish();
    assert_eq!(map.h_len(), matrices.matrices().a.column_count());
    let claim_shape = shape_for(map.h_len());
    let committed_shape = shape_for(map.f_len() - 1);
    let params = BitZParams::<Q100>::new(claim_shape, smallest_generator()).expect("params");
    let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3).expect("pcs");
    // Prepared::witness.
    let mut witgen = ProductWitgen::with_inputs(&inputs);
    statement.synthesize(&mut witgen, &inputs).expect("witgen");
    let (f, h, exact) = witgen.into_parts();
    assert_eq!(f.bit_len() + 1, map.f_len());
    assert_eq!(h.bit_len(), map.h_len());
    let products = build_product_mles(&exact, matrices.matrices().a.row_count()).unwrap();
    assert!(products.az.iter().zip(products.bz.iter()).zip(products.cz.iter()).all(|((&a, &b), &c)| a * b == c));
    let assignment = build_assignment_mle::<FqDefault>(&h, map.h_len()).unwrap();
    let committed = pack(&f, committed_shape);
    // Prepared::commit.
    let (root, _data) = pcs.commit(&committed).expect("commit");
    // Prepared::prove's prefix: bind, then Spartan.
    let public = statement.public_bytes();
    let map_digest = map.digest();
    let bind = |t: &mut dyn BindTarget| t.bind(&public, &root.0, &params, &pcs, &map_digest);
    let mut prover = build_prover(E2E_SESSION, statement.domain());
    bind(&mut prover);
    let (proof, terminal, bytes, next, prove_ms, verify_ms) = prove_and_check(
        prover,
        |p| {
            let mut v = build_verifier(E2E_SESSION, statement.domain(), p);
            bind(&mut v);
            v
        },
        &matrices,
        &products,
        &assignment,
    );
    let empty = transcript::Proof::default();
    let (challenges, d_table) = {
        let mut v = build_verifier(E2E_SESSION, statement.domain(), &empty);
        bind(&mut v);
        challenges_and_table(v, &matrices, &proof)
    };
    let claim = opening_claim(&params, &terminal);
    let mut claim_bytes = Vec::new();
    for weights in [claim.row_weights(), claim.column_weights()] {
        claim_bytes.extend_from_slice(&(weights.len() as u64).to_le_bytes());
        for w in weights {
            claim_bytes.extend_from_slice(&fq_bytes(w));
        }
    }
    claim_bytes.extend_from_slice(&fq_bytes(&claim.target()));
    let dump = Dump {
        meta: vec![
            ("kind".into(), "e2e".into()),
            ("circuit".into(), match circuit { Sha256Circuit::Compression => "sha256-compression".into(), Sha256Circuit::Chain => "sha256-chain".into() }),
            ("blocks".into(), blocks.to_string()),
            ("seed".into(), seed.to_string()),
            ("session".into(), String::from_utf8_lossy(E2E_SESSION).into()),
            ("instance".into(), String::from_utf8_lossy(statement.domain()).into()),
            ("input_bits".into(), inputs.len().to_string()),
            ("root".into(), hex(&root.0)),
            ("claim_t".into(), claim_shape.log_rows().to_string()),
            ("claim_s".into(), claim_shape.log_columns().to_string()),
            ("committed_t".into(), committed_shape.log_rows().to_string()),
            ("committed_s".into(), committed_shape.log_columns().to_string()),
            ("q".into(), Q100.to_string()),
            ("generator".into(), hex(&params.generator().to_bytes())),
            ("constraint_digest".into(), hex(matrices.digest())),
            ("map_digest".into(), hex(&map.digest())),
            ("rows".into(), matrices.matrices().a.row_count().to_string()),
            ("h_len".into(), map.h_len().to_string()),
            ("f_len".into(), map.f_len().to_string()),
            ("map_nnz".into(), map.nonzero_count().to_string()),
            ("num_row_vars".into(), matrices.num_row_vars().to_string()),
            ("num_column_vars".into(), matrices.num_column_vars().to_string()),
            ("outer_rounds".into(), proof.outer.sumcheck.round_polynomials.len().to_string()),
            ("inner_rounds".into(), proof.inner.round_polynomials.len().to_string()),
            ("terminal_scale".into(), hex(&fq_bytes(&terminal.scale()))),
            ("next_challenge".into(), hex(&next.to_bytes())),
            ("spartan_len".into(), bytes.len().to_string()),
            ("prove_ms".into(), format!("{prove_ms:.3}")),
            ("verify_ms".into(), format!("{verify_ms:.3}")),
        ],
        files: vec![
            ("public.bin", public.clone()),
            ("inputs.bin", bools_to_bytes(&inputs)),
            ("f.bin", packed_bits(&f)),
            ("h.bin", packed_bits(&h)),
            ("products.bin", products_bytes(&products)),
            ("spartan.bin", bytes),
            ("claim_h.bin", claim_bytes),
            ("challenges.bin", challenges),
            ("d.bin", d_table),
        ],
    };
    dump.write(out);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});
    match args[0].as_str() {
        "abc" => abc(std::path::Path::new(&args[1])),
        circuit => {
            let circuit = match circuit {
                "sha256-compression" => Sha256Circuit::Compression,
                "sha256-chain" => Sha256Circuit::Chain,
                other => panic!("unknown circuit {other}"),
            };
            let blocks: usize = args[1].parse().expect("blocks");
            let seed: u64 = args[2].parse().expect("seed");
            e2e(circuit, blocks, seed, std::path::Path::new(&args[3]));
        }
    }
    let _ = std::io::stdout().flush();
}
