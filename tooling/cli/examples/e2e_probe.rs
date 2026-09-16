//! Feasibility probe for the end-to-end circuit proof (f2z-pcs parity work).
//!
//! `e2e_probe <sha256-compression|sha256-chain> <blocks> <seed> [--skip-e2e]`
//!
//! Derives `blocks` 64-byte blocks from `seed` (SplitMix64, 8 bytes per
//! output, little-endian), computes the expected chaining value with
//! `sha2::compress256` the way the d6b637e CLI did, and prints what
//! `Prepared::new` derives: the constraint matrices' counts and digest, the
//! materialised `M^T`'s counts and digest, both shapes, the resolved
//! Ligerito ladder (decoded from `Pcs::bitz_statement`), a timing of the
//! transpose and of the Spartan PIOP alone, then the whole e2e through
//! `bitz_cli::benchmark::run` and once more by hand for the proof anatomy.
//!
//! The SHA-256 statement below is `tooling/cli/src/sha256.rs` at d6b637e,
//! verbatim except for the crate path.
use std::time::Instant;

use bitz_cli::end_to_end::{Prepared, CircuitStatement};
use circuit::{
    constraints::ConstraintGenerator,
    matrix_transpose::MTransposeGenerator,
    witgen::ProductWitgen,
};
use common::{BitZParams, Shape, VirtualMap};
use field::{F128, FqDefault, Q100, gf128::smallest_generator};
use pcs::{HashKind, LigeritoProfile, Pcs};
use spartan::{
    PreparedConstraintMatrices, bigint_to_fq, build_assignment_mle, build_product_mles,
    prove_spartan_piop, verify_spartan_proof,
};
use transcript::{build_prover, build_verifier};

#[path = "common/sha256.rs"]
mod sha256;

use sha256::{Sha256Circuit, Sha256Statement};

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

/// `blocks` 64-byte blocks from `seed`, and the chaining value after them.
fn seeded_statement(circuit: Sha256Circuit, blocks: usize, seed: u64) -> Sha256Statement {
    let mut state = seed;
    let mut digest = circuit::sha256::INITIAL_STATE;
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
        initial_state: circuit::sha256::INITIAL_STATE,
        digest,
    }
}

/// `Prepared::new`'s private `shape_for` on the oracle branch.
fn shape_for(bits: usize) -> Shape {
    let padded = bits.checked_next_power_of_two().expect("witness too large");
    let log_bits = (padded.ilog2() as usize).max(common::shape::MIN_LOG_BITS);
    Shape::for_log_bits(log_bits).expect("shape")
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
    let skip_e2e = args.iter().any(|a| a == "--skip-e2e");
    // `--bench-only`: nothing but `benchmark::run`, so the process peak is the e2e's own.
    let bench_only = args.iter().any(|a| a == "--bench-only");
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});
    println!("threads={}", rayon::current_num_threads());

    let statement = seeded_statement(circuit, blocks, seed);
    let inputs = statement.input();
    println!(
        "circuit={:?} blocks={blocks} seed={seed} domain={} public_bytes={} input_bits={} digest_word0={:08x}",
        circuit,
        String::from_utf8_lossy(statement.domain()),
        statement.public_bytes().len(),
        statement.input_bits(),
        statement.digest[0]
    );

    if bench_only {
        let timings = bitz_cli::benchmark::run(statement.clone(), &inputs).expect("benchmark::run");
        println!("benchmark::run: {timings}");
        return;
    }

    // --- what Prepared::new derives, step by step ---
    let started = Instant::now();
    let mut constraints = ConstraintGenerator::new(statement.input_bits());
    let symbolic: Vec<_> = (0..statement.input_bits())
        .map(|i| constraints.input(i))
        .collect();
    statement.synthesize(&mut constraints, &symbolic).expect("synthesize");
    let integer = constraints.into_matrices();
    let t_constraints = started.elapsed();
    let nnz = |m: &circuit::constraints::SparseMatrix<_>| -> usize {
        m.rows().iter().map(|r| r.entries().len()).sum()
    };
    let m_nnz: usize = integer.m.rows().iter().map(|r| r.positions().len()).sum();
    println!(
        "M: rows={} cols={} nnz={}  A: rows={} cols={} nnz={}  B: nnz={}  C: nnz={}  (gen {:.1} ms)",
        integer.m.row_count(),
        integer.m.column_count(),
        m_nnz,
        integer.a.row_count(),
        integer.a.column_count(),
        nnz(&integer.a),
        nnz(&integer.b),
        nnz(&integer.c),
        ms(t_constraints)
    );
    let started = Instant::now();
    let matrices =
        PreparedConstraintMatrices::new(integer.map_coefficients(|c| bigint_to_fq(&c))).expect("prepared");
    println!(
        "constraint_digest={} num_row_vars={} num_column_vars={} ({}) (prepare {:.1} ms)",
        hex(matrices.digest()),
        matrices.num_row_vars(),
        matrices.num_column_vars(),
        matrices.short_debug_info(),
        ms(started.elapsed())
    );

    let started = Instant::now();
    let mut generator = MTransposeGenerator::new(statement.input_bits());
    let map_inputs = generator.take_inputs();
    statement.synthesize(&mut generator, &map_inputs).expect("synthesize map");
    let map = generator.finish();
    println!(
        "map: h_len={} f_len={} nnz={} payload_bytes={} digest={} (gen {:.1} ms)",
        map.h_len(),
        map.f_len(),
        map.nonzero_count(),
        map.payload_bytes(),
        hex(&map.digest()),
        ms(started.elapsed())
    );
    assert_eq!(map.h_len(), matrices.matrices().a.column_count());

    let claim_shape = shape_for(map.h_len());
    let committed_shape = shape_for(map.f_len() - 1);
    println!(
        "claim_shape=({},{}) committed_shape=({},{})",
        claim_shape.log_rows(),
        claim_shape.log_columns(),
        committed_shape.log_rows(),
        committed_shape.log_columns()
    );
    let _params = BitZParams::<Q100>::new(claim_shape, smallest_generator()).expect("params");
    let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3).expect("pcs");
    println!("pcs: bit_len={} packed_len={}", pcs.bit_len(), pcs.packed_len());
    // The resolved configuration, through the only public view of it.
    println!("pcs debug: {pcs:?}");


    // --- the transpose alone, on random weights of the padded virtual length ---
    let mut rng = seed ^ 0xA5A5_5A5A;
    let weights: Vec<F128> = (0..1usize << claim_shape.log_bits())
        .map(|_| F128::new(splitmix64(&mut rng), splitmix64(&mut rng)))
        .collect();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        let transposed = map.transpose(&weights).expect("transpose");
        best = best.min(ms(started.elapsed()));
        std::hint::black_box(transposed);
    }
    println!("transpose: {best:.1} ms (min of 3) over {} weights, {} nnz", weights.len(), map.nonzero_count());
    drop(weights);
    // The two costs `transpose_query` and the PCS binding add around the transpose:
    // the serial row (x) column expansion of the GKR query, and the sponge absorbing
    // the single-column claim (2^committed_bits weights of 16 bytes).
    {
        let rows: Vec<F128> = (0..claim_shape.rows())
            .map(|_| F128::new(splitmix64(&mut rng), splitmix64(&mut rng)))
            .collect();
        let columns: Vec<F128> = (0..claim_shape.columns())
            .map(|_| F128::new(splitmix64(&mut rng), splitmix64(&mut rng)))
            .collect();
        let started = Instant::now();
        let expanded: Vec<F128> = columns
            .iter()
            .flat_map(|column| rows.iter().map(move |row| *row * *column))
            .collect();
        let t_expand = started.elapsed();
        let point: Vec<F128> = (0..claim_shape.log_bits())
            .map(|_| F128::new(splitmix64(&mut rng), splitmix64(&mut rng)))
            .collect();
        let started = Instant::now();
        let table = poly::eq_table(&point);
        let t_eq = started.elapsed();
        std::hint::black_box((&expanded, &table));
        let single = common::LinearClaim::from_shape(
            &Shape::new(committed_shape.log_bits(), 0).unwrap(),
            expanded[..1 << committed_shape.log_bits()].to_vec(),
            vec![F128::from(1u64)],
            F128::from(0u64),
        )
        .unwrap();
        let mut sponge = build_prover(b"e2e-probe/absorb", b"single-column-claim");
        let started = Instant::now();
        sponge.public_message(&single);
        let t_absorb = started.elapsed();
        std::hint::black_box(sponge.finish());
        println!(
            "query costs: row(x)column expansion {:.1} ms (serial, {} products), eq_table({}) {:.1} ms, absorb of the single-column claim ({} B) {:.1} ms",
            ms(t_expand),
            expanded.len(),
            claim_shape.log_bits(),
            ms(t_eq),
            16 * (1usize << committed_shape.log_bits()) + 40,
            ms(t_absorb)
        );
    }

    // --- the Spartan PIOP alone ---
    let started = Instant::now();
    let mut witgen = ProductWitgen::with_inputs(&inputs);
    statement.synthesize(&mut witgen, &inputs).expect("witgen");
    let (f, h, products) = witgen.into_parts();
    println!(
        "witness: f_bits={} h_bits={} (witgen {:.1} ms)",
        f.bit_len(),
        h.bit_len(),
        ms(started.elapsed())
    );
    let products = build_product_mles(&products, matrices.matrices().a.row_count()).expect("products");
    let assignment = build_assignment_mle::<FqDefault>(&h, map.h_len()).expect("assignment");
    let mut prover = build_prover(b"e2e-probe/spartan", statement.domain());
    let started = Instant::now();
    let (spartan_proof, terminal) =
        prove_spartan_piop(&mut prover, &matrices, &products, &assignment).expect("spartan");
    let t_spartan = started.elapsed();
    let proof = prover.finish();
    let mut verifier = build_verifier(b"e2e-probe/spartan", statement.domain(), &proof);
    let started = Instant::now();
    let checked = verify_spartan_proof(&mut verifier, &matrices, &spartan_proof).expect("spartan verify");
    let t_spartan_verify = started.elapsed();
    verifier.check_eof().expect("eof");
    assert_eq!(checked.point(), terminal.point());
    println!(
        "spartan alone: prove {:.1} ms, verify {:.1} ms, outer_rounds={} inner_rounds={} narg_bytes={} (must be 0) point_len={}",
        ms(t_spartan),
        ms(t_spartan_verify),
        spartan_proof.outer.sumcheck.round_polynomials.len(),
        spartan_proof.inner.round_polynomials.len(),
        proof.narg_string.len(),
        terminal.point().len()
    );
    drop(products);
    drop(assignment);
    drop(matrices);
    drop(map);

    if skip_e2e {
        return;
    }

    // --- the canonical e2e timings ---
    let timings = bitz_cli::benchmark::run(statement.clone(), &inputs).expect("benchmark::run");
    println!("benchmark::run: {timings}");

    // --- once more by hand, for the proof anatomy and the verify split ---
    let started = Instant::now();
    let prepared = Prepared::new(statement.clone()).expect("prepared");
    let t_setup = started.elapsed();
    let started = Instant::now();
    let witness = prepared.witness(&inputs).expect("witness");
    let t_witness = started.elapsed();
    let started = Instant::now();
    let data = prepared.commit(&witness).expect("commit");
    let t_commit = started.elapsed();
    let started = Instant::now();
    let proof = prepared.prove(witness, &data).expect("prove");
    let t_prove = started.elapsed();
    let started = Instant::now();
    prepared.verify(&proof).expect("verify");
    let t_verify = started.elapsed();
    let spartan_bytes = 16
        * (4 * proof.spartan.outer.sumcheck.round_polynomials.len()
            + 3
            + 3 * proof.spartan.inner.round_polynomials.len());
    println!(
        "by hand: setup {:.1} witness {:.1} commit {:.1} prove {:.1} verify {:.1} ms; root={} narg={} B hints={} B spartan(out of band)={} B (outer {} rounds x4, [Ah,Bh,Ch], inner {} rounds x3)",
        ms(t_setup),
        ms(t_witness),
        ms(t_commit),
        ms(t_prove),
        ms(t_verify),
        hex(&proof.root.0),
        proof.opening.narg_string.len(),
        proof.opening.hints.len(),
        spartan_bytes,
        proof.spartan.outer.sumcheck.round_polynomials.len(),
        proof.spartan.inner.round_polynomials.len()
    );
}
