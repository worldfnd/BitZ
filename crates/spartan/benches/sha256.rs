//! Spartan PIOP prover and verifier over the block-aligned SHA-256 R1CS.
//!
//! Run with `cargo bench -p spartan --bench sha256`. The statement is built
//! once per block count, the way the e2e proof builds it, and shared by both
//! benchmarks. Allocation counts are reported alongside timings.

use std::sync::{Arc, Mutex};

use circuit::constraints::ConstraintGenerator;
use circuit::sha256::sha256_block_aligned_circuit;
use circuit::witgen::ProductWitgen;
use divan::{AllocProfiler, Bencher, black_box};
use field::FqDefault;
use poly::{DenseMultilinearExtension, ScaledMleEvaluationClaim};
use spartan::{
    PreparedConstraintMatrices, R1csProductMles, SpartanPiopProof, bigint_to_fq,
    build_assignment_mle, build_product_mles, prove_spartan_piop, verify_spartan_proof,
};
use transcript::{Proof, build_prover, build_verifier};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const SESSION: &[u8] = b"spartan/piop/sha256-block-aligned/v1";
const INSTANCE: &[u8] = b"bench";

/// Block counts benchmarked. 608 is the e2e size: its bit witness fills the
/// 2^22-bit committed shape.
const BLOCKS: &[usize] = &[608];

/// The prover's inputs for one message, built once.
#[derive(Debug, Clone)]
struct Statement {
    matrices: PreparedConstraintMatrices<FqDefault>,
    products: R1csProductMles<FqDefault>,
    assignment: DenseMultilinearExtension<FqDefault>,
}

static STATEMENTS: Mutex<Vec<(usize, Arc<Statement>)>> = Mutex::new(Vec::new());

fn main() {
    BLOCKS.iter().for_each(|&blocks| {
        let _ = statement(blocks);
    });
    divan::main();
}

/// The statement for `blocks`, built on first use.
fn statement(blocks: usize) -> Arc<Statement> {
    let mut cache = STATEMENTS.lock().unwrap();
    if let Some((_, statement)) = cache.iter().find(|(cached, _)| *cached == blocks) {
        return Arc::clone(statement)
    }
    let statement = Arc::new(build(blocks));
    cache.push((blocks, Arc::clone(&statement)));
    statement
}

fn build(blocks: usize) -> Statement {
    // Pool spin-up is not the prover's.
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});

    let message_bits = blocks * 512;
    let message: Vec<bool> = (0..message_bits)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) & 1 == 1)
        .collect();

    // Symbolic replay: M, A, B and C.
    let mut generator = ConstraintGenerator::new(message_bits);
    let symbolic: Vec<_> = (0..message_bits).map(|i| generator.input(i)).collect();
    let _ = sha256_block_aligned_circuit(&mut generator, message_bits, |bit| symbolic[bit].clone());
    let integer_matrices = generator.into_matrices();

    // Concrete replay: f, h = M(1 || f), Ah, Bh and Ch.
    let mut witgen = ProductWitgen::with_inputs(&message);
    let _ = sha256_block_aligned_circuit(&mut witgen, message_bits, |bit| message[bit]);
    let (_witness, assignment_bits, exact_products) = witgen.into_parts();

    // Lower to Q100 and pad to the Boolean domains.
    let matrices = integer_matrices.map_coefficients(|c| bigint_to_fq(&c));
    let products = build_product_mles(&exact_products, matrices.a.row_count()).unwrap();
    let assignment =
        build_assignment_mle::<FqDefault>(&assignment_bits, matrices.a.column_count()).unwrap();
    let matrices = PreparedConstraintMatrices::new(matrices).unwrap();

    let nonzeros: usize = [
        &matrices.matrices().a,
        &matrices.matrices().b,
        &matrices.matrices().c,
    ]
    .iter()
    .flat_map(|matrix| matrix.rows())
    .map(|row| row.entries().len())
    .sum();
    eprintln!(
        "SHA-256 {blocks} blocks: {} r1cs rows -> 2^{}, {} h entries -> 2^{}, {nonzeros} nonzeros",
        matrices.matrices().a.row_count(),
        matrices.num_row_vars(),
        matrices.matrices().a.column_count(),
        matrices.num_column_vars(),
    );

    Statement {
        matrices,
        products,
        assignment,
    }
}

fn prove_once(
    statement: &Statement,
) -> (
    Proof,
    SpartanPiopProof<FqDefault>,
    ScaledMleEvaluationClaim<FqDefault>,
) {
    let mut prover = build_prover(SESSION, INSTANCE);
    let (piop, claim) = prove_spartan_piop(
        &mut prover,
        &statement.matrices,
        statement.products.clone(),
        statement.assignment.clone(),
    )
    .unwrap();
    let proof = prover.finish();
    (proof, piop, claim)
}

fn verify_once(
    statement: &Statement,
    proof: &Proof,
    piop: &SpartanPiopProof<FqDefault>,
) -> ScaledMleEvaluationClaim<FqDefault> {
    let mut verifier = build_verifier(SESSION, INSTANCE, proof);
    let claim = verify_spartan_proof(&mut verifier, &statement.matrices, piop).unwrap();
    verifier.check_eof().unwrap();
    claim
}

/// The prover consumes its tables, so they are cloned per iteration outside
/// the timing; the e2e proof pays for the same clones.
#[divan::bench(args = BLOCKS, sample_count = 10, sample_size = 1)]
fn prove(bencher: Bencher, blocks: usize) {
    let statement = &*statement(blocks);

    // Sanity check
    let (proof, piop, claim) = prove_once(&statement);
    let claim2 = verify_once(&statement, &proof, &piop);
    assert_eq!(claim, claim2, "prover and verifier disagree on the claim");
    claim.nonsuccinct_verify(&statement.assignment).unwrap();

    bencher.bench_local(|| {
        let result = prove_once(&statement);
        black_box(result)
    });
}

#[divan::bench(args = BLOCKS, sample_count = 10, sample_size = 1)]
fn verify(bencher: Bencher, blocks: usize) {
    let statement = &*statement(blocks);

    let (proof, piop, claim) = prove_once(&statement);

    // Sanity check
    let claim2 = verify_once(&statement, &proof, &piop);
    assert_eq!(claim, claim2, "prover and verifier disagree on the claim");
    claim.nonsuccinct_verify(&statement.assignment).unwrap();

    bencher.bench_local(|| {
        let result = verify_once(&statement, &proof, &piop);
        black_box(result)
    });
}
