//! Spartan PIOP prover and verifier over the block-aligned SHA-256 R1CS.
//!
//! Run with `cargo bench -p spartan --bench sha256`. The instance-witness pair is built
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

/// An R1CS instance with a witness satisfying it.
#[derive(Debug, Clone)]
struct R1csInstanceWitness {
    instance: PreparedConstraintMatrices<FqDefault>,
    witness: DenseMultilinearExtension<FqDefault>,
    /// Holds precomputed `Az`, `Bz` and `Cz` tables.
    products: R1csProductMles<FqDefault>,
}

static R1CS_INSTANCE_WITNESSES: Mutex<Vec<(usize, Arc<R1csInstanceWitness>)>> =
    Mutex::new(Vec::new());

fn main() {
    BLOCKS.iter().for_each(|&blocks| {
        let _ = r1cs_instance_witness(blocks);
    });
    divan::main();
}

/// The instance-witness pair for `blocks`, built on first use.
fn r1cs_instance_witness(blocks: usize) -> Arc<R1csInstanceWitness> {
    let mut cache = R1CS_INSTANCE_WITNESSES.lock().unwrap();
    if let Some((_, r1cs)) = cache.iter().find(|(cached, _)| *cached == blocks) {
        return Arc::clone(r1cs);
    }
    let r1cs = Arc::new(build(blocks));
    cache.push((blocks, Arc::clone(&r1cs)));
    r1cs
}

fn build(blocks: usize) -> R1csInstanceWitness {
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

    eprintln!("SHA-256 {blocks} blocks: {}", matrices.short_debug_info());

    R1csInstanceWitness {
        instance: matrices,
        witness: assignment,
        products,
    }
}

fn spartan_prove(
    r1cs: &R1csInstanceWitness,
) -> (
    Proof,
    SpartanPiopProof<FqDefault>,
    ScaledMleEvaluationClaim<FqDefault>,
) {
    let mut prover = build_prover(SESSION, INSTANCE);
    let (piop, claim) =
        prove_spartan_piop(&mut prover, &r1cs.instance, &r1cs.products, &r1cs.witness).unwrap();
    let proof = prover.finish();
    (proof, piop, claim)
}

fn spartan_verify(
    r1cs_instance: &PreparedConstraintMatrices<FqDefault>,
    proof: &Proof,
    piop: &SpartanPiopProof<FqDefault>,
) -> ScaledMleEvaluationClaim<FqDefault> {
    let mut verifier = build_verifier(SESSION, INSTANCE, proof);
    let claim = verify_spartan_proof(&mut verifier, r1cs_instance, piop).unwrap();
    verifier.check_eof().unwrap();
    claim
}

/// The prover consumes its tables, so they are cloned per iteration outside
/// the timing; the e2e proof pays for the same clones.
#[divan::bench(args = BLOCKS, sample_count = 10, sample_size = 1)]
fn prove(bencher: Bencher, blocks: usize) {
    let r1cs = &*r1cs_instance_witness(blocks);

    // Sanity check
    let (proof, piop, claim) = spartan_prove(r1cs);
    let claim2 = spartan_verify(&r1cs.instance, &proof, &piop);
    assert_eq!(claim, claim2, "prover and verifier disagree on the claim");
    claim.nonsuccinct_verify(&r1cs.witness).unwrap();

    bencher.bench_local(|| {
        let result = spartan_prove(r1cs);
        black_box(result)
    });
}

#[divan::bench(args = BLOCKS, sample_count = 10, sample_size = 1)]
fn verify(bencher: Bencher, blocks: usize) {
    let r1cs = &*r1cs_instance_witness(blocks);

    let (proof, piop, claim) = spartan_prove(r1cs);

    // Sanity check
    let claim2 = spartan_verify(&r1cs.instance, &proof, &piop);
    assert_eq!(claim, claim2, "prover and verifier disagree on the claim");
    claim.nonsuccinct_verify(&r1cs.witness).unwrap();

    bencher.bench_local(|| {
        let result = spartan_verify(&r1cs.instance, &proof, &piop);
        black_box(result)
    });
}
