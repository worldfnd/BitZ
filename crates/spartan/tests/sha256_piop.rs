use std::array;

use circuit::{
    constraints::ConstraintGenerator,
    sha256::{
        ABC_BLOCK, ABC_DIGEST, COMPRESSION_HINT_BITS, COMPRESSION_INPUT_BITS, INITIAL_STATE,
        compression_circuit,
    },
    witgen::{PackedWitness, ProductWitgen},
};
use num_bigint::BigInt;
use spartan::{
    PreparedConstraintMatrices, bigint_to_fq, build_assignment_mle, build_product_mles,
    prove_spartan_piop, verify_spartan_with_mle_claim,
};
use transcript::{build_prover, build_verifier};

const SESSION: &[u8] = b"spartan/piop/sha256-compression/v1";
const INSTANCE: &[u8] = b"abc-single-compression";

#[test]
fn sha256_compression_verifies_through_spartan_piop() {
    let inputs = abc_compression_input();

    // Concrete replay: generate f, h = M(1 || f), Ah, Bh, and Ch.
    let mut witgen = ProductWitgen::with_inputs_and_capacity(
        inputs.as_ref(),
        COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS,
    );
    let digest = compression_circuit(&mut witgen, inputs.as_ref());
    assert_eq!(words_from_le_bits(&digest), ABC_DIGEST);
    let (boolean_witness, recorded_assignment, exact_products) = witgen.into_parts();

    // Symbolic replay: generate M, A, B, and C from the identical circuit.
    let mut generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
    let symbolic_inputs = generator.boxed_inputs::<COMPRESSION_INPUT_BITS>();
    let _ = compression_circuit(&mut generator, symbolic_inputs.as_ref());
    let integer_matrices = generator.into_matrices();

    assert_eq!(integer_matrices.m.row_count(), 20_457);
    assert_eq!(integer_matrices.m.column_count(), 7_145);
    assert_eq!(integer_matrices.a.row_count(), 184);
    assert_eq!(integer_matrices.b.row_count(), 184);
    assert_eq!(integer_matrices.c.row_count(), 184);
    assert_eq!(integer_matrices.a.column_count(), 20_457);
    assert_eq!(recorded_assignment.bit_len(), 20_457);

    // Check exact integer satisfaction and independently reconstruct h before
    // consuming the temporary integer matrices at the Q100 setup boundary.
    integer_matrices.check_witness(&boolean_witness).unwrap();
    let recomputed_assignment = integer_matrices.integer_witness(&boolean_witness).unwrap();
    assert_assignment_matches(&recomputed_assignment, &recorded_assignment);

    let matrices = integer_matrices.map_coefficients(|coefficient| bigint_to_fq(&coefficient));
    let products = build_product_mles(&exact_products, matrices.a.row_count()).unwrap();
    let assignment = build_assignment_mle(&recorded_assignment, matrices.a.column_count()).unwrap();
    let matrices = PreparedConstraintMatrices::new(matrices).unwrap();

    // 184 rows pad to 2^8; 20,457 assignment entries pad to 2^15.
    assert_eq!(products.az.num_vars(), 8);
    assert_eq!(assignment.num_vars(), 15);

    let mut prover = build_prover(SESSION, INSTANCE);
    let (proof, claim) =
        prove_spartan_piop(&mut prover, &matrices, &products, &assignment).unwrap();
    assert_eq!(proof.outer.sumcheck.round_polynomials.len(), 8);
    assert_eq!(proof.inner.round_polynomials.len(), 15);
    let transcript_proof = prover.finish();

    let mut verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
    verify_spartan_with_mle_claim(
        &mut verifier,
        &matrices,
        &proof,
        &claim,
        &recorded_assignment,
    )
    .unwrap();
    verifier.check_eof().unwrap();
}

fn abc_compression_input() -> Box<[bool; COMPRESSION_INPUT_BITS]> {
    let bits: Box<[bool]> = (0..COMPRESSION_INPUT_BITS)
        .map(|index| {
            let (word, bit) = if index < 512 {
                (ABC_BLOCK[index / 32], index % 32)
            } else {
                let state_index = index - 512;
                (INITIAL_STATE[state_index / 32], state_index % 32)
            };
            word >> bit & 1 == 1
        })
        .collect();
    bits.try_into()
        .unwrap_or_else(|_| unreachable!("fixed compression input length"))
}

fn words_from_le_bits(bits: &[bool; 256]) -> [u32; 8] {
    array::from_fn(|word| {
        (0..32).fold(0_u32, |value, bit| {
            value | (u32::from(bits[word * 32 + bit]) << bit)
        })
    })
}

fn assert_assignment_matches(computed: &[BigInt], recorded: &PackedWitness) {
    assert_eq!(computed.len(), recorded.bit_len());
    for (index, expected) in computed.iter().enumerate() {
        assert_eq!(
            expected,
            &BigInt::from(recorded.bit(index)),
            "assignment differs at index {index}",
        );
    }
}
