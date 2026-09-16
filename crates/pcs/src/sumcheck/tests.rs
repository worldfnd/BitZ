use std::sync::OnceLock;

use common::{LinearClaim, Shape};
use field::F128;
use num_traits::ConstZero;
use transcript::{Proof, PublicTranscript, VerifierState, build_prover, build_verifier};

use super::{prove, verify};
use crate::{
    CommitScheme, HashKind, LigeritoProfile, OpeningQuery, Pcs, StatementBinding, VerifyError,
};

const M: usize = 22;
const SESSION: &[u8] = b"pcs-sumcheck-format-test";
const INSTANCE: &[u8] = b"factored-inner-product";
const SET_BITS: [usize; 8] = [0, 63, 64, 127, 128, 255, 256, (1 << M) - 1];
const ROUND_BYTES: usize = 3 * 16;
const EVALUATION_OFFSET: usize = M * ROUND_BYTES;

fn shape() -> Shape {
    Shape::new(8, M - 8).unwrap()
}

fn factor_weight(index: usize) -> F128 {
    let index = index as u64;
    F128::new(
        index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x0123_4567_89ab_cdef,
        index.rotate_left(29) ^ 0xa5a5_5a5a_f0f0_0f0f,
    )
}

fn sparse_witness(packed_len: usize) -> Vec<F128> {
    let mut witness = vec![F128::ZERO; packed_len];
    for index in SET_BITS {
        if index % 128 < 64 {
            witness[index / 128].lo |= 1 << (index % 128);
        } else {
            witness[index / 128].hi |= 1 << (index % 128 - 64);
        }
    }
    witness
}

fn bind_claim(transcript: &mut impl PublicTranscript, claim: &LinearClaim<F128>) {
    transcript.public_message(b"sumcheck-test/claim/v1" as &[u8]);
    transcript.public_message(claim);
}

struct Fixture {
    claim: LinearClaim<F128>,
    proof: Proof,
    point: Vec<F128>,
    evaluation: F128,
}

impl Fixture {
    fn build() -> Self {
        // Shape requires at least 2^22 bits, so this is the smallest valid commitment size.
        let shape = shape();
        let target = SET_BITS
            .into_iter()
            .map(|index| {
                factor_weight(index / shape.rows() + shape.rows())
                    * factor_weight(index % shape.rows())
            })
            .sum();
        let claim = LinearClaim::from_shape(
            &shape,
            (0..shape.rows()).map(factor_weight).collect(),
            (0..shape.columns())
                .map(|column| factor_weight(column + shape.rows()))
                .collect(),
            target,
        )
        .unwrap();
        let witness = sparse_witness(1 << shape.log_packed_len());
        let mut prover = build_prover(SESSION, INSTANCE);
        bind_claim(&mut prover, &claim);
        let reduced = prove(&claim, &witness, &mut prover).unwrap();
        Self {
            claim,
            proof: prover.finish(),
            point: reduced.point,
            evaluation: reduced.target,
        }
    }

    fn verifier<'proof>(&self, proof: &'proof Proof) -> VerifierState<'proof> {
        let mut verifier = build_verifier(SESSION, INSTANCE, proof);
        bind_claim(&mut verifier, &self.claim);
        verifier
    }
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(Fixture::build)
}

#[test]
fn standalone_proof_returns_the_witness_mle_evaluation() {
    let fixture = fixture();
    let mut verifier = fixture.verifier(&fixture.proof);
    let reduced = verify(&fixture.claim, &mut verifier).unwrap();
    assert_eq!(reduced.point, fixture.point);
    assert_eq!(reduced.target, fixture.evaluation);
    let expected = SET_BITS
        .into_iter()
        .map(|index| {
            reduced.point.iter().copied().enumerate().fold(
                F128::from(1u64),
                |product, (coordinate, value)| {
                    product
                        * if (index >> coordinate) & 1 == 1 {
                            value
                        } else {
                            F128::from(1u64) + value
                        }
                },
            )
        })
        .sum::<F128>();
    assert_eq!(reduced.target, expected);
    verifier.check_eof().unwrap();
}

#[test]
fn rejects_truncated_rounds_and_witness_evaluation() {
    let fixture = fixture();
    for length in [
        0,
        16,
        ROUND_BYTES - 1,
        EVALUATION_OFFSET - 1,
        EVALUATION_OFFSET + 15,
    ] {
        let mut proof = fixture.proof.clone();
        proof.narg_string.truncate(length);
        let mut verifier = fixture.verifier(&proof);
        assert_eq!(
            verify(&fixture.claim, &mut verifier).err(),
            Some(VerifyError::MalformedProof),
        );
    }
}

#[test]
fn rejects_changed_round_coefficients_and_witness_evaluation() {
    let fixture = fixture();
    for offset in [
        0,
        16,
        32,
        ROUND_BYTES * (M / 2) + 16,
        ROUND_BYTES * (M - 1) + 32,
        EVALUATION_OFFSET,
    ] {
        let mut proof = fixture.proof.clone();
        proof.narg_string[offset] ^= 1;
        let mut verifier = fixture.verifier(&proof);
        assert_eq!(
            verify(&fixture.claim, &mut verifier).err(),
            Some(VerifyError::VerificationFailed),
        );
    }
}

#[test]
fn zero_weight_factor_still_requires_the_correct_pcs_witness_evaluation() {
    let shape = shape();
    let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let witness = sparse_witness(pcs.packed_len());
    let (commitment, data) = pcs.commit(&witness).unwrap();

    for zero_rows in [true, false] {
        let mut row_weights = (0..shape.rows()).map(factor_weight).collect::<Vec<_>>();
        let mut column_weights = (0..shape.columns())
            .map(|column| factor_weight(column + shape.rows()))
            .collect::<Vec<_>>();
        if zero_rows {
            row_weights.fill(F128::ZERO);
        } else {
            column_weights.fill(F128::ZERO);
        }
        let query = OpeningQuery::InnerProduct {
            claim: LinearClaim::from_shape(&shape, row_weights, column_weights, F128::ZERO)
                .unwrap(),
        };
        let mut prover = build_prover(SESSION, b"zero-inner-product-factor");
        pcs.prove_lin(
            &data,
            witness.clone(),
            &query,
            StatementBinding::Bind,
            &mut prover,
        )
        .unwrap();
        let proof = prover.finish();
        let mut verifier = build_verifier(SESSION, b"zero-inner-product-factor", &proof);
        pcs.verify_lin(&commitment, &query, StatementBinding::Bind, &mut verifier)
            .unwrap();
        verifier.check_eof().unwrap();

        // A zero factor leaves this value unconstrained until the full PCS checks the MLE opening.
        let mut changed_proof = proof;
        changed_proof.narg_string[EVALUATION_OFFSET] ^= 1;
        let mut verifier = build_verifier(SESSION, b"zero-inner-product-factor", &changed_proof);
        assert_eq!(
            pcs.verify_lin(&commitment, &query, StatementBinding::Bind, &mut verifier),
            Err(VerifyError::VerificationFailed),
        );
    }
}
