use std::sync::OnceLock;

use common::{LinearClaim, Shape};
use field::F128;
use num_traits::ConstZero;
use pcs::{
    CommitScheme, Commitment, OpeningQuery, Pcs, ProveError, Root, SecurityLevel, StatementBinding,
    VerifyError,
};
use transcript::{Proof, PublicTranscript, VerifierState, build_prover, build_verifier};

const M: usize = 22;
const SINGLETON: usize = (1 << 21) | (1 << 7) | 0b101_0101;
const SESSION: &[u8] = b"pcs-interface-test";
const INSTANCE: &[u8] = b"m22-singleton-opening";
const INNER_PRODUCT_INSTANCE: &[u8] = b"m22-factored-inner-product";
const INNER_PRODUCT_SET_BITS: [usize; 8] = [0, 63, 64, 127, 128, 255, 256, (1 << M) - 1];

fn shape() -> Shape {
    Shape::new(7, M - 7).unwrap()
}

fn inner_product_shape() -> Shape {
    Shape::new(8, M - 8).unwrap()
}

struct RealFixture {
    pcs: Pcs,
    commitment: Root,
    query: OpeningQuery,
    proof: Proof,
}

impl RealFixture {
    fn build(security: SecurityLevel) -> Self {
        let pcs = Pcs::new(&shape(), security).unwrap();
        // One nonzero bit gives the expected MLE value a simple independent formula.
        let mut packed_witness = vec![F128::ZERO; pcs.packed_len()];
        let packed_index = SINGLETON / 128;
        let bit_index = SINGLETON % 128;
        if bit_index < 64 {
            packed_witness[packed_index].lo |= 1 << bit_index;
        } else {
            packed_witness[packed_index].hi |= 1 << (bit_index - 64);
        }

        let point = (0..M)
            .map(|coordinate| F128::from(coordinate as u64 + 2))
            .collect::<Vec<_>>();
        let query = OpeningQuery::Mle {
            target: singleton_target(&point, SINGLETON),
            point,
        };

        let mut prover = build_prover(SESSION, INSTANCE);
        let (commitment, data) = pcs.commit(&packed_witness, &mut prover).unwrap();

        pcs.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        )
        .unwrap();

        Self {
            pcs,
            commitment,
            query,
            proof: prover.finish(),
        }
    }
}

fn verify_opening(
    pcs: &Pcs,
    root: &Root,
    query: &OpeningQuery,
    binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), VerifyError> {
    let data = pcs.receive_commitment(*root, transcript)?;
    pcs.verify_lin(&data, query, binding, transcript)
}

fn fixture() -> &'static RealFixture {
    static FIXTURE: OnceLock<RealFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| RealFixture::build(SecurityLevel::Bits100))
}

fn singleton_target(point: &[F128], index: usize) -> F128 {
    let one = F128::from(1u64);
    point
        .iter()
        .copied()
        .enumerate()
        .fold(one, |target, (coordinate, value)| {
            let weight = if ((index >> coordinate) & 1) == 1 {
                value
            } else {
                one + value
            };
            target * weight
        })
}

#[test]
fn dynamic_shapes_open_nonzero_witnesses_across_ladder_shapes() {
    use pcs::SecurityLevel::{Bits100, Bits128};
    // These sizes exercise final folds of one, two, and three variables.
    // Size 23 also adds a recursive level.
    for m in 20..=23 {
        let shape = Shape::new(7, m - 7).unwrap();
        for level in [Bits100, Bits128] {
            let pcs = Pcs::new(&shape, level).unwrap();
            let index = (1usize << m) - 1;
            let mut witness = vec![F128::ZERO; pcs.packed_len()];
            witness[index / 128].hi = 1 << 63;
            let point: Vec<_> = (0..m).map(|i| F128::from(i as u64 + 2)).collect();
            let query = OpeningQuery::Mle {
                target: singleton_target(&point, index),
                point,
            };
            let mut prover = build_prover(SESSION, b"dynamic");
            let (root, data) = pcs.commit(&witness, &mut prover).unwrap();

            pcs.prove_lin(&data, witness, &query, StatementBinding::Bind, &mut prover)
                .unwrap();
            let continuation = prover.verifier_message::<F128>();
            let proof = prover.finish();
            let mut verifier = build_verifier(SESSION, b"dynamic", &proof);
            verify_opening(&pcs, &root, &query, StatementBinding::Bind, &mut verifier).unwrap();
            assert_eq!(verifier.verifier_message::<F128>(), continuation);
            verifier.check_eof().unwrap();

            let mut truncated = proof.clone();
            truncated.narg_string.pop();
            let mut verifier = build_verifier(SESSION, b"dynamic", &truncated);
            assert!(
                verify_opening(&pcs, &root, &query, StatementBinding::Bind, &mut verifier).is_err()
            );
        }
    }
}

fn factor_weight(index: usize) -> F128 {
    let index = index as u64;
    F128::new(
        index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x0123_4567_89ab_cdef,
        index.rotate_left(29) ^ 0xa5a5_5a5a_f0f0_0f0f,
    )
}

fn factored_query(shape: &Shape, target: F128) -> OpeningQuery {
    OpeningQuery::InnerProduct {
        claim: LinearClaim::from_shape(
            shape,
            (0..shape.rows()).map(factor_weight).collect(),
            (0..shape.columns())
                .map(|column| factor_weight(column + shape.rows()))
                .collect(),
            target,
        )
        .unwrap(),
    }
}

fn inner_product_witness(packed_len: usize) -> Vec<F128> {
    let mut witness = vec![F128::ZERO; packed_len];
    for index in INNER_PRODUCT_SET_BITS {
        if index % 128 < 64 {
            witness[index / 128].lo |= 1 << (index % 128);
        } else {
            witness[index / 128].hi |= 1 << (index % 128 - 64);
        }
    }
    witness
}

fn inner_product_target(shape: &Shape) -> F128 {
    INNER_PRODUCT_SET_BITS
        .into_iter()
        .map(|index| {
            factor_weight(index / shape.rows() + shape.rows()) * factor_weight(index % shape.rows())
        })
        .sum()
}

struct InnerProductFixture {
    pcs: Pcs,
    commitment: Root,
    query: OpeningQuery,
    proofs: [Proof; 2],
}

impl InnerProductFixture {
    fn build(security: SecurityLevel) -> Self {
        let shape = inner_product_shape();
        let pcs = Pcs::new(&shape, security).unwrap();
        let witness = inner_product_witness(pcs.packed_len());
        let target = inner_product_target(&shape);
        assert_ne!(target, F128::ZERO);
        let query = factored_query(&shape, target);
        let mut commitment = Root([0; 32]);
        let proofs = [StatementBinding::Bind, StatementBinding::AlreadyBound].map(|binding| {
            let mut prover = build_prover(SESSION, INNER_PRODUCT_INSTANCE);
            let (root, data) = pcs.commit(&witness, &mut prover).unwrap();
            commitment = root;
            if binding == StatementBinding::AlreadyBound {
                bind_outer_inner_product_statement(&mut prover, &pcs, &commitment, &query);
            }
            pcs.prove_lin(&data, witness.clone(), &query, binding, &mut prover)
                .unwrap();
            prover.finish()
        });
        Self {
            pcs,
            commitment,
            query,
            proofs,
        }
    }

    fn proof(&self, binding: StatementBinding) -> &Proof {
        &self.proofs[usize::from(binding == StatementBinding::AlreadyBound)]
    }

    fn verifier<'proof>(
        &self,
        commitment: &Root,
        query: &OpeningQuery,
        proof: &'proof Proof,
        binding: StatementBinding,
    ) -> Result<(Commitment, VerifierState<'proof>), VerifyError> {
        let mut verifier = build_verifier(SESSION, INNER_PRODUCT_INSTANCE, proof);
        let data = self.pcs.receive_commitment(*commitment, &mut verifier)?;
        if binding == StatementBinding::AlreadyBound {
            bind_outer_inner_product_statement(&mut verifier, &self.pcs, commitment, query);
        }
        Ok((data, verifier))
    }

    fn verify(
        &self,
        commitment: &Root,
        query: &OpeningQuery,
        proof: &Proof,
        binding: StatementBinding,
    ) -> Result<(), VerifyError> {
        let (data, mut verifier) = self.verifier(commitment, query, proof, binding)?;
        self.pcs.verify_lin(&data, query, binding, &mut verifier)
    }
}

fn inner_product_fixture(security: SecurityLevel) -> &'static InnerProductFixture {
    static BITS100: OnceLock<InnerProductFixture> = OnceLock::new();
    static BITS128: OnceLock<InnerProductFixture> = OnceLock::new();
    let fixture = match security {
        SecurityLevel::Bits100 => &BITS100,
        SecurityLevel::Bits128 => &BITS128,
    };
    fixture.get_or_init(|| InnerProductFixture::build(security))
}

fn bind_outer_inner_product_statement(
    transcript: &mut impl PublicTranscript,
    pcs: &Pcs,
    commitment: &Root,
    query: &OpeningQuery,
) {
    let OpeningQuery::InnerProduct { claim } = query else {
        panic!("expected an inner-product query");
    };
    transcript.public_message(b"outer/pcs-inner-product/v2" as &[u8]);
    transcript.public_message(pcs);
    transcript.public_message(&commitment.0);
    transcript.public_message(claim);
}

fn bind_outer_statement(
    transcript: &mut impl PublicTranscript,
    pcs: &Pcs,
    commitment: &Root,
    query: &OpeningQuery,
) {
    let OpeningQuery::Mle { point, target } = query else {
        panic!("expected an MLE query");
    };
    transcript.public_message(b"outer/pcs-opening/v1" as &[u8]);
    transcript.public_message(pcs);
    transcript.public_message(&commitment.0);
    transcript.public_message(&(point.len() as u64));
    for coordinate in point {
        transcript.public_message(coordinate);
    }
    transcript.public_message(target);
}

#[test]
fn real_pcs_opening_round_trip_succeeds() {
    let fixture = fixture();
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);

    verify_opening(
        &fixture.pcs,
        &fixture.commitment,
        &fixture.query,
        StatementBinding::Bind,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn factored_inner_product_round_trip_succeeds_for_both_security_levels_and_bindings() {
    for security in [SecurityLevel::Bits100, SecurityLevel::Bits128] {
        let fixture = inner_product_fixture(security);
        for binding in [StatementBinding::Bind, StatementBinding::AlreadyBound] {
            let (data, mut verifier) = fixture
                .verifier(
                    &fixture.commitment,
                    &fixture.query,
                    fixture.proof(binding),
                    binding,
                )
                .unwrap();
            fixture
                .pcs
                .verify_lin(&data, &fixture.query, binding, &mut verifier)
                .unwrap();
            verifier.check_eof().unwrap();
        }
    }
}

#[test]
fn factored_inner_product_prover_rejects_a_false_target() {
    let shape = inner_product_shape();
    let pcs = Pcs::new(&shape, SecurityLevel::Bits100).unwrap();
    let witness = inner_product_witness(pcs.packed_len());
    let query = factored_query(&shape, inner_product_target(&shape) + F128::from(1u64));

    for binding in [StatementBinding::Bind, StatementBinding::AlreadyBound] {
        let mut prover = build_prover(SESSION, INNER_PRODUCT_INSTANCE);
        let (commitment, data) = pcs.commit(&witness, &mut prover).unwrap();
        if binding == StatementBinding::AlreadyBound {
            bind_outer_inner_product_statement(&mut prover, &pcs, &commitment, &query);
        }
        assert_eq!(
            pcs.prove_lin(&data, witness.clone(), &query, binding, &mut prover),
            Err(ProveError::InvalidClaim),
        );
    }
}

#[test]
fn factored_inner_product_rejects_statement_mutations() {
    let fixture = inner_product_fixture(SecurityLevel::Bits100);
    let OpeningQuery::InnerProduct { claim } = &fixture.query else {
        unreachable!();
    };

    for binding in [StatementBinding::Bind, StatementBinding::AlreadyBound] {
        for mutation in 0..3 {
            let mut row_weights = claim.row_weights().to_vec();
            let mut column_weights = claim.column_weights().to_vec();
            let mut target = claim.target();
            match mutation {
                0 => row_weights[2] += F128::from(1u64),
                1 => column_weights[2] += F128::from(1u64),
                2 => target += F128::from(1u64),
                _ => unreachable!(),
            }
            let query = OpeningQuery::InnerProduct {
                claim: LinearClaim::from_shape(
                    &inner_product_shape(),
                    row_weights,
                    column_weights,
                    target,
                )
                .unwrap(),
            };
            assert_eq!(
                fixture.verify(&fixture.commitment, &query, fixture.proof(binding), binding),
                Err(VerifyError::VerificationFailed),
            );
        }

        let mut commitment = fixture.commitment;
        commitment.0[0] ^= 1;
        assert_eq!(
            fixture.verify(&commitment, &fixture.query, fixture.proof(binding), binding),
            Err(VerifyError::VerificationFailed),
        );
    }
}

#[test]
fn factored_inner_product_requires_complete_transcript_consumption() {
    let fixture = inner_product_fixture(SecurityLevel::Bits100);
    for binding in [StatementBinding::Bind, StatementBinding::AlreadyBound] {
        for append_hint in [false, true] {
            let mut proof = fixture.proof(binding).clone();
            if append_hint {
                proof.hints.push(0);
            } else {
                proof.narg_string.push(0);
            }
            let (data, mut verifier) = fixture
                .verifier(&fixture.commitment, &fixture.query, &proof, binding)
                .unwrap();
            fixture
                .pcs
                .verify_lin(&data, &fixture.query, binding, &mut verifier)
                .unwrap();
            assert!(verifier.check_eof().is_err());
        }
    }
}

#[test]
fn factored_inner_product_rejects_wrong_weight_lengths() {
    let pcs = Pcs::new(&shape(), SecurityLevel::Bits128).unwrap();
    let packed_witness = vec![F128::ZERO; pcs.packed_len()];
    let mut prover = build_prover(SESSION, b"wrong-inner-product-weight-count");
    let (commitment, data) = pcs.commit(&packed_witness, &mut prover).unwrap();
    let larger_shape = Shape::new(8, M - 7).unwrap();
    let query = factored_query(&larger_shape, F128::ZERO);

    assert_eq!(
        pcs.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::WeightLengthMismatch),
    );

    let proof = Proof::default();
    let mut verifier = build_verifier(SESSION, b"wrong-inner-product-weight-count", &proof);
    assert_eq!(
        verify_opening(
            &pcs,
            &commitment,
            &query,
            StatementBinding::Bind,
            &mut verifier
        ),
        Err(VerifyError::WeightLengthMismatch),
    );
}

#[test]
fn factored_inner_product_rejects_invalid_prover_inputs_before_sumcheck() {
    let shape = inner_product_shape();
    let pcs = Pcs::new(&shape, SecurityLevel::Bits100).unwrap();
    let packed_witness = vec![F128::ZERO; pcs.packed_len()];
    let mut prover = build_prover(SESSION, b"inner-product-wrong-packed-length");
    let (_, data) = pcs.commit(&packed_witness, &mut prover).unwrap();
    let query = factored_query(&shape, F128::ZERO);

    let mut short_witness = packed_witness.clone();
    short_witness.pop();

    assert_eq!(
        pcs.prove_lin(
            &data,
            short_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::PackedWitnessLengthMismatch),
    );

    let other = Pcs::new(&shape, SecurityLevel::Bits128).unwrap();
    let mut prover = build_prover(SESSION, b"inner-product-mismatched-parameters");
    assert_eq!(
        other.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::ProverDataMismatch),
    );
}

#[test]
fn bits128_opening_round_trip_exercises_pow() {
    let fixture = RealFixture::build(SecurityLevel::Bits128);
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);

    verify_opening(
        &fixture.pcs,
        &fixture.commitment,
        &fixture.query,
        StatementBinding::Bind,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn real_pcs_accepts_an_already_bound_statement() {
    let pcs = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
    let packed_witness = vec![F128::ZERO; pcs.packed_len()];
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };
    let mut prover = build_prover(SESSION, b"already-bound");
    let (commitment, data) = pcs.commit(&packed_witness, &mut prover).unwrap();

    bind_outer_statement(&mut prover, &pcs, &commitment, &query);
    pcs.prove_lin(
        &data,
        packed_witness,
        &query,
        StatementBinding::AlreadyBound,
        &mut prover,
    )
    .unwrap();
    let proof = prover.finish();

    let mut verifier = build_verifier(SESSION, b"already-bound", &proof);
    let data = pcs.receive_commitment(commitment, &mut verifier).unwrap();
    bind_outer_statement(&mut verifier, &pcs, &commitment, &query);
    pcs.verify_lin(&data, &query, StatementBinding::AlreadyBound, &mut verifier)
        .unwrap();
    verifier.check_eof().unwrap();

    let mut mismatched_verifier = build_verifier(SESSION, b"already-bound", &proof);
    let data = pcs
        .receive_commitment(commitment, &mut mismatched_verifier)
        .unwrap();
    bind_outer_statement(&mut mismatched_verifier, &pcs, &commitment, &query);
    assert!(
        pcs.verify_lin(
            &data,
            &query,
            StatementBinding::Bind,
            &mut mismatched_verifier,
        )
        .is_err()
    );
}

#[test]
fn real_pcs_rejects_point_length_mismatches() {
    let pcs = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
    let packed_witness = vec![F128::ZERO; pcs.packed_len()];
    let mut prover = build_prover(SESSION, b"wrong-prover-point");
    let (commitment, data) = pcs.commit(&packed_witness, &mut prover).unwrap();
    let short_query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M - 1],
        target: F128::from(0u64),
    };

    assert_eq!(
        pcs.prove_lin(
            &data,
            packed_witness,
            &short_query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::PointLengthMismatch)
    );

    let long_query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M + 1],
        target: F128::from(0u64),
    };
    let mut initial = build_prover(SESSION, b"wrong-verifier-point");
    let packed = vec![F128::ZERO; pcs.packed_len()];
    pcs.commit(&packed, &mut initial).unwrap();
    let proof = initial.finish();
    let mut verifier = build_verifier(SESSION, b"wrong-verifier-point", &proof);
    assert_eq!(
        verify_opening(
            &pcs,
            &commitment,
            &long_query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::PointLengthMismatch)
    );
}

#[test]
fn real_pcs_rejects_packed_witness_length_mismatches_during_opening() {
    let pcs = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
    let mut packed_witness = vec![F128::ZERO; pcs.packed_len()];
    let mut prover = build_prover(SESSION, b"wrong-packed-length");
    let (_, data) = pcs.commit(&packed_witness, &mut prover).unwrap();
    packed_witness.pop();
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };

    assert_eq!(
        pcs.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::PackedWitnessLengthMismatch)
    );
}

#[test]
fn real_pcs_rejects_mismatched_prover_parameters() {
    let source = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
    let packed_witness = vec![F128::ZERO; source.packed_len()];
    let mut prover = build_prover(SESSION, b"mismatched-parameters");
    let (_, data) = source.commit(&packed_witness, &mut prover).unwrap();
    let other = Pcs::new(&shape(), SecurityLevel::Bits128).unwrap();
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };

    assert_eq!(
        other.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::ProverDataMismatch),
    );
}

#[test]
fn real_pcs_prover_rejects_a_false_evaluation_without_consuming_prover_data() {
    let pcs = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
    let packed_witness = vec![F128::ZERO; pcs.packed_len()];
    let mut prover = build_prover(SESSION, b"false-evaluation");
    let (_, data) = pcs.commit(&packed_witness, &mut prover).unwrap();
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(1u64),
    };

    let codeword_len = data.codeword_len();

    assert_eq!(
        pcs.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::InvalidClaim)
    );
    assert_eq!(data.codeword_len(), codeword_len);
}

#[test]
fn real_pcs_rejects_an_opening_for_a_different_packed_witness() {
    let pcs = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
    let packed_witness = vec![F128::ZERO; pcs.packed_len()];
    let mut prover = build_prover(SESSION, b"different-packed-witness");
    let (commitment, data) = pcs.commit(&packed_witness, &mut prover).unwrap();
    let mut different_witness = packed_witness;
    different_witness[0].lo = 1;
    let point = (0..M)
        .map(|coordinate| F128::from(coordinate as u64 + 2))
        .collect::<Vec<_>>();
    let query = OpeningQuery::Mle {
        target: singleton_target(&point, 0),
        point,
    };

    pcs.prove_lin(
        &data,
        different_witness,
        &query,
        StatementBinding::Bind,
        &mut prover,
    )
    .unwrap();
    let proof = prover.finish();
    let mut verifier = build_verifier(SESSION, b"different-packed-witness", &proof);

    assert_eq!(
        verify_opening(
            &pcs,
            &commitment,
            &query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::VerificationFailed)
    );
}

#[test]
fn real_pcs_rejects_statement_mutations() {
    let fixture = fixture();

    let mut changed_query = fixture.query.clone();
    let OpeningQuery::Mle { target, .. } = &mut changed_query else {
        unreachable!();
    };
    *target += F128::from(1u64);
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);
    assert_eq!(
        verify_opening(
            &fixture.pcs,
            &fixture.commitment,
            &changed_query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::VerificationFailed)
    );

    let mut changed_root = fixture.commitment.0;
    changed_root[0] ^= 1;
    let changed_commitment = Root(changed_root);
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);
    assert_eq!(
        verify_opening(
            &fixture.pcs,
            &changed_commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::VerificationFailed)
    );
}

#[test]
fn real_pcs_rejects_malformed_transcript_streams() {
    let fixture = fixture();

    let mut truncated_stream = fixture.proof.clone();
    truncated_stream.narg_string.truncate(1);
    let mut verifier = build_verifier(SESSION, INSTANCE, &truncated_stream);
    assert_eq!(
        verify_opening(
            &fixture.pcs,
            &fixture.commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::MalformedProof)
    );

    let mut truncated_hint = fixture.proof.clone();
    truncated_hint.hints.pop();
    let mut verifier = build_verifier(SESSION, INSTANCE, &truncated_hint);
    assert_eq!(
        verify_opening(
            &fixture.pcs,
            &fixture.commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::MalformedProof)
    );

    let mut changed_hint = fixture.proof.clone();
    *changed_hint.hints.last_mut().unwrap() ^= 1;
    let mut verifier = build_verifier(SESSION, INSTANCE, &changed_hint);
    assert_eq!(
        verify_opening(
            &fixture.pcs,
            &fixture.commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::MalformedProof)
    );
}

#[test]
fn real_pcs_requires_complete_transcript_consumption() {
    let fixture = fixture();

    let mut trailing_narg = fixture.proof.clone();
    trailing_narg.narg_string.push(0);
    let mut verifier = build_verifier(SESSION, INSTANCE, &trailing_narg);
    verify_opening(
        &fixture.pcs,
        &fixture.commitment,
        &fixture.query,
        StatementBinding::Bind,
        &mut verifier,
    )
    .unwrap();
    assert!(verifier.check_eof().is_err());

    let mut trailing_hint = fixture.proof.clone();
    trailing_hint.hints.push(0);
    let mut verifier = build_verifier(SESSION, INSTANCE, &trailing_hint);

    verify_opening(
        &fixture.pcs,
        &fixture.commitment,
        &fixture.query,
        StatementBinding::Bind,
        &mut verifier,
    )
    .unwrap();
    assert!(verifier.check_eof().is_err());
}
