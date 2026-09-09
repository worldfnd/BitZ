use std::sync::OnceLock;

use common::Shape;
use field::F128;
use pcs::{
    CommitScheme, HashKind, LigeritoProfile, OpeningQuery, Pcs, ProveError, Root, StatementBinding,
    VerifyError,
};
use poly::eq_table;
use transcript::{Proof, PublicTranscript, build_prover, build_verifier};

const M: usize = 22;
const SINGLETON: usize = (1 << 21) | (1 << 7) | 0b101_0101;
const SESSION: &[u8] = b"pcs-interface-test";
const INSTANCE: &[u8] = b"m22-singleton-opening";
const INNER_PRODUCT_INSTANCE: &[u8] = b"m22-arbitrary-inner-product";
const INNER_PRODUCT_SET_BITS: [usize; 8] = [0, 1, 63, 64, 127, 128, SINGLETON, (1 << M) - 1];

fn shape() -> Shape {
    Shape::new(7, M - 7).unwrap()
}

struct RealFixture {
    pcs: Pcs,
    commitment: Root,
    query: OpeningQuery,
    proof: Proof,
}

impl RealFixture {
    fn build(profile: LigeritoProfile) -> Self {
        let pcs = Pcs::new(&shape(), profile, HashKind::Blake3).unwrap();
        // One nonzero bit gives the expected MLE value a simple independent formula.
        let mut packed_witness = vec![F128::default(); pcs.packed_len()];
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

        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let mut prover = build_prover(SESSION, INSTANCE);
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

fn fixture() -> &'static RealFixture {
    static FIXTURE: OnceLock<RealFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| RealFixture::build(LigeritoProfile::Fast))
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

struct InnerProductFixture {
    pcs: Pcs,
    commitment: Root,
    query: OpeningQuery,
    proof: Proof,
}

impl InnerProductFixture {
    fn build() -> Self {
        let pcs = Pcs::new(&shape(), LigeritoProfile::Secure, HashKind::Blake3).unwrap();
        let mut packed_witness = vec![F128::default(); pcs.packed_len()];
        for index in INNER_PRODUCT_SET_BITS {
            set_packed_bit(&mut packed_witness, index);
        }
        let weights = (0..pcs.bit_len()).map(arbitrary_weight).collect::<Vec<_>>();
        let target = INNER_PRODUCT_SET_BITS
            .into_iter()
            .map(|index| weights[index])
            .sum();
        let query = OpeningQuery::InnerProduct { weights, target };
        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let mut prover = build_prover(SESSION, INNER_PRODUCT_INSTANCE);
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

fn inner_product_fixture() -> &'static InnerProductFixture {
    static FIXTURE: OnceLock<InnerProductFixture> = OnceLock::new();
    FIXTURE.get_or_init(InnerProductFixture::build)
}

fn arbitrary_weight(index: usize) -> F128 {
    let index = index as u64;
    F128::new(
        index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x0123_4567_89ab_cdef,
        index.rotate_left(29) ^ 0xa5a5_5a5a_f0f0_0f0f,
    )
}

fn set_packed_bit(packed_witness: &mut [F128], index: usize) {
    let packed_index = index / 128;
    let bit_index = index % 128;
    if bit_index < 64 {
        packed_witness[packed_index].lo |= 1 << bit_index;
    } else {
        packed_witness[packed_index].hi |= 1 << (bit_index - 64);
    }
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

fn bind_outer_inner_product_statement(
    transcript: &mut impl PublicTranscript,
    pcs: &Pcs,
    commitment: &Root,
    query: &OpeningQuery,
) {
    let OpeningQuery::InnerProduct { weights, target } = query else {
        panic!("expected an inner-product query");
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"outer/pcs-inner-product-weights/v1");
    hasher.update(&(weights.len() as u64).to_le_bytes());
    for weight in weights {
        hasher.update(&weight.to_bytes());
    }

    transcript.public_message(b"outer/pcs-inner-product/v1" as &[u8]);
    transcript.public_message(pcs);
    transcript.public_message(&commitment.0);
    transcript.public_message(&(weights.len() as u64));
    transcript.public_message(hasher.finalize().as_bytes());
    transcript.public_message(target);
}

#[test]
fn real_pcs_opening_round_trip_succeeds() {
    let fixture = fixture();
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);

    fixture
        .pcs
        .verify_lin(
            &fixture.commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        )
        .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn real_arbitrary_inner_product_round_trip_succeeds() {
    let fixture = inner_product_fixture();
    let mut verifier = build_verifier(SESSION, INNER_PRODUCT_INSTANCE, &fixture.proof);

    fixture
        .pcs
        .verify_lin(
            &fixture.commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        )
        .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn explicit_mle_weights_match_the_mle_opening_path() {
    let pcs = Pcs::new(&shape(), LigeritoProfile::Secure, HashKind::Blake3).unwrap();
    let point = (0..M)
        .map(|coordinate| F128::from(coordinate as u64 + 2))
        .collect::<Vec<_>>();
    let weights = eq_table(&point);
    assert_eq!(weights.len(), pcs.bit_len());

    let target_from_weights = INNER_PRODUCT_SET_BITS
        .into_iter()
        .map(|index| weights[index])
        .sum::<F128>();
    let target_from_point = INNER_PRODUCT_SET_BITS
        .into_iter()
        .map(|index| singleton_target(&point, index))
        .sum::<F128>();
    assert_eq!(target_from_weights, target_from_point);

    let mut packed_witness = vec![F128::default(); pcs.packed_len()];
    for index in INNER_PRODUCT_SET_BITS {
        set_packed_bit(&mut packed_witness, index);
    }
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let mle_query = OpeningQuery::Mle {
        point,
        target: target_from_point,
    };
    let inner_product_query = OpeningQuery::InnerProduct {
        weights,
        target: target_from_weights,
    };

    let mle_instance = b"m22-explicit-mle-weights/mle";
    let mut mle_prover = build_prover(SESSION, mle_instance);
    pcs.prove_lin(
        &data,
        packed_witness.clone(),
        &mle_query,
        StatementBinding::Bind,
        &mut mle_prover,
    )
    .unwrap();
    let mle_proof = mle_prover.finish();
    let mut mle_verifier = build_verifier(SESSION, mle_instance, &mle_proof);
    pcs.verify_lin(
        &commitment,
        &mle_query,
        StatementBinding::Bind,
        &mut mle_verifier,
    )
    .unwrap();
    mle_verifier.check_eof().unwrap();

    let inner_product_instance = b"m22-explicit-mle-weights/inner-product";
    let mut inner_product_prover = build_prover(SESSION, inner_product_instance);
    pcs.prove_lin(
        &data,
        packed_witness,
        &inner_product_query,
        StatementBinding::Bind,
        &mut inner_product_prover,
    )
    .unwrap();
    let inner_product_proof = inner_product_prover.finish();
    let mut inner_product_verifier =
        build_verifier(SESSION, inner_product_instance, &inner_product_proof);
    pcs.verify_lin(
        &commitment,
        &inner_product_query,
        StatementBinding::Bind,
        &mut inner_product_verifier,
    )
    .unwrap();
    inner_product_verifier.check_eof().unwrap();
}

#[test]
fn opening_query_variants_are_not_interchangeable() {
    let fixture = inner_product_fixture();
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::default(),
    };
    let mut verifier = build_verifier(SESSION, INNER_PRODUCT_INSTANCE, &fixture.proof);

    assert_eq!(
        fixture.pcs.verify_lin(
            &fixture.commitment,
            &query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::VerificationFailed),
    );
}

#[test]
fn arbitrary_inner_product_rejects_wrong_weight_lengths() {
    let pcs = Pcs::new(&shape(), LigeritoProfile::Secure, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let query = OpeningQuery::InnerProduct {
        weights: Vec::new(),
        target: F128::default(),
    };

    let mut prover = build_prover(SESSION, b"wrong-inner-product-weight-count");
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
        pcs.verify_lin(&commitment, &query, StatementBinding::Bind, &mut verifier),
        Err(VerifyError::WeightLengthMismatch),
    );
}

#[test]
fn arbitrary_inner_product_rejects_non_secure_profiles() {
    let query = OpeningQuery::InnerProduct {
        weights: Vec::new(),
        target: F128::default(),
    };
    let proof = Proof::default();
    let commitment = Root([0; 32]);

    for profile in [LigeritoProfile::Fast, LigeritoProfile::Slim] {
        let pcs = Pcs::new(&shape(), profile, HashKind::Blake3).unwrap();
        let packed_witness = vec![F128::default(); pcs.packed_len()];
        let (_, data) = pcs.commit(&packed_witness).unwrap();
        let mut prover = build_prover(SESSION, b"unsupported-inner-product-profile");
        assert_eq!(
            pcs.prove_lin(
                &data,
                packed_witness,
                &query,
                StatementBinding::Bind,
                &mut prover,
            ),
            Err(ProveError::UnsupportedInnerProductProfile),
        );

        let mut verifier = build_verifier(SESSION, b"unsupported-inner-product-profile", &proof);
        assert_eq!(
            pcs.verify_lin(&commitment, &query, StatementBinding::Bind, &mut verifier,),
            Err(VerifyError::UnsupportedInnerProductProfile),
        );
    }
}

#[test]
fn arbitrary_inner_product_prover_rejects_a_false_target() {
    let pcs = Pcs::new(&shape(), LigeritoProfile::Secure, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (_, data) = pcs.commit(&packed_witness).unwrap();
    let query = OpeningQuery::InnerProduct {
        weights: vec![F128::default(); pcs.bit_len()],
        target: F128::from(1u64),
    };
    let mut prover = build_prover(SESSION, b"false-inner-product");

    assert_eq!(
        pcs.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(ProveError::InvalidClaim),
    );
}

#[test]
fn arbitrary_inner_product_accepts_an_already_bound_statement() {
    let pcs = Pcs::new(&shape(), LigeritoProfile::Secure, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let query = OpeningQuery::InnerProduct {
        weights: vec![F128::default(); pcs.bit_len()],
        target: F128::default(),
    };
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();

    let mut prover = build_prover(SESSION, b"already-bound-inner-product");
    bind_outer_inner_product_statement(&mut prover, &pcs, &commitment, &query);
    pcs.prove_lin(
        &data,
        packed_witness,
        &query,
        StatementBinding::AlreadyBound,
        &mut prover,
    )
    .unwrap();
    let proof = prover.finish();

    let mut verifier = build_verifier(SESSION, b"already-bound-inner-product", &proof);
    bind_outer_inner_product_statement(&mut verifier, &pcs, &commitment, &query);
    pcs.verify_lin(
        &commitment,
        &query,
        StatementBinding::AlreadyBound,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();

    let mut mismatched_verifier = build_verifier(SESSION, b"already-bound-inner-product", &proof);
    bind_outer_inner_product_statement(&mut mismatched_verifier, &pcs, &commitment, &query);
    assert!(
        pcs.verify_lin(
            &commitment,
            &query,
            StatementBinding::Bind,
            &mut mismatched_verifier,
        )
        .is_err(),
    );
}

#[test]
fn arbitrary_inner_product_rejects_statement_and_coordinate_mutations() {
    let fixture = inner_product_fixture();
    let mut changed_query = fixture.query.clone();
    let OpeningQuery::InnerProduct { weights, .. } = &mut changed_query else {
        unreachable!();
    };
    weights[2] += F128::from(1u64);
    let mut verifier = build_verifier(SESSION, INNER_PRODUCT_INSTANCE, &fixture.proof);
    assert!(
        fixture
            .pcs
            .verify_lin(
                &fixture.commitment,
                &changed_query,
                StatementBinding::Bind,
                &mut verifier,
            )
            .is_err(),
    );

    let mut changed_query = fixture.query.clone();
    let OpeningQuery::InnerProduct { target, .. } = &mut changed_query else {
        unreachable!();
    };
    *target += F128::from(1u64);
    let mut verifier = build_verifier(SESSION, INNER_PRODUCT_INSTANCE, &fixture.proof);
    assert_eq!(
        fixture.pcs.verify_lin(
            &fixture.commitment,
            &changed_query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::VerificationFailed),
    );

    let mut changed_root = fixture.commitment.0;
    changed_root[0] ^= 1;
    let changed_commitment = Root(changed_root);
    let mut verifier = build_verifier(SESSION, INNER_PRODUCT_INSTANCE, &fixture.proof);
    assert_eq!(
        fixture.pcs.verify_lin(
            &changed_commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::VerificationFailed),
    );

    let mut changed_proof = fixture.proof.clone();
    changed_proof.narg_string[1] ^= 1;
    let mut verifier = build_verifier(SESSION, INNER_PRODUCT_INSTANCE, &changed_proof);
    assert_eq!(
        fixture.pcs.verify_lin(
            &fixture.commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(VerifyError::VerificationFailed),
    );
}

#[test]
fn slim_profile_opening_round_trip_exercises_pow() {
    let fixture = RealFixture::build(LigeritoProfile::Slim);
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);

    fixture
        .pcs
        .verify_lin(
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
    let pcs = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();

    let mut prover = build_prover(SESSION, b"already-bound");
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
    bind_outer_statement(&mut verifier, &pcs, &commitment, &query);
    pcs.verify_lin(
        &commitment,
        &query,
        StatementBinding::AlreadyBound,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();

    let mut mismatched_verifier = build_verifier(SESSION, b"already-bound", &proof);
    bind_outer_statement(&mut mismatched_verifier, &pcs, &commitment, &query);
    assert!(
        pcs.verify_lin(
            &commitment,
            &query,
            StatementBinding::Bind,
            &mut mismatched_verifier,
        )
        .is_err()
    );
}

#[test]
fn real_pcs_rejects_point_length_mismatches() {
    let pcs = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let short_query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M - 1],
        target: F128::from(0u64),
    };

    let mut prover = build_prover(SESSION, b"wrong-prover-point");
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
    let proof = Proof::default();
    let mut verifier = build_verifier(SESSION, b"wrong-verifier-point", &proof);
    assert_eq!(
        pcs.verify_lin(
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
    let pcs = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let mut packed_witness = vec![F128::default(); pcs.packed_len()];
    let (_, data) = pcs.commit(&packed_witness).unwrap();
    packed_witness.pop();
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };
    let mut prover = build_prover(SESSION, b"wrong-packed-length");

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
    let source = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); source.packed_len()];
    let (_, data) = source.commit(&packed_witness).unwrap();
    let other = Pcs::new(&shape(), LigeritoProfile::Slim, HashKind::Blake3).unwrap();
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };
    let mut prover = build_prover(SESSION, b"mismatched-parameters");

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
    let pcs = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (_, data) = pcs.commit(&packed_witness).unwrap();
    let query = OpeningQuery::Mle {
        point: vec![F128::from(2u64); M],
        target: F128::from(1u64),
    };
    let mut prover = build_prover(SESSION, b"false-evaluation");
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
    let pcs = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let mut different_witness = packed_witness;
    different_witness[0].lo = 1;
    let point = (0..M)
        .map(|coordinate| F128::from(coordinate as u64 + 2))
        .collect::<Vec<_>>();
    let query = OpeningQuery::Mle {
        target: singleton_target(&point, 0),
        point,
    };
    let mut prover = build_prover(SESSION, b"different-packed-witness");
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
        pcs.verify_lin(&commitment, &query, StatementBinding::Bind, &mut verifier,),
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
        fixture.pcs.verify_lin(
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
        fixture.pcs.verify_lin(
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
        fixture.pcs.verify_lin(
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
        fixture.pcs.verify_lin(
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
        fixture.pcs.verify_lin(
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
    fixture
        .pcs
        .verify_lin(
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

    fixture
        .pcs
        .verify_lin(
            &fixture.commitment,
            &fixture.query,
            StatementBinding::Bind,
            &mut verifier,
        )
        .unwrap();
    assert!(verifier.check_eof().is_err());
}
