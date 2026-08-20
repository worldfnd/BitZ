use std::sync::OnceLock;

use field::F128;
use pcs::{
    CommitError, CommitScheme, Commitment, HashKind, LigeritoProfile, OpeningQuery, Pcs,
    ProverData, ScopedOpeningQuery, StatementBinding,
};
use transcript::{Proof, ProverState, VerifierState, build_prover, build_verifier};

const M: usize = 22;
const SINGLETON: usize = (1 << 21) | (1 << 7) | 0b101_0101;
const SESSION: &[u8] = b"pcs-interface-test";
const INSTANCE: &[u8] = b"m22-singleton-opening";
const BATCH_INSTANCE: &[u8] = b"m22-three-query-opening";
const ALREADY_BOUND_INSTANCE: &[u8] = b"m22-transcript-derived-opening";
const OUTER_STATEMENT_LABEL: &[u8] = b"pcs-test/already-bound-statement/v1";

struct RealFixture {
    pcs: Pcs,
    commitment: Commitment,
    query: OpeningQuery,
    proof: Proof,
}

impl RealFixture {
    fn build() -> Self {
        let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
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
        let query = OpeningQuery {
            target: singleton_target(&point, SINGLETON),
            point,
        };

        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let mut prover = build_prover(SESSION, INSTANCE);
        prove_single(&pcs, &data, packed_witness, &query, &mut prover).unwrap();

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
    FIXTURE.get_or_init(RealFixture::build)
}

struct BatchFixture {
    pcs: Pcs,
    commitment: Commitment,
    queries: [OpeningQuery; 3],
    proof: Proof,
}

impl BatchFixture {
    fn build() -> Self {
        let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let mut packed_witness = vec![F128::default(); pcs.packed_len()];
        let packed_index = SINGLETON / 128;
        let bit_index = SINGLETON % 128;
        if bit_index < 64 {
            packed_witness[packed_index].lo |= 1 << bit_index;
        } else {
            packed_witness[packed_index].hi |= 1 << (bit_index - 64);
        }

        let queries = [2u64, 19u64, 31u64].map(|offset| {
            let point = (0..M)
                .map(|coordinate| F128::from(coordinate as u64 + offset))
                .collect::<Vec<_>>();
            OpeningQuery {
                target: singleton_target(&point, SINGLETON),
                point,
            }
        });

        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let scoped_queries = scoped_batch(&queries);
        let mut prover = build_prover(SESSION, BATCH_INSTANCE);
        pcs.prove_lin_batch(
            &data,
            packed_witness,
            &scoped_queries,
            StatementBinding::Bind,
            &mut prover,
        )
        .unwrap();

        Self {
            pcs,
            commitment,
            queries,
            proof: prover.finish(),
        }
    }
}

fn scoped_batch(queries: &[OpeningQuery; 3]) -> [ScopedOpeningQuery<'_>; 3] {
    [
        ScopedOpeningQuery::new(0, &queries[0]),
        ScopedOpeningQuery::new(2, &queries[1]),
        ScopedOpeningQuery::new(5, &queries[2]),
    ]
}

fn prove_single(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    query: &OpeningQuery,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    pcs.prove_lin_batch(
        data,
        packed_witness,
        &[ScopedOpeningQuery::new(0, query)],
        StatementBinding::Bind,
        transcript,
    )
}

fn verify_single(
    pcs: &Pcs,
    commitment: &Commitment,
    query: &OpeningQuery,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    pcs.verify_lin_batch(
        commitment,
        &[ScopedOpeningQuery::new(0, query)],
        StatementBinding::Bind,
        transcript,
    )
}

fn batch_fixture() -> &'static BatchFixture {
    static FIXTURE: OnceLock<BatchFixture> = OnceLock::new();
    FIXTURE.get_or_init(BatchFixture::build)
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
fn real_pcs_opening_round_trip_succeeds() {
    let fixture = fixture();
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);

    verify_single(
        &fixture.pcs,
        &fixture.commitment,
        &fixture.query,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn real_pcs_batch_opening_round_trip_succeeds() {
    let fixture = batch_fixture();
    let scoped_queries = scoped_batch(&fixture.queries);
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    fixture
        .pcs
        .verify_lin_batch(
            &fixture.commitment,
            &scoped_queries,
            StatementBinding::Bind,
            &mut verifier,
        )
        .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn real_pcs_already_bound_batch_derives_queries_from_transcript() {
    const SCOPES: [u32; 3] = [0, 2, 5];

    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let target = F128::default();

    let mut prover = build_prover(SESSION, ALREADY_BOUND_INSTANCE);
    prover.public_message(OUTER_STATEMENT_LABEL);
    prover.public_message(&pcs);
    prover.public_message(commitment.root());
    prover.public_message(&(SCOPES.len() as u64));
    let prover_queries = SCOPES.map(|scope| {
        prover.public_message(&scope);
        prover.public_message(&target);
        OpeningQuery {
            point: (0..M).map(|_| prover.verifier_message::<F128>()).collect(),
            target,
        }
    });
    let prover_scoped_queries = scoped_batch(&prover_queries);
    pcs.prove_lin_batch(
        &data,
        packed_witness,
        &prover_scoped_queries,
        StatementBinding::AlreadyBound,
        &mut prover,
    )
    .unwrap();
    let proof = prover.finish();

    let mut verifier = build_verifier(SESSION, ALREADY_BOUND_INSTANCE, &proof);
    verifier.public_message(OUTER_STATEMENT_LABEL);
    verifier.public_message(&pcs);
    verifier.public_message(commitment.root());
    verifier.public_message(&(SCOPES.len() as u64));
    let verifier_queries = SCOPES.map(|scope| {
        verifier.public_message(&scope);
        verifier.public_message(&target);
        OpeningQuery {
            point: (0..M)
                .map(|_| verifier.verifier_message::<F128>())
                .collect(),
            target,
        }
    });
    assert_eq!(verifier_queries, prover_queries);
    let verifier_scoped_queries = scoped_batch(&verifier_queries);

    pcs.verify_lin_batch(
        &commitment,
        &verifier_scoped_queries,
        StatementBinding::AlreadyBound,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn real_pcs_batch_rejects_query_reordering() {
    let fixture = batch_fixture();
    let reordered = [
        ScopedOpeningQuery::new(0, &fixture.queries[1]),
        ScopedOpeningQuery::new(2, &fixture.queries[0]),
        ScopedOpeningQuery::new(5, &fixture.queries[2]),
    ];
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    assert!(matches!(
        fixture.pcs.verify_lin_batch(
            &fixture.commitment,
            &reordered,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(CommitError::VerificationFailed | CommitError::MalformedProof)
    ));
}

#[test]
fn real_pcs_batch_rejects_changed_scopes() {
    let fixture = batch_fixture();
    let changed_scopes = [
        ScopedOpeningQuery::new(0, &fixture.queries[0]),
        ScopedOpeningQuery::new(3, &fixture.queries[1]),
        ScopedOpeningQuery::new(5, &fixture.queries[2]),
    ];
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    assert_eq!(
        fixture.pcs.verify_lin_batch(
            &fixture.commitment,
            &changed_scopes,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(CommitError::MalformedProof)
    );
}

#[test]
fn real_pcs_batch_rejects_a_changed_later_ring_message() {
    const RING_FRAME_LEN: usize = 24 + 4 + 128 * 16;

    let fixture = batch_fixture();
    let mut changed_proof = fixture.proof.clone();
    changed_proof.narg_string[RING_FRAME_LEN + 28] ^= 1;
    let scoped_queries = scoped_batch(&fixture.queries);
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &changed_proof);

    assert_eq!(
        fixture.pcs.verify_lin_batch(
            &fixture.commitment,
            &scoped_queries,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(CommitError::MalformedProof)
    );
}

#[test]
fn real_pcs_batch_rejects_non_increasing_scopes() {
    let fixture = batch_fixture();
    let invalid_scopes = [
        ScopedOpeningQuery::new(0, &fixture.queries[0]),
        ScopedOpeningQuery::new(2, &fixture.queries[1]),
        ScopedOpeningQuery::new(1, &fixture.queries[2]),
    ];
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    assert_eq!(
        fixture.pcs.verify_lin_batch(
            &fixture.commitment,
            &invalid_scopes,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(CommitError::InvalidClaimScopeOrder)
    );
}

#[test]
fn real_pcs_batch_rejects_a_false_later_claim() {
    let fixture = batch_fixture();
    let mut changed_queries = fixture.queries.clone();
    changed_queries[2].target += F128::from(1u64);
    let changed_queries = scoped_batch(&changed_queries);
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    assert_eq!(
        fixture.pcs.verify_lin_batch(
            &fixture.commitment,
            &changed_queries,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(CommitError::VerificationFailed)
    );
}

#[test]
fn real_pcs_batch_rejects_a_query_count_mismatch() {
    let fixture = batch_fixture();
    let prefix = [
        ScopedOpeningQuery::new(0, &fixture.queries[0]),
        ScopedOpeningQuery::new(2, &fixture.queries[1]),
    ];
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    assert_eq!(
        fixture.pcs.verify_lin_batch(
            &fixture.commitment,
            &prefix,
            StatementBinding::Bind,
            &mut verifier,
        ),
        Err(CommitError::VerificationFailed)
    );
}

#[test]
fn real_pcs_rejects_empty_batches() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let mut prover = build_prover(SESSION, b"empty-batch");

    assert_eq!(
        pcs.prove_lin_batch(
            &data,
            packed_witness,
            &[],
            StatementBinding::Bind,
            &mut prover,
        ),
        Err(CommitError::EmptyBatch)
    );

    let proof = Proof::default();
    let mut verifier = build_verifier(SESSION, b"empty-batch", &proof);
    assert_eq!(
        pcs.verify_lin_batch(&commitment, &[], StatementBinding::Bind, &mut verifier,),
        Err(CommitError::EmptyBatch)
    );
}

#[test]
fn real_pcs_rejects_point_length_mismatches() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let short_query = OpeningQuery {
        point: vec![F128::from(2u64); M - 1],
        target: F128::from(0u64),
    };

    let mut prover = build_prover(SESSION, b"wrong-prover-point");
    assert_eq!(
        prove_single(&pcs, &data, packed_witness, &short_query, &mut prover),
        Err(CommitError::PointLengthMismatch)
    );

    let long_query = OpeningQuery {
        point: vec![F128::from(2u64); M + 1],
        target: F128::from(0u64),
    };
    let proof = Proof::default();
    let mut verifier = build_verifier(SESSION, b"wrong-verifier-point", &proof);
    assert_eq!(
        verify_single(&pcs, &commitment, &long_query, &mut verifier),
        Err(CommitError::PointLengthMismatch)
    );
}

#[test]
fn real_pcs_rejects_packed_witness_length_mismatches_during_opening() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let mut packed_witness = vec![F128::default(); pcs.packed_len()];
    let (_, data) = pcs.commit(&packed_witness).unwrap();
    packed_witness.pop();
    let query = OpeningQuery {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };
    let mut prover = build_prover(SESSION, b"wrong-packed-length");

    assert_eq!(
        prove_single(&pcs, &data, packed_witness, &query, &mut prover),
        Err(CommitError::InvalidBitLength)
    );
}

#[test]
fn real_pcs_rejects_mismatched_prover_parameters() {
    let source = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); source.packed_len()];
    let (_, data) = source.commit(&packed_witness).unwrap();
    let other = Pcs::new(M, LigeritoProfile::Slim, HashKind::Blake3).unwrap();
    let query = OpeningQuery {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };
    let mut prover = build_prover(SESSION, b"mismatched-parameters");

    assert!(matches!(
        prove_single(&other, &data, packed_witness, &query, &mut prover),
        Err(CommitError::InvalidConfiguration(description))
            if description == "prover data parameters mismatch"
    ));
}

#[test]
fn real_pcs_prover_rejects_a_false_evaluation_without_consuming_prover_data() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (_, data) = pcs.commit(&packed_witness).unwrap();
    let query = OpeningQuery {
        point: vec![F128::from(2u64); M],
        target: F128::from(1u64),
    };
    let mut prover = build_prover(SESSION, b"false-evaluation");
    let codeword_len = data.codeword_len();

    assert_eq!(
        prove_single(&pcs, &data, packed_witness, &query, &mut prover),
        Err(CommitError::VerificationFailed)
    );
    assert_eq!(data.codeword_len(), codeword_len);
}

#[test]
fn real_pcs_rejects_an_opening_for_a_different_packed_witness() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let mut different_witness = packed_witness;
    different_witness[0].lo = 1;
    let point = (0..M)
        .map(|coordinate| F128::from(coordinate as u64 + 2))
        .collect::<Vec<_>>();
    let query = OpeningQuery {
        target: singleton_target(&point, 0),
        point,
    };
    let mut prover = build_prover(SESSION, b"different-packed-witness");
    prove_single(&pcs, &data, different_witness, &query, &mut prover).unwrap();
    let proof = prover.finish();
    let mut verifier = build_verifier(SESSION, b"different-packed-witness", &proof);

    assert_eq!(
        verify_single(&pcs, &commitment, &query, &mut verifier),
        Err(CommitError::VerificationFailed)
    );
}

#[test]
fn real_pcs_rejects_statement_mutations() {
    let fixture = fixture();

    let mut changed_query = fixture.query.clone();
    changed_query.target += F128::from(1u64);
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);
    assert_eq!(
        verify_single(
            &fixture.pcs,
            &fixture.commitment,
            &changed_query,
            &mut verifier,
        ),
        Err(CommitError::VerificationFailed)
    );

    let mut changed_root = *fixture.commitment.root();
    changed_root[0] ^= 1;
    let changed_commitment = Commitment::from_root(changed_root);
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);
    assert_eq!(
        verify_single(
            &fixture.pcs,
            &changed_commitment,
            &fixture.query,
            &mut verifier,
        ),
        Err(CommitError::VerificationFailed)
    );
}

#[test]
fn real_pcs_rejects_malformed_transcript_streams() {
    let fixture = fixture();

    let mut changed_stream = fixture.proof.clone();
    changed_stream.narg_string[0] ^= 1;
    let mut verifier = build_verifier(SESSION, INSTANCE, &changed_stream);
    assert_eq!(
        verify_single(
            &fixture.pcs,
            &fixture.commitment,
            &fixture.query,
            &mut verifier,
        ),
        Err(CommitError::MalformedProof)
    );

    let mut truncated_hint = fixture.proof.clone();
    truncated_hint.hints.pop();
    let mut verifier = build_verifier(SESSION, INSTANCE, &truncated_hint);
    assert_eq!(
        verify_single(
            &fixture.pcs,
            &fixture.commitment,
            &fixture.query,
            &mut verifier,
        ),
        Err(CommitError::MalformedProof)
    );

    let mut changed_hint = fixture.proof.clone();
    *changed_hint.hints.last_mut().unwrap() ^= 1;
    let mut verifier = build_verifier(SESSION, INSTANCE, &changed_hint);
    assert_eq!(
        verify_single(
            &fixture.pcs,
            &fixture.commitment,
            &fixture.query,
            &mut verifier,
        ),
        Err(CommitError::MalformedProof)
    );
}

#[test]
fn real_pcs_rejects_trailing_bytes_inside_opening_hint() {
    let fixture = fixture();
    let mut changed_hint = fixture.proof.clone();
    let encoded_len = u32::from_le_bytes(changed_hint.hints[..4].try_into().unwrap());
    changed_hint.hints[..4].copy_from_slice(&(encoded_len + 1).to_le_bytes());
    changed_hint.hints.push(0);
    let mut verifier = build_verifier(SESSION, INSTANCE, &changed_hint);

    assert_eq!(
        verify_single(
            &fixture.pcs,
            &fixture.commitment,
            &fixture.query,
            &mut verifier,
        ),
        Err(CommitError::MalformedProof)
    );
}

#[test]
fn real_pcs_requires_complete_transcript_consumption() {
    let fixture = fixture();

    let mut trailing_narg = fixture.proof.clone();
    trailing_narg.narg_string.push(0);
    let mut verifier = build_verifier(SESSION, INSTANCE, &trailing_narg);
    verify_single(
        &fixture.pcs,
        &fixture.commitment,
        &fixture.query,
        &mut verifier,
    )
    .unwrap();
    assert!(verifier.check_eof().is_err());

    let mut trailing_hint = fixture.proof.clone();
    trailing_hint.hints.push(0);
    let mut verifier = build_verifier(SESSION, INSTANCE, &trailing_hint);

    verify_single(
        &fixture.pcs,
        &fixture.commitment,
        &fixture.query,
        &mut verifier,
    )
    .unwrap();
    assert!(verifier.check_eof().is_err());
}
