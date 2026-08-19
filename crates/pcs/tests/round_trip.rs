use std::sync::OnceLock;

use field::F128;
use pcs::{CommitError, CommitScheme, Commitment, HashKind, LigeritoProfile, OpeningQuery, Pcs};
use transcript::{Proof, build_prover, build_verifier};

const M: usize = 22;
const SINGLETON: usize = (1 << 21) | (1 << 7) | 0b101_0101;
const SESSION: &[u8] = b"pcs-interface-test";
const INSTANCE: &[u8] = b"m22-singleton-opening";
const BATCH_INSTANCE: &[u8] = b"m22-two-query-opening";

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
        pcs.prove_lin(data, packed_witness, &query, &mut prover)
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
    FIXTURE.get_or_init(RealFixture::build)
}

struct BatchFixture {
    pcs: Pcs,
    commitment: Commitment,
    queries: [OpeningQuery; 2],
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

        let queries = [2u64, 19u64].map(|offset| {
            let point = (0..M)
                .map(|coordinate| F128::from(coordinate as u64 + offset))
                .collect::<Vec<_>>();
            OpeningQuery {
                target: singleton_target(&point, SINGLETON),
                point,
            }
        });

        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let mut prover = build_prover(SESSION, BATCH_INSTANCE);
        pcs.prove_lin_batch(data, packed_witness, &queries, &mut prover)
            .unwrap();

        Self {
            pcs,
            commitment,
            queries,
            proof: prover.finish(),
        }
    }
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

    fixture
        .pcs
        .verify_lin(&fixture.commitment, &fixture.query, &mut verifier)
        .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn real_pcs_batch_opening_round_trip_succeeds() {
    let fixture = batch_fixture();
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    fixture
        .pcs
        .verify_lin_batch(&fixture.commitment, &fixture.queries, &mut verifier)
        .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn real_pcs_batch_rejects_query_reordering() {
    let fixture = batch_fixture();
    let reversed = [fixture.queries[1].clone(), fixture.queries[0].clone()];
    let mut verifier = build_verifier(SESSION, BATCH_INSTANCE, &fixture.proof);

    assert!(matches!(
        fixture
            .pcs
            .verify_lin_batch(&fixture.commitment, &reversed, &mut verifier),
        Err(CommitError::VerificationFailed | CommitError::MalformedProof)
    ));
}

#[test]
fn real_pcs_rejects_empty_batches() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (commitment, data) = pcs.commit(&packed_witness).unwrap();
    let mut prover = build_prover(SESSION, b"empty-batch");

    assert_eq!(
        pcs.prove_lin_batch(data, packed_witness, &[], &mut prover),
        Err(CommitError::EmptyBatch)
    );

    let proof = Proof::default();
    let mut verifier = build_verifier(SESSION, b"empty-batch", &proof);
    assert_eq!(
        pcs.verify_lin_batch(&commitment, &[], &mut verifier),
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
        pcs.prove_lin(data, packed_witness, &short_query, &mut prover),
        Err(CommitError::PointLengthMismatch)
    );

    let long_query = OpeningQuery {
        point: vec![F128::from(2u64); M + 1],
        target: F128::from(0u64),
    };
    let proof = Proof::default();
    let mut verifier = build_verifier(SESSION, b"wrong-verifier-point", &proof);
    assert_eq!(
        pcs.verify_lin(&commitment, &long_query, &mut verifier),
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
        pcs.prove_lin(data, packed_witness, &query, &mut prover),
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
        other.prove_lin(data, packed_witness, &query, &mut prover),
        Err(CommitError::InvalidConfiguration(description))
            if description.starts_with("prover data parameters do not match the active PCS")
    ));
}

#[test]
fn real_pcs_prover_rejects_a_false_evaluation() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let packed_witness = vec![F128::default(); pcs.packed_len()];
    let (_, data) = pcs.commit(&packed_witness).unwrap();
    let query = OpeningQuery {
        point: vec![F128::from(2u64); M],
        target: F128::from(1u64),
    };
    let mut prover = build_prover(SESSION, b"false-evaluation");

    assert_eq!(
        pcs.prove_lin(data, packed_witness, &query, &mut prover),
        Err(CommitError::VerificationFailed)
    );
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
    pcs.prove_lin(data, different_witness, &query, &mut prover)
        .unwrap();
    let proof = prover.finish();
    let mut verifier = build_verifier(SESSION, b"different-packed-witness", &proof);

    assert_eq!(
        pcs.verify_lin(&commitment, &query, &mut verifier),
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
        fixture
            .pcs
            .verify_lin(&fixture.commitment, &changed_query, &mut verifier),
        Err(CommitError::VerificationFailed)
    );

    let mut changed_root = *fixture.commitment.root();
    changed_root[0] ^= 1;
    let changed_commitment = Commitment::from_root(changed_root);
    let mut verifier = build_verifier(SESSION, INSTANCE, &fixture.proof);
    assert_eq!(
        fixture
            .pcs
            .verify_lin(&changed_commitment, &fixture.query, &mut verifier),
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
        fixture
            .pcs
            .verify_lin(&fixture.commitment, &fixture.query, &mut verifier),
        Err(CommitError::MalformedProof)
    );

    let mut truncated_hint = fixture.proof.clone();
    truncated_hint.hints.pop();
    let mut verifier = build_verifier(SESSION, INSTANCE, &truncated_hint);
    assert_eq!(
        fixture
            .pcs
            .verify_lin(&fixture.commitment, &fixture.query, &mut verifier),
        Err(CommitError::MalformedProof)
    );

    let mut changed_hint = fixture.proof.clone();
    *changed_hint.hints.last_mut().unwrap() ^= 1;
    let mut verifier = build_verifier(SESSION, INSTANCE, &changed_hint);
    assert_eq!(
        fixture
            .pcs
            .verify_lin(&fixture.commitment, &fixture.query, &mut verifier),
        Err(CommitError::MalformedProof)
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
        .verify_lin(&fixture.commitment, &fixture.query, &mut verifier)
        .unwrap();
    assert!(verifier.check_eof().is_err());

    let mut trailing_hint = fixture.proof.clone();
    trailing_hint.hints.push(0);
    let mut verifier = build_verifier(SESSION, INSTANCE, &trailing_hint);

    fixture
        .pcs
        .verify_lin(&fixture.commitment, &fixture.query, &mut verifier)
        .unwrap();
    assert!(verifier.check_eof().is_err());
}
