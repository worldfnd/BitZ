use std::sync::OnceLock;

use field::F128;
use pcs::{CommitError, CommitScheme, Commitment, HashKind, LigeritoProfile, OpeningQuery, Pcs};
use transcript::{Proof, build_prover, build_verifier};

const M: usize = 22;
const SINGLETON: usize = (1 << 21) | (1 << 7) | 0b101_0101;
const SESSION: &[u8] = b"pcs-interface-test";
const INSTANCE: &[u8] = b"m22-singleton-opening";

struct RealFixture {
    pcs: Pcs,
    commitment: Commitment,
    query: OpeningQuery,
    proof: Proof,
}

impl RealFixture {
    fn build() -> Self {
        let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3);
        // One nonzero bit gives the expected MLE value a simple independent formula.
        let mut bits = vec![false; pcs.bit_len()];
        bits[SINGLETON] = true;

        let point = (0..M)
            .map(|coordinate| F128::from(coordinate as u64 + 2))
            .collect::<Vec<_>>();
        let query = OpeningQuery {
            target: singleton_target(&point, SINGLETON),
            point,
        };

        let (commitment, data) = pcs.commit(&bits).unwrap();
        let mut prover = build_prover(SESSION, INSTANCE);
        pcs.prove_lin(data, &query, &mut prover).unwrap();

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
fn real_pcs_rejects_point_length_mismatches() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3);
    let (commitment, data) = pcs.commit(&vec![false; pcs.bit_len()]).unwrap();
    let short_query = OpeningQuery {
        point: vec![F128::from(2u64); M - 1],
        target: F128::from(0u64),
    };

    let mut prover = build_prover(SESSION, b"wrong-prover-point");
    assert_eq!(
        pcs.prove_lin(data, &short_query, &mut prover),
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
fn real_pcs_rejects_mismatched_prover_parameters() {
    let source = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3);
    let (_, data) = source.commit(&vec![false; source.bit_len()]).unwrap();
    let other = Pcs::new(M, LigeritoProfile::Slim, HashKind::Blake3);
    let query = OpeningQuery {
        point: vec![F128::from(2u64); M],
        target: F128::from(0u64),
    };
    let mut prover = build_prover(SESSION, b"mismatched-parameters");

    assert_eq!(
        other.prove_lin(data, &query, &mut prover),
        Err(CommitError::InvalidConfiguration)
    );
}

#[test]
fn real_pcs_prover_rejects_a_false_evaluation() {
    let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3);
    let (_, data) = pcs.commit(&vec![false; pcs.bit_len()]).unwrap();
    let query = OpeningQuery {
        point: vec![F128::from(2u64); M],
        target: F128::from(1u64),
    };
    let mut prover = build_prover(SESSION, b"false-evaluation");

    assert_eq!(
        pcs.prove_lin(data, &query, &mut prover),
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
        Err(CommitError::MalformedProof)
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
