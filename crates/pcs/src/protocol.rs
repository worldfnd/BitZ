//! Shared encoding and transcript rules for multilinear openings.
//!
//! The complete opening proof uses one bounded hint because Flock verifies an in-memory proof.
//! Fiat–Shamir values also use NARG and are checked against the hint during replay.

use bincode::Options;
use flock_core::challenger::Challenger;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::{BatchOpeningProofLigerito, LOG_PACKING};
use transcript::{Encoding, ProverState, VerifierState};

use crate::challenger::ScopedChallenger;
use crate::{CommitError, Pcs, ScopedOpeningQuery};

pub(crate) const PROOF_HINT_LIMIT: usize = 64 * 1024 * 1024;
pub(crate) const STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v3";
pub(crate) const RING_SWITCH_LABEL: &[u8] = b"f2z/pcs/group-8/ring-switch/v1";
pub(crate) const RING_SWITCH_CLAIM_LABEL: &[u8] = b"f2z/pcs/group-8/claim/v1";
pub(crate) const ETA_SQUEEZE_LABEL: &[u8] = b"f2z/pcs/group-9/eta/v1";

/// Absorbs every scoped ring-switch message, then samples the shared point.
pub(crate) fn sample_shared_ring_switch_point<'a>(
    challenger: &mut impl ScopedChallenger,
    claims: impl IntoIterator<Item = (u64, &'a [FlockF128])>,
) -> Vec<FlockF128> {
    challenger.observe_label(RING_SWITCH_LABEL);
    for (scope, s_hat_v) in claims {
        challenger.observe_label(RING_SWITCH_CLAIM_LABEL);
        challenger.observe_scope(scope);
        challenger.observe_f128_slice(s_hat_v);
    }
    challenger.sample_f128_vec(LOG_PACKING)
}

/// Samples one batching scalar per claim after the shared point.
pub(crate) fn sample_batching_scalars(
    challenger: &mut impl Challenger,
    claim_count: usize,
) -> Vec<FlockF128> {
    challenger.observe_label(ETA_SQUEEZE_LABEL);
    challenger.sample_f128_vec(claim_count)
}

pub(crate) fn write_opening_proof(
    proof: &BatchOpeningProofLigerito,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    let proof_bytes = proof_options()
        .serialize(proof)
        .map_err(map_serialization_error)?;
    if proof_bytes.len() > PROOF_HINT_LIMIT {
        return Err(CommitError::ProofTooLarge);
    }
    transcript.hint_bytes(&proof_bytes);
    Ok(())
}

fn map_serialization_error(error: bincode::Error) -> CommitError {
    if matches!(error.as_ref(), bincode::ErrorKind::SizeLimit) {
        CommitError::ProofTooLarge
    } else {
        CommitError::SerializationFailed(error.to_string())
    }
}

pub(crate) fn read_opening_proof(
    transcript: &mut VerifierState<'_>,
) -> Result<BatchOpeningProofLigerito, CommitError> {
    let proof_bytes = transcript
        .hint_bytes(PROOF_HINT_LIMIT)
        .map_err(|_| CommitError::MalformedProof)?;
    proof_options()
        .deserialize(&proof_bytes)
        .map_err(|_| CommitError::MalformedProof)
}

pub(crate) trait PublicTranscript {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T);
}

impl PublicTranscript for ProverState {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        ProverState::public_message(self, message);
    }
}

impl PublicTranscript for VerifierState<'_> {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        VerifierState::public_message(self, message);
    }
}

/// Absorbs the public statement in either transcript.
pub(crate) fn bind_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    queries: &[ScopedOpeningQuery<'_>],
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(queries.len() as u64));
    for scoped_query in queries {
        transcript.public_message(&scoped_query.scope);
        let query = scoped_query.query;
        transcript.public_message(&(query.point.len() as u64));
        for coordinate in &query.point {
            transcript.public_message(coordinate);
        }
        transcript.public_message(&query.target);
    }
}

pub(crate) fn validate_batch(
    queries: &[ScopedOpeningQuery<'_>],
    expected_m: usize,
) -> Result<(), CommitError> {
    if queries.is_empty() {
        return Err(CommitError::EmptyBatch);
    }
    if queries
        .windows(2)
        .any(|pair| pair[0].scope >= pair[1].scope)
    {
        return Err(CommitError::InvalidClaimScopeOrder);
    }
    if queries
        .iter()
        .any(|scoped_query| scoped_query.query.point.len() != expected_m)
    {
        return Err(CommitError::PointLengthMismatch);
    }
    Ok(())
}

fn proof_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(PROOF_HINT_LIMIT as u64)
        .reject_trailing_bytes()
}

#[cfg(test)]
mod tests {
    use field::F128;
    use proptest::prelude::*;
    use transcript::{NargSerialize, Proof, build_prover, build_verifier};

    use super::*;
    use crate::challenger::ProverChallenger;
    use crate::{HashKind, LigeritoProfile, OpeningQuery, ScopedOpeningQuery};

    #[test]
    fn proof_serialization_errors_are_specific() {
        assert_eq!(
            map_serialization_error(Box::new(bincode::ErrorKind::SizeLimit)),
            CommitError::ProofTooLarge
        );
        assert_eq!(
            map_serialization_error(Box::new(bincode::ErrorKind::Custom(
                "serialization failed".to_owned(),
            ))),
            CommitError::SerializationFailed("serialization failed".to_owned())
        );
    }

    #[test]
    fn opening_proof_reader_rejects_malformed_bytes() {
        let mut prover = build_prover(b"pcs-protocol-test", b"malformed-proof");
        prover.hint_bytes(&[0xff]);
        let proof = prover.finish();
        let mut verifier = build_verifier(b"pcs-protocol-test", b"malformed-proof", &proof);

        assert_eq!(
            read_opening_proof(&mut verifier),
            Err(CommitError::MalformedProof)
        );
    }

    #[test]
    fn opening_proof_reader_rejects_an_oversized_hint() {
        let oversized = u32::try_from(PROOF_HINT_LIMIT + 1).unwrap();
        let mut hints = Vec::new();
        oversized.serialize_into_narg(&mut hints);
        let proof = Proof {
            narg_string: Vec::new(),
            hints,
        };
        let mut verifier = build_verifier(b"pcs-protocol-test", b"oversized-proof", &proof);

        assert_eq!(
            read_opening_proof(&mut verifier),
            Err(CommitError::MalformedProof)
        );
    }

    proptest! {
        #[test]
        fn statement_binding_matches_between_roles(
            root in any::<[u8; 32]>(),
            point_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..32),
            target_words in (any::<u64>(), any::<u64>()),
        ) {
            let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
            let query = OpeningQuery {
                point: point_words
                    .iter()
                    .map(|&(lo, hi)| F128::new(lo, hi))
                    .collect(),
                target: F128::new(target_words.0, target_words.1),
            };
            let scoped_query = ScopedOpeningQuery::new(7, &query);

            let mut prover = build_prover(b"pcs-protocol-test", b"statement-binding");
            bind_statement(
                &pcs,
                &root,
                core::slice::from_ref(&scoped_query),
                &mut prover,
            );
            let expected = prover.verifier_message::<F128>();
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"statement-binding",
                &proof,
            );
            bind_statement(
                &pcs,
                &root,
                core::slice::from_ref(&scoped_query),
                &mut verifier,
            );
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }
    }

    fn statement_challenge(queries: &[ScopedOpeningQuery<'_>]) -> F128 {
        let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let mut prover = build_prover(b"pcs-protocol-test", b"batch-binding");
        bind_statement(&pcs, &[7; 32], queries, &mut prover);
        prover.verifier_message::<F128>()
    }

    #[test]
    fn statement_binding_commits_to_batch_order_and_count() {
        let first = OpeningQuery {
            point: vec![F128::from(1u64), F128::from(2u64)],
            target: F128::from(3u64),
        };
        let second = OpeningQuery {
            point: vec![F128::from(4u64), F128::from(5u64)],
            target: F128::from(6u64),
        };

        let ordered = statement_challenge(&[
            ScopedOpeningQuery::new(0, &first),
            ScopedOpeningQuery::new(2, &second),
        ]);
        let reversed = statement_challenge(&[
            ScopedOpeningQuery::new(0, &second),
            ScopedOpeningQuery::new(2, &first),
        ]);
        let changed_scope = statement_challenge(&[
            ScopedOpeningQuery::new(0, &first),
            ScopedOpeningQuery::new(3, &second),
        ]);
        let prefix = statement_challenge(&[ScopedOpeningQuery::new(0, &first)]);

        assert_ne!(ordered, reversed);
        assert_ne!(ordered, changed_scope);
        assert_ne!(ordered, prefix);
    }

    fn ring_switch_challenges(last_value: FlockF128) -> (Vec<FlockF128>, Vec<FlockF128>) {
        let first = [FlockF128::new(3, 5); 1 << LOG_PACKING];
        let mut last = [FlockF128::new(7, 11); 1 << LOG_PACKING];
        last[1 << (LOG_PACKING - 1)] = last_value;
        let mut prover = build_prover(b"pcs-protocol-test", b"ring-switch-schedule");
        let mut challenger = ProverChallenger::new(&mut prover);

        let r_dprime = sample_shared_ring_switch_point(
            &mut challenger,
            [(0, first.as_slice()), (2, last.as_slice())],
        );
        let etas = sample_batching_scalars(&mut challenger, 2);
        (r_dprime, etas)
    }

    #[test]
    fn last_ring_switch_message_changes_shared_point_and_later_etas() {
        let original = ring_switch_challenges(FlockF128::new(13, 17));
        let changed = ring_switch_challenges(FlockF128::new(19, 23));

        assert_ne!(original.0, changed.0);
        assert_ne!(original.1, changed.1);
    }

    #[test]
    fn batch_validation_requires_strictly_increasing_scopes() {
        let query = OpeningQuery {
            point: vec![F128::default(); 22],
            target: F128::default(),
        };

        assert_eq!(
            validate_batch(
                &[
                    ScopedOpeningQuery::new(2, &query),
                    ScopedOpeningQuery::new(2, &query),
                ],
                22,
            ),
            Err(CommitError::InvalidClaimScopeOrder)
        );
        assert_eq!(
            validate_batch(
                &[
                    ScopedOpeningQuery::new(3, &query),
                    ScopedOpeningQuery::new(2, &query),
                ],
                22,
            ),
            Err(CommitError::InvalidClaimScopeOrder)
        );
    }
}
