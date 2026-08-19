//! Shared encoding and transcript rules for multilinear openings.

use bincode::Options;
use flock_core::pcs::BatchOpeningProofLigerito;
use transcript::{Encoding, ProverState, VerifierState};

use crate::{CommitError, OpeningQuery, Pcs};

pub(crate) const PROOF_HINT_LIMIT: usize = 64 * 1024 * 1024;
pub(crate) const STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v2";
pub(crate) const RING_SWITCH_LABEL: &[u8] = b"flock-ring-switch-v0";

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
    queries: &[OpeningQuery],
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(queries.len() as u64));
    for query in queries {
        transcript.public_message(&(query.point.len() as u64));
        for coordinate in &query.point {
            transcript.public_message(coordinate);
        }
        transcript.public_message(&query.target);
    }
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
    use crate::{HashKind, LigeritoProfile};

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

            let mut prover = build_prover(b"pcs-protocol-test", b"statement-binding");
            bind_statement(
                &pcs,
                &root,
                core::slice::from_ref(&query),
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
                core::slice::from_ref(&query),
                &mut verifier,
            );
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }
    }

    fn statement_challenge(queries: &[OpeningQuery]) -> F128 {
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

        let ordered = statement_challenge(&[first.clone(), second.clone()]);
        let reversed = statement_challenge(&[second, first.clone()]);
        let prefix = statement_challenge(&[first]);

        assert_ne!(ordered, reversed);
        assert_ne!(ordered, prefix);
    }
}
