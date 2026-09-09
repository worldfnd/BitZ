//! Bounded hint transport for the Ligerito proof.
//!
//! Hint bytes do not affect transcript challenges.

use bincode::Options;
use flock_core::pcs::ligerito::LigeritoProof;
use transcript::{ProverState, VerifierState};

use crate::{ProveError, VerifyError};

const PROOF_HINT_LIMIT: usize = 64 * 1024 * 1024;

pub(crate) fn write_opening_proof(
    proof: &LigeritoProof,
    transcript: &mut ProverState,
) -> Result<(), ProveError> {
    let proof_bytes = proof_options()
        .serialize(proof)
        .map_err(map_serialization_error)?;
    transcript.hint_bytes(&proof_bytes);
    Ok(())
}

pub(crate) fn read_opening_proof(
    transcript: &mut VerifierState<'_>,
) -> Result<LigeritoProof, VerifyError> {
    let proof_bytes = transcript
        .hint_bytes(PROOF_HINT_LIMIT)
        .map_err(|_| VerifyError::MalformedProof)?;
    proof_options()
        .deserialize(&proof_bytes)
        .map_err(|_| VerifyError::MalformedProof)
}

fn map_serialization_error(error: bincode::Error) -> ProveError {
    if matches!(error.as_ref(), bincode::ErrorKind::SizeLimit) {
        ProveError::ProofTooLarge
    } else {
        ProveError::SerializationFailed
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
    use transcript::{NargSerialize, Proof, build_prover, build_verifier};

    use super::*;

    #[test]
    fn proof_serialization_errors_are_specific() {
        assert_eq!(
            map_serialization_error(Box::new(bincode::ErrorKind::SizeLimit)),
            ProveError::ProofTooLarge
        );
        assert_eq!(
            map_serialization_error(Box::new(bincode::ErrorKind::Custom(
                "serialization failed".to_owned(),
            ))),
            ProveError::SerializationFailed
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
            Err(VerifyError::MalformedProof)
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
            Err(VerifyError::MalformedProof)
        );
    }
}
