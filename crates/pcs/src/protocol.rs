//! Shared encoding and transcript rules for multilinear openings.

use bincode::Options;
use flock_core::pcs::BatchOpeningProofLigerito;
use transcript::{ProverState, VerifierState};

use crate::{CommitError, OpeningQuery, Pcs};

pub(crate) const PROOF_HINT_LIMIT: usize = 64 * 1024 * 1024;
pub(crate) const STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v1";
pub(crate) const RING_SWITCH_LABEL: &[u8] = b"flock-ring-switch-v0";

pub(crate) fn write_opening_proof(
    proof: &BatchOpeningProofLigerito,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    let proof_bytes = proof_options()
        .serialize(proof)
        .map_err(|_| CommitError::Flock)?;
    if proof_bytes.len() > PROOF_HINT_LIMIT {
        return Err(CommitError::Flock);
    }
    transcript.hint_bytes(&proof_bytes);
    Ok(())
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

/// Absorbs the public statement in the prover transcript.
pub(crate) fn bind_statement_prover(
    pcs: &Pcs,
    root: &[u8; 32],
    query: &OpeningQuery,
    transcript: &mut ProverState,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);

    for tag in pcs.statement_tags() {
        transcript.public_message(&tag);
    }

    transcript.public_message(&(query.point.len() as u64));
    for coordinate in &query.point {
        transcript.public_message(coordinate);
    }
    transcript.public_message(&query.target);
}

/// Absorbs the public statement in the verifier transcript.
pub(crate) fn bind_statement_verifier(
    pcs: &Pcs,
    root: &[u8; 32],
    query: &OpeningQuery,
    transcript: &mut VerifierState<'_>,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);

    for tag in pcs.statement_tags() {
        transcript.public_message(&tag);
    }

    transcript.public_message(&(query.point.len() as u64));
    for coordinate in &query.point {
        transcript.public_message(coordinate);
    }
    transcript.public_message(&query.target);
}

fn proof_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(PROOF_HINT_LIMIT as u64)
        .reject_trailing_bytes()
}
