//! Verifier flow for explicit inner-product openings.

use field::F128;
use transcript::VerifierState;

use super::ring_switch::RingSwitch;
use super::validate_profile;
use crate::ligerito;
use crate::utils::{
    bind_inner_product_statement, read_inner_product_claims,
    sample_inner_product_batching_challenges,
};
use crate::{CommitError, Pcs, Root, StatementBinding};

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Root,
    weights: &[F128],
    target: F128,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    validate_profile(pcs)?;
    let ring_switch = RingSwitch::new(weights, pcs.packed_len())?;

    if statement_binding == StatementBinding::Bind {
        bind_inner_product_statement(pcs, &commitment.0, weights, target, transcript);
    }

    let proof = ligerito::read_proof(pcs, commitment, transcript)?;

    let claims = read_inner_product_claims(transcript)?;
    if !ring_switch.target_matches(&claims, target) {
        return Err(CommitError::VerificationFailed);
    }
    let challenge = sample_inner_product_batching_challenges(transcript);
    let reduced_claim = ring_switch.reduce_verifier(&claims, &challenge);
    ligerito::verify_dense(pcs, commitment, &proof, reduced_claim, transcript)
}
