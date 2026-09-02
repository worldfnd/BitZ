//! Verifier flow for explicit inner-product openings.

use field::F128;
use flock_core::pcs::ligerito::recursive_verifier_with_basis;
use transcript::VerifierState;

use super::ring_switch::{CoordinateClaims, InnerProductRingSwitch, ReducedClaim};
use super::validate_profile;
use crate::challenger::VerifierChallenger;
use crate::utils::{
    bind_inner_product_statement, observe_opening_target, read_inner_product_coordinates,
    read_opening_proof, sample_inner_product_batching_challenges,
};
use crate::validation::validate_ligerito_proof_shape;
use crate::{CommitError, Commitment, Pcs, StatementBinding};

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Commitment,
    weights: &[F128],
    claimed_target: F128,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    validate_profile(pcs)?;
    let ring_switch = InnerProductRingSwitch::new(weights, pcs.packed_len())?;
    let ligerito_config = pcs.verifier_config();

    if statement_binding == StatementBinding::Bind {
        bind_inner_product_statement(pcs, commitment.root(), weights, claimed_target, transcript);
    }

    let proof = read_opening_proof(transcript)?;
    if !proof.ring_switches.is_empty() {
        return Err(CommitError::VerificationFailed);
    }
    validate_ligerito_proof_shape(
        &proof.ligerito,
        ligerito_config,
        pcs.final_log_n(),
        commitment.root(),
    )?;

    let coordinate_claims =
        CoordinateClaims::from_array(read_inner_product_coordinates(transcript)?);
    if coordinate_claims.reconstructed_target() != claimed_target {
        return Err(CommitError::VerificationFailed);
    }
    let batching_challenges = sample_inner_product_batching_challenges(transcript);
    let ReducedClaim {
        packed_basis,
        packed_target,
    } = ring_switch.reduce(&coordinate_claims, &batching_challenges);

    observe_opening_target(transcript, pcs.opening_log_n(), packed_target);
    let mut challenger = VerifierChallenger::new_ligerito(transcript, packed_target);
    let valid = recursive_verifier_with_basis(
        ligerito_config,
        &proof.ligerito,
        &packed_basis,
        packed_target,
        commitment.root(),
        &mut challenger,
    );
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }
    if !valid {
        return Err(CommitError::VerificationFailed);
    }
    Ok(())
}
