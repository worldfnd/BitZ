//! Verifier flow for multilinear openings.

use field::F128;
use transcript::VerifierState;

use super::ring_switch::RingSwitch;
use crate::ligerito;
use crate::utils::{ClaimDomain, bind_statement, read_claims, sample_ring_switch_point};
use crate::{CommitError, Pcs, Root, StatementBinding};

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Root,
    point: &[F128],
    target: F128,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    let ring_switch = RingSwitch::new(point, pcs.params().m)?;

    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, &commitment.0, point, target, transcript);
    }

    let proof = ligerito::read_proof(pcs, commitment, transcript)?;

    let claims = read_claims(transcript, ClaimDomain::Mle)?;
    if !ring_switch.target_matches(&claims, target) {
        return Err(CommitError::VerificationFailed);
    }

    let challenge = sample_ring_switch_point(transcript);
    let reduced_claim = ring_switch.reduce_verifier(&claims, &challenge);
    ligerito::verify_succinct(
        pcs,
        commitment,
        &proof,
        ring_switch.suffix_dimension(),
        reduced_claim.packed_target,
        |ris, yr_log_n| reduced_claim.evaluate_basis(ris, yr_log_n),
        transcript,
    )
}
