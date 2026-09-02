//! Prover flow for multilinear openings.

use field::F128;
use flock_core::pcs::RingSwitchProof;
use transcript::ProverState;

use super::ring_switch::RingSwitch;
use crate::ligerito::ReducedProver;
use crate::utils::{bind_ring_switch_message, bind_statement, sample_ring_switch_point};
use crate::{CommitError, Pcs, ProverData, StatementBinding};

pub(crate) fn open(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    point: &[F128],
    target: F128,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    let ring_switch = RingSwitch::new(point, pcs.params().m)?;
    let prover = ReducedProver::new(pcs, data, packed_witness)?;

    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, &data.commitment().root, point, target, transcript);
    }

    let prepared_claims = ring_switch.prepare_claims(prover.witness(), target)?;
    bind_ring_switch_message(transcript, prepared_claims.claims().as_array())?;
    let proof_claims = prepared_claims.claims().as_array().to_vec();

    let challenge = sample_ring_switch_point(transcript);
    let reduced_claim = prepared_claims.reduce(&challenge);
    prover.prove(
        reduced_claim,
        vec![RingSwitchProof {
            s_hat_v: proof_claims,
        }],
        transcript,
    )
}
