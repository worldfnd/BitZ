//! Prover flow for multilinear openings.

use field::F128;
use transcript::ProverState;

use super::ring_switch::RingSwitch;
use crate::ligerito::ReducedProver;
use crate::utils::{ClaimDomain, bind_statement, sample_ring_switch_point, write_claims};
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
    write_claims(transcript, ClaimDomain::Mle, prepared_claims.claims());
    let challenge = sample_ring_switch_point(transcript);
    let reduced_claim = prepared_claims.reduce(&challenge);
    prover.prove(reduced_claim, transcript)
}
