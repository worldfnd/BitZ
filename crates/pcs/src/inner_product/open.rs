//! Prover flow for explicit inner-product openings.

use field::F128;
use transcript::ProverState;

use super::ring_switch::RingSwitch;
use super::validate_profile;
use crate::ligerito::ReducedProver;
use crate::utils::{
    bind_inner_product_statement, sample_inner_product_batching_challenges,
    write_inner_product_claims,
};
use crate::{CommitError, Pcs, ProverData, StatementBinding};

pub(crate) fn open(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    weights: &[F128],
    target: F128,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    validate_profile(pcs)?;
    let ring_switch = RingSwitch::new(weights, pcs.packed_len())?;
    let prover = ReducedProver::new(pcs, data, packed_witness)?;

    if statement_binding == StatementBinding::Bind {
        bind_inner_product_statement(pcs, &data.commitment().root, weights, target, transcript);
    }

    let prepared_claims = ring_switch.prepare_claims(prover.witness(), target)?;
    write_inner_product_claims(transcript, prepared_claims.claims());
    let challenge = sample_inner_product_batching_challenges(transcript);
    let reduced_claim = prepared_claims.reduce(&challenge);
    prover.prove(reduced_claim, transcript)
}
