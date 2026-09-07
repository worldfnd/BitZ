//! Transcript orchestration for MLE and inner-product openings.

use field::F128;
use transcript::{ProverState, VerifierState};

use crate::ligerito::{self, ReducedProver};
use crate::utils::{
    ClaimDomain, bind_inner_product_statement, bind_mle_statement, read_claims,
    sample_inner_product_batching_point, sample_mle_batching_point, write_claims,
};
use crate::{
    CommitError, LigeritoProfile, OpeningQuery, Pcs, ProverData, Root, StatementBinding,
    inner_product, mle,
};

pub(crate) fn prove(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    query: &OpeningQuery,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    match query {
        OpeningQuery::Mle { point, target } => {
            let ring_switch = mle::RingSwitch::new(point, pcs.params().m)?;
            let prover = ReducedProver::new(pcs, data, packed_witness)?;

            if statement_binding == StatementBinding::Bind {
                bind_mle_statement(pcs, &data.commitment().root, point, *target, transcript);
            }

            let prepared_claims = ring_switch.prepare_claims(prover.witness(), *target)?;
            write_claims(transcript, ClaimDomain::Mle, &prepared_claims.claims);
            let batching_point = sample_mle_batching_point(transcript);
            let dense_reduction = prepared_claims.reduce_dense(&batching_point);
            prover.prove(dense_reduction, transcript)
        }
        OpeningQuery::InnerProduct { weights, target } => {
            validate_inner_product_profile(pcs)?;
            let ring_switch = inner_product::RingSwitch::new(weights, pcs.bit_len())?;
            let prover = ReducedProver::new(pcs, data, packed_witness)?;

            if statement_binding == StatementBinding::Bind {
                bind_inner_product_statement(
                    pcs,
                    &data.commitment().root,
                    weights,
                    *target,
                    transcript,
                );
            }

            let claims = ring_switch.prepare_claims(prover.witness(), *target)?;
            write_claims(transcript, ClaimDomain::InnerProduct, &claims);
            let batching_point = sample_inner_product_batching_point(transcript);
            let dense_reduction = ring_switch.reduce_dense(&claims, &batching_point);
            prover.prove(dense_reduction, transcript)
        }
    }
}

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Root,
    query: &OpeningQuery,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    match query {
        OpeningQuery::Mle { point, target } => {
            let ring_switch = mle::RingSwitch::new(point, pcs.params().m)?;

            if statement_binding == StatementBinding::Bind {
                bind_mle_statement(pcs, &commitment.0, point, *target, transcript);
            }

            let proof = ligerito::read_proof(pcs, commitment, transcript)?;
            let claims = read_claims(transcript, ClaimDomain::Mle)?;
            if !ring_switch.target_matches(&claims, *target) {
                return Err(CommitError::VerificationFailed);
            }

            let batching_point = sample_mle_batching_point(transcript);
            let succinct_reduction = ring_switch.reduce_succinct(&claims, &batching_point);
            ligerito::verify_succinct(
                pcs,
                commitment,
                &proof,
                ring_switch.suffix_dimension(),
                succinct_reduction.packed_target,
                |ris, yr_log_n| succinct_reduction.evaluate_basis(ris, yr_log_n),
                transcript,
            )
        }
        OpeningQuery::InnerProduct { weights, target } => {
            validate_inner_product_profile(pcs)?;
            let ring_switch = inner_product::RingSwitch::new(weights, pcs.bit_len())?;

            if statement_binding == StatementBinding::Bind {
                bind_inner_product_statement(pcs, &commitment.0, weights, *target, transcript);
            }

            let proof = ligerito::read_proof(pcs, commitment, transcript)?;
            let claims = read_claims(transcript, ClaimDomain::InnerProduct)?;
            if !ring_switch.target_matches(&claims, *target) {
                return Err(CommitError::VerificationFailed);
            }

            let batching_point = sample_inner_product_batching_point(transcript);
            let dense_reduction = ring_switch.reduce_dense(&claims, &batching_point);
            ligerito::verify_dense(pcs, commitment, &proof, dense_reduction, transcript)
        }
    }
}

fn validate_inner_product_profile(pcs: &Pcs) -> Result<(), CommitError> {
    if pcs.params().profile != LigeritoProfile::Secure {
        return Err(CommitError::UnsupportedInnerProductProfile);
    }
    Ok(())
}
