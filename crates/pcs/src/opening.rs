//! Transcript orchestration for MLE and inner-product openings.

use field::F128;
use transcript::{ProverState, VerifierState};

use crate::ligerito::{self, ReducedProver};
use crate::utils::{
    bind_inner_product_statement, bind_mle_statement, read_claims, sample_challenges, write_claims,
};
use crate::{
    LigeritoProfile, OpeningQuery, Pcs, ProverData, Root, StatementBinding, inner_product, mle,
};

/// Errors from opening proof creation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProveError {
    /// The packed witness length does not match the configured polynomial.
    PackedWitnessLengthMismatch,
    /// The weight count does not match the configured polynomial.
    WeightLengthMismatch,
    /// The evaluation point length does not match the committed polynomial.
    PointLengthMismatch,
    /// The retained prover data uses different PCS parameters.
    ProverDataMismatch,
    /// The claimed target does not match the supplied witness.
    InvalidClaim,
    /// The selected profile does not support arbitrary inner products.
    UnsupportedInnerProductProfile,
    /// The opening proof could not be serialized.
    SerializationFailed,
    /// The serialized opening proof exceeds the transcript hint limit.
    ProofTooLarge,
    /// An internal PCS invariant failed.
    Internal,
}

/// Errors from opening proof verification.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifyError {
    /// The weight count does not match the configured polynomial.
    WeightLengthMismatch,
    /// The evaluation point length does not match the committed polynomial.
    PointLengthMismatch,
    /// The selected profile does not support arbitrary inner products.
    UnsupportedInnerProductProfile,
    /// The transcript does not contain one complete canonical opening proof.
    MalformedProof,
    /// The opening proof does not verify against the statement.
    VerificationFailed,
    /// An internal PCS invariant failed.
    Internal,
}

/// Query validation shared by prover and verifier entry points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum QueryError {
    WeightLengthMismatch,
    PointLengthMismatch,
    UnsupportedInnerProductProfile,
    Internal,
}

impl From<QueryError> for ProveError {
    fn from(error: QueryError) -> Self {
        match error {
            QueryError::WeightLengthMismatch => Self::WeightLengthMismatch,
            QueryError::PointLengthMismatch => Self::PointLengthMismatch,
            QueryError::UnsupportedInnerProductProfile => Self::UnsupportedInnerProductProfile,
            QueryError::Internal => Self::Internal,
        }
    }
}

impl From<QueryError> for VerifyError {
    fn from(error: QueryError) -> Self {
        match error {
            QueryError::WeightLengthMismatch => Self::WeightLengthMismatch,
            QueryError::PointLengthMismatch => Self::PointLengthMismatch,
            QueryError::UnsupportedInnerProductProfile => Self::UnsupportedInnerProductProfile,
            QueryError::Internal => Self::Internal,
        }
    }
}

pub(crate) fn prove(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    query: &OpeningQuery,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), ProveError> {
    match query {
        OpeningQuery::Mle { point, target } => {
            let ring_switch = mle::RingSwitch::new(point, pcs.params().m)?;
            let prover = ReducedProver::new(pcs, data, packed_witness)?;

            if statement_binding == StatementBinding::Bind {
                bind_mle_statement(pcs, &data.commitment().root, point, *target, transcript);
            }

            let prepared_claims = ring_switch.prepare_claims(prover.witness(), *target)?;
            write_claims(transcript, query, &prepared_claims.claims);
            let batching_point = sample_challenges(transcript);
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
            write_claims(transcript, query, &claims);
            let batching_point = sample_challenges(transcript);
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
) -> Result<(), VerifyError> {
    match query {
        OpeningQuery::Mle { point, target } => {
            let ring_switch = mle::RingSwitch::new(point, pcs.params().m)?;

            if statement_binding == StatementBinding::Bind {
                bind_mle_statement(pcs, &commitment.0, point, *target, transcript);
            }

            let proof = ligerito::read_proof(pcs, commitment, transcript)?;
            let claims = read_claims(transcript, query)?;
            if !ring_switch.target_matches(&claims, *target) {
                return Err(VerifyError::VerificationFailed);
            }

            let batching_point = sample_challenges(transcript);
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
            let claims = read_claims(transcript, query)?;
            let batching_point = sample_challenges(transcript);
            let dense_reduction = ring_switch.reduce_verified(&claims, *target, &batching_point)?;
            ligerito::verify_dense(pcs, commitment, &proof, dense_reduction, transcript)
        }
    }
}

fn validate_inner_product_profile(pcs: &Pcs) -> Result<(), QueryError> {
    if pcs.params().profile != LigeritoProfile::Secure {
        return Err(QueryError::UnsupportedInnerProductProfile);
    }
    Ok(())
}
