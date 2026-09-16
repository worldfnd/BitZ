//! Transcript orchestration for MLE openings and inner-product sumcheck reduction.

use common::LinearClaim;
use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::pack::PACKING_WIDTH as CLAIM_COUNT;
use transcript::{ProverState, PublicTranscript, VerifierState};

use crate::bridge::{as_flock_f128, as_flock_f128s, from_flock_f128};
use crate::ligerito::{self, ReducedProver, validate_prover_data};
use crate::{OpeningQuery, Pcs, ProverData, Root, StatementBinding, mle};

const MLE_STATEMENT_LABEL: &[u8] = b"bitz/pcs/mle-opening/v1";
const INNER_PRODUCT_STATEMENT_LABEL: &[u8] = b"bitz/pcs/bit-inner-product/v3";
const INNER_PRODUCT_DIGEST_CONTEXT: &str = "bitz/pcs/bit-inner-product-weights/v1";
const SUMCHECK_LABEL: &[u8] = b"bitz/pcs/inner-product-sumcheck/v2";
const MLE_CLAIMS_LABEL: &[u8] = b"bitz/pcs/mle-claims/v1";
const CHALLENGES_LABEL: &[u8] = b"bitz/pcs/ring-switch-challenges/v1";

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
    Internal,
}

impl From<QueryError> for ProveError {
    fn from(error: QueryError) -> Self {
        match error {
            QueryError::WeightLengthMismatch => Self::WeightLengthMismatch,
            QueryError::PointLengthMismatch => Self::PointLengthMismatch,
            QueryError::Internal => Self::Internal,
        }
    }
}

impl From<QueryError> for VerifyError {
    fn from(error: QueryError) -> Self {
        match error {
            QueryError::WeightLengthMismatch => Self::WeightLengthMismatch,
            QueryError::PointLengthMismatch => Self::PointLengthMismatch,
            QueryError::Internal => Self::Internal,
        }
    }
}

impl From<post_gkr::ProveError> for ProveError {
    fn from(error: post_gkr::ProveError) -> Self {
        match error {
            post_gkr::ProveError::WitnessLengthMismatch => Self::PackedWitnessLengthMismatch,
            post_gkr::ProveError::ClaimDoesNotHold => Self::InvalidClaim,
        }
    }
}

impl From<post_gkr::VerifyError> for VerifyError {
    fn from(error: post_gkr::VerifyError) -> Self {
        match error {
            post_gkr::VerifyError::MalformedProof => Self::MalformedProof,
            post_gkr::VerifyError::EvaluationMismatch => Self::VerificationFailed,
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
            prove_mle(prover, ring_switch, *target, transcript)
        }
        OpeningQuery::InnerProduct { claim } => {
            validate_inner_product_claim(pcs, claim)?;
            validate_prover_data(pcs, data)?;
            if statement_binding == StatementBinding::Bind {
                bind_inner_product_statement(pcs, &data.commitment().root, claim, transcript);
            }
            transcript.public_message(SUMCHECK_LABEL);
            let reduced = post_gkr::prove(claim, &packed_witness, transcript)?;
            // The evaluation claim the sumcheck leaves is opened like any
            // other, and bound whatever the caller's mode: `AlreadyBound`
            // covers the original claim only.
            prove(
                pcs,
                data,
                packed_witness,
                &reduced,
                StatementBinding::Bind,
                transcript,
            )
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
            verify_mle(pcs, commitment, ring_switch, *target, transcript)
        }
        OpeningQuery::InnerProduct { claim } => {
            validate_inner_product_claim(pcs, claim)?;
            if statement_binding == StatementBinding::Bind {
                bind_inner_product_statement(pcs, &commitment.0, claim, transcript);
            }
            transcript.public_message(SUMCHECK_LABEL);
            let reduced = post_gkr::verify(claim, transcript)?;
            verify(
                pcs,
                commitment,
                &reduced,
                StatementBinding::Bind,
                transcript,
            )
        }
    }
}

fn validate_inner_product_claim(pcs: &Pcs, claim: &LinearClaim<F128>) -> Result<(), QueryError> {
    // LinearClaim::from_shape checks each factor against a valid Shape.
    if claim
        .row_weights()
        .len()
        .checked_mul(claim.column_weights().len())
        != Some(pcs.bit_len())
    {
        return Err(QueryError::WeightLengthMismatch);
    }
    Ok(())
}

/// Proves an MLE claim after its statement enters the transcript.
fn prove_mle(
    prover: ReducedProver<'_>,
    ring_switch: mle::RingSwitch<'_>,
    target: F128,
    transcript: &mut ProverState,
) -> Result<(), ProveError> {
    let prepared_claims = ring_switch.prepare_claims(as_flock_f128s(prover.witness()), target)?;
    write_claims(transcript, &prepared_claims.claims);
    let batching_point = sample_challenges(transcript);
    let dense_reduction = prepared_claims.reduce_dense(&batching_point);
    prover.prove(dense_reduction, transcript)
}

/// Verifies an MLE claim after its statement enters the transcript.
fn verify_mle(
    pcs: &Pcs,
    commitment: &Root,
    ring_switch: mle::RingSwitch<'_>,
    target: F128,
    transcript: &mut VerifierState<'_>,
) -> Result<(), VerifyError> {
    let proof = ligerito::read_proof(pcs, commitment, transcript)?;
    let claims = read_claims(transcript)?;
    if !ring_switch.target_matches(&claims, target) {
        return Err(VerifyError::VerificationFailed);
    }
    let batching_point = sample_challenges(transcript);
    let reduction = ring_switch.reduce_succinct(&claims, &batching_point);
    ligerito::verify_succinct(
        pcs,
        commitment,
        &proof,
        ring_switch.suffix_dimension(),
        reduction.packed_target,
        |ris, yr_log_n| reduction.evaluate_basis(ris, yr_log_n),
        transcript,
    )
}

/// Writes the fixed MLE ring-switch claim array.
fn write_claims(transcript: &mut ProverState, claims: &[FlockF128; CLAIM_COUNT]) {
    transcript.public_message(MLE_CLAIMS_LABEL);
    let claims: [F128; CLAIM_COUNT] = core::array::from_fn(|index| from_flock_f128(claims[index]));
    transcript.prover_message(&claims);
}

/// Reads the fixed MLE ring-switch claim array.
fn read_claims(
    transcript: &mut VerifierState<'_>,
) -> Result<[FlockF128; CLAIM_COUNT], VerifyError> {
    transcript.public_message(MLE_CLAIMS_LABEL);
    transcript
        .prover_message::<[F128; CLAIM_COUNT]>()
        .map(|claims| claims.map(as_flock_f128))
        .map_err(|_| VerifyError::MalformedProof)
}

/// Samples the seven MLE ring-switch challenges.
fn sample_challenges(transcript: &mut impl PublicTranscript) -> mle::BatchingPoint {
    transcript.public_message(CHALLENGES_LABEL);
    core::array::from_fn(|_| as_flock_f128(transcript.verifier_message_f128()))
}

/// Absorbs an MLE statement in either transcript.
fn bind_mle_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    point: &[F128],
    target: F128,
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(MLE_STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(point.len() as u64));
    for coordinate in point {
        transcript.public_message(coordinate);
    }
    transcript.public_message(&target);
}

/// Binds both tensor factors before the first sumcheck challenge: their
/// lengths, a digest of their weights, and the target. A factor can be one
/// weight per committed bit, and absorbing it whole would cost the sponge
/// as much as the sumcheck costs the prover.
fn bind_inner_product_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    claim: &LinearClaim<F128>,
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(INNER_PRODUCT_STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(claim.row_weights().len() as u64));
    transcript.public_message(&(claim.column_weights().len() as u64));

    // Weights per digest update: `2^16` elements, one megabyte.
    const INNER_PRODUCT_DIGEST_CHUNK: usize = 1 << 16;

    // Hashing this way is much faster that going through `public_message` directly.
    // Does not depend on the chunking.
    let mut hasher = blake3::Hasher::new_derive_key(INNER_PRODUCT_DIGEST_CONTEXT);
    let mut buffer = Vec::with_capacity(INNER_PRODUCT_DIGEST_CHUNK * 16);
    for factor in [claim.row_weights(), claim.column_weights()] {
        for chunk in factor.chunks(INNER_PRODUCT_DIGEST_CHUNK) {
            buffer.clear();
            for weight in chunk {
                buffer.extend_from_slice(&weight.to_bytes());
            }
            hasher.update_rayon(&buffer);
        }
    }
    transcript.public_message(hasher.finalize().as_bytes());

    transcript.public_message(&claim.target());
}

#[cfg(test)]
mod tests;
