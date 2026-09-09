//! Transcript orchestration for MLE and inner-product openings.

use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::pack::PACKING_WIDTH as CLAIM_COUNT;
use transcript::{ProverState, PublicTranscript, VerifierState};

use crate::bridge::{as_flock_f128, from_flock_f128};
use crate::ligerito::{self, ReducedProver};
use crate::{
    LigeritoProfile, OpeningQuery, Pcs, ProverData, Root, StatementBinding, inner_product, mle,
};

const MLE_STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v1";
const INNER_PRODUCT_STATEMENT_LABEL: &[u8] = b"f2z/pcs/bit-inner-product/v1";
const CHALLENGES_LABEL: &[u8] = b"f2z/pcs/ring-switch-challenges/v1";
const WEIGHT_DIGEST_LABEL: &[u8] = b"f2z/pcs/bit-inner-product-weights/v1";

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
        OpeningQuery::InnerProduct { claim } => {
            validate_inner_product_profile(pcs)?;
            let ring_switch = inner_product::RingSwitch::new(
                claim.row_weights(),
                claim.column_weights(),
                pcs.bit_len(),
            )?;
            let prover = ReducedProver::new(pcs, data, packed_witness)?;

            if statement_binding == StatementBinding::Bind {
                bind_inner_product_statement(
                    pcs,
                    &data.commitment().root,
                    &ring_switch,
                    claim.target(),
                    transcript,
                );
            }

            let claims = ring_switch.prepare_claims(prover.witness(), claim.target())?;
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
        OpeningQuery::InnerProduct { claim } => {
            validate_inner_product_profile(pcs)?;
            let ring_switch = inner_product::RingSwitch::new(
                claim.row_weights(),
                claim.column_weights(),
                pcs.bit_len(),
            )?;

            if statement_binding == StatementBinding::Bind {
                bind_inner_product_statement(
                    pcs,
                    &commitment.0,
                    &ring_switch,
                    claim.target(),
                    transcript,
                );
            }

            let proof = ligerito::read_proof(pcs, commitment, transcript)?;
            let claims = read_claims(transcript, query)?;
            let batching_point = sample_challenges(transcript);
            let dense_reduction =
                ring_switch.reduce_verified(&claims, claim.target(), &batching_point)?;
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

/// Writes one fixed ring-switch claim array.
fn write_claims(
    transcript: &mut ProverState,
    query: &OpeningQuery,
    claims: &[FlockF128; CLAIM_COUNT],
) {
    transcript.public_message(query.label());
    let claims: [field::F128; CLAIM_COUNT] =
        core::array::from_fn(|index| from_flock_f128(claims[index]));
    transcript.prover_message(&claims);
}

/// Reads one fixed ring-switch claim array.
fn read_claims(
    transcript: &mut VerifierState<'_>,
    query: &OpeningQuery,
) -> Result<[FlockF128; CLAIM_COUNT], VerifyError> {
    transcript.public_message(query.label());
    transcript
        .prover_message::<[field::F128; CLAIM_COUNT]>()
        .map(|claims| claims.map(as_flock_f128))
        .map_err(|_| VerifyError::MalformedProof)
}

/// Samples one fixed group of ring-switch challenges.
fn sample_challenges<const N: usize>(transcript: &mut impl PublicTranscript) -> [FlockF128; N] {
    transcript.public_message(CHALLENGES_LABEL);
    core::array::from_fn(|_| as_flock_f128(transcript.verifier_message_f128()))
}

/// Absorbs an MLE statement in either transcript.
fn bind_mle_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    point: &[field::F128],
    target: field::F128,
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

/// Absorbs the expanded inner-product statement without allocating all weights.
fn bind_inner_product_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    ring_switch: &inner_product::RingSwitch<'_>,
    target: field::F128,
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(INNER_PRODUCT_STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(ring_switch.bit_len() as u64));
    transcript.public_message(&weight_digest(ring_switch));
    transcript.public_message(&target);
}

fn weight_digest(ring_switch: &inner_product::RingSwitch<'_>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(WEIGHT_DIGEST_LABEL);
    hasher.update(&(ring_switch.bit_len() as u64).to_le_bytes());
    for block in ring_switch.weight_blocks() {
        for weight in block {
            hasher.update(&weight.to_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

// @dev: round trip prove/verify tests are not included here. They are covered in `tests` module as round-trip tests
#[cfg(test)]
mod tests {
    use ::transcript::{build_prover, build_verifier};
    use common::{LinearClaim, Shape};
    use field::F128;
    use proptest::prelude::*;

    use super::*;
    use crate::{HashKind, LigeritoProfile};

    #[test]
    fn fixed_claim_arrays_round_trip_and_separate_domains() {
        let claims = core::array::from_fn(|index| FlockF128::new(index as u64, 0));
        let mut challenges = [F128::default(); 2];

        for (index, query) in [
            OpeningQuery::Mle {
                point: Vec::new(),
                target: F128::default(),
            },
            OpeningQuery::InnerProduct {
                claim: LinearClaim::from_shape(
                    &Shape::new(7, 15).unwrap(),
                    vec![F128::default(); 128],
                    vec![F128::default(); 1 << 15],
                    F128::default(),
                )
                .unwrap(),
            },
        ]
        .iter()
        .enumerate()
        {
            let mut prover = build_prover(b"pcs-protocol-test", b"fixed-claims");
            write_claims(&mut prover, query, &claims);
            challenges[index] = prover.verifier_message();
            let proof = prover.finish();
            assert_eq!(proof.narg_string.len(), CLAIM_COUNT * 16);

            let mut verifier = build_verifier(b"pcs-protocol-test", b"fixed-claims", &proof);
            assert_eq!(read_claims(&mut verifier, query).unwrap(), claims);
            assert_eq!(verifier.verifier_message::<F128>(), challenges[index]);
            verifier.check_eof().unwrap();
        }

        assert_ne!(challenges[0], challenges[1]);
    }

    proptest! {
        #[test]
        fn mle_statement_binding_matches_between_roles(
            root in any::<[u8; 32]>(),
            point_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..32),
            target_words in (any::<u64>(), any::<u64>()),
        ) {
            let shape = Shape::new(7, 15).unwrap();
            let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
            let point = point_words
                .iter()
                .map(|&(lo, hi)| F128::new(lo, hi))
                .collect::<Vec<_>>();
            let target = F128::new(target_words.0, target_words.1);

            let mut prover = build_prover(b"pcs-protocol-test", b"statement-binding");
            bind_mle_statement(&pcs, &root, &point, target, &mut prover);
            let expected = prover.verifier_message::<F128>();
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"statement-binding",
                &proof,
            );
            bind_mle_statement(&pcs, &root, &point, target, &mut verifier);
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }

        #[test]
        fn inner_product_statement_binding_matches_between_roles(
            root in any::<[u8; 32]>(),
            row_words in prop::collection::vec((any::<u64>(), any::<u64>()), 256),
            column_words in prop::collection::vec((any::<u64>(), any::<u64>()), 1..5),
            target_words in (any::<u64>(), any::<u64>()),
        ) {
            let shape = Shape::new(7, 15).unwrap();
            let pcs = Pcs::new(&shape, LigeritoProfile::Secure, HashKind::Blake3).unwrap();
            let rows = row_words.iter().map(|&(lo, hi)| F128::new(lo, hi)).collect::<Vec<_>>();
            let columns = column_words.iter().map(|&(lo, hi)| F128::new(lo, hi)).collect::<Vec<_>>();
            let bit_len = rows.len() * columns.len();
            let ring_switch = inner_product::RingSwitch::new(&rows, &columns, bit_len).unwrap();
            let target = F128::new(target_words.0, target_words.1);

            let mut prover = build_prover(b"pcs-protocol-test", b"inner-product-statement");
            bind_inner_product_statement(&pcs, &root, &ring_switch, target, &mut prover);
            let expected = prover.verifier_message::<F128>();

            let weights = (0..bit_len)
                .map(|index| columns[index / rows.len()] * rows[index % rows.len()])
                .collect::<Vec<_>>();
            let mut hasher = blake3::Hasher::new();
            hasher.update(WEIGHT_DIGEST_LABEL);
            hasher.update(&(weights.len() as u64).to_le_bytes());
            for weight in &weights {
                hasher.update(&weight.to_bytes());
            }
            let mut legacy = build_prover(b"pcs-protocol-test", b"inner-product-statement");
            legacy.public_message(INNER_PRODUCT_STATEMENT_LABEL);
            legacy.public_message(&root);
            legacy.public_message(&pcs);
            legacy.public_message(&(weights.len() as u64));
            legacy.public_message(hasher.finalize().as_bytes());
            legacy.public_message(&target);
            prop_assert_eq!(legacy.verifier_message::<F128>(), expected);
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"inner-product-statement",
                &proof,
            );
            bind_inner_product_statement(&pcs, &root, &ring_switch, target, &mut verifier);
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }
    }
}
