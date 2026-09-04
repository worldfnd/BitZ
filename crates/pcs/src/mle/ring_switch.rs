//! Algebraic ring switch for multilinear openings.
//!
//! For `r_lo = r[0..7]` and `r_hi = r[7..m]`, define
//! `s_v = q_hat(r_hi, v)` and `target = sum_v eq(r_lo, v) * s_v`.
//! A fresh seven-coordinate point batches the transposed `s_v` values.
//! The prover materializes the resulting packed basis.
//! The verifier evaluates the same basis succinctly.

use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ring_switch::{
    build_eq_split, claim_check, eval_rs_eq_finish_from_prefix_binary_q, eval_rs_eq_prefix,
    fold_1b_rows_naive, fold_b128_elems, inner_product, tensor_algebra_transpose,
};
use flock_core::pcs::{LOG_PACKING, pack::PACKING_WIDTH as CLAIM_COUNT};
use flock_core::zerocheck::univariate_skip::build_eq;

use crate::CommitError;
use crate::bridge::{as_flock_f128, as_flock_f128s};
use crate::ligerito::ReducedClaim;
pub(super) type Challenge = [FlockF128; LOG_PACKING];

/// A validated MLE point split into packed and unpacked coordinates.
pub(super) struct RingSwitch<'a> {
    point: &'a [FlockF128],
}

/// Ring-switch claims with the prover data needed for challenge reduction.
pub(super) struct PreparedClaims {
    claims: [FlockF128; CLAIM_COUNT],
    suffix_tensor: Vec<FlockF128>,
}

/// One packed-field claim with a succinct basis evaluator.
pub(super) struct VerifierReduction<'a> {
    pub(super) packed_target: FlockF128,
    suffix_point: &'a [FlockF128],
    challenge_tensor: Vec<FlockF128>,
}

impl<'a> RingSwitch<'a> {
    /// Validates the point and records its field-compatible view.
    pub(super) fn new(point: &'a [F128], variable_count: usize) -> Result<Self, CommitError> {
        if point.len() != variable_count {
            return Err(CommitError::PointLengthMismatch);
        }
        if variable_count < LOG_PACKING {
            return Err(CommitError::invalid_configuration(format!(
                "PCS variable count {variable_count} is smaller than the packing width {LOG_PACKING}"
            )));
        }
        Ok(Self {
            point: as_flock_f128s(point),
        })
    }

    /// Returns the number of unpacked suffix coordinates.
    pub(super) fn suffix_dimension(&self) -> usize {
        self.point.len() - LOG_PACKING
    }

    /// Derives 128 ring-switch claims, checks their target, and retains reduction data.
    pub(super) fn prepare_claims(
        &self,
        packed_witness: &[FlockF128],
        claimed_target: F128,
    ) -> Result<PreparedClaims, CommitError> {
        let (prefix_tensor, suffix_tensor) = build_eq_split(self.point, LOG_PACKING);
        if suffix_tensor.len() != packed_witness.len() {
            return Err(CommitError::InvalidBitLength);
        }

        let claims = claims_from_prover(fold_1b_rows_naive(packed_witness, &suffix_tensor))?;
        if claim_check(&prefix_tensor, &claims) != as_flock_f128(claimed_target) {
            return Err(CommitError::InvalidClaim);
        }

        Ok(PreparedClaims {
            claims,
            suffix_tensor,
        })
    }

    /// Checks the original MLE target against proof-provided partial evaluations.
    pub(super) fn target_matches(
        &self,
        claims: &[FlockF128; CLAIM_COUNT],
        claimed_target: F128,
    ) -> bool {
        let prefix_tensor = build_eq(&self.point[..LOG_PACKING]);
        claim_check(&prefix_tensor, claims) == as_flock_f128(claimed_target)
    }

    /// Reduces proof-provided partial evaluations without materializing the basis.
    pub(super) fn reduce_verifier(
        &self,
        claims: &[FlockF128; CLAIM_COUNT],
        challenge: &Challenge,
    ) -> VerifierReduction<'a> {
        let challenge_tensor = build_eq(challenge);
        let packed_target = batch_claims(claims, &challenge_tensor);
        VerifierReduction {
            packed_target,
            suffix_point: &self.point[LOG_PACKING..],
            challenge_tensor,
        }
    }
}

impl PreparedClaims {
    pub(super) fn claims(&self) -> &[FlockF128; CLAIM_COUNT] {
        &self.claims
    }

    /// Materializes the packed basis and batches the partial evaluations.
    pub(super) fn reduce(self, challenge: &Challenge) -> ReducedClaim {
        let challenge_tensor = build_eq(challenge);
        let packed_target = batch_claims(&self.claims, &challenge_tensor);
        let packed_basis = fold_b128_elems(&self.suffix_tensor, &challenge_tensor);
        debug_assert_eq!(packed_basis.len(), self.suffix_tensor.len());

        ReducedClaim {
            packed_basis,
            packed_target,
        }
    }
}

impl VerifierReduction<'_> {
    /// Evaluates the packed basis at one recursive Ligerito residual domain.
    pub(super) fn evaluate_basis(&self, ris: &[FlockF128], yr_log_n: usize) -> Vec<FlockF128> {
        if yr_log_n > 32 || ris.len().checked_add(yr_log_n) != Some(self.suffix_point.len()) {
            return Vec::new();
        }
        let Some(yr_len) = 1usize.checked_shl(yr_log_n as u32) else {
            return Vec::new();
        };
        let prefix = eval_rs_eq_prefix(self.suffix_point, ris);
        let suffix = &self.suffix_point[ris.len()..];
        (0..yr_len)
            .map(|y| {
                eval_rs_eq_finish_from_prefix_binary_q(
                    &prefix,
                    suffix,
                    y as u32,
                    &self.challenge_tensor,
                )
            })
            .collect()
    }
}

fn batch_claims(claims: &[FlockF128; CLAIM_COUNT], challenge_tensor: &[FlockF128]) -> FlockF128 {
    debug_assert_eq!(challenge_tensor.len(), CLAIM_COUNT);
    let transposed_claims = tensor_algebra_transpose(claims);
    inner_product(&transposed_claims, challenge_tensor)
}

/// Converts the prover result into the fixed protocol shape.
fn claims_from_prover(values: Vec<FlockF128>) -> Result<[FlockF128; CLAIM_COUNT], CommitError> {
    let values: [FlockF128; CLAIM_COUNT] =
        values.try_into().map_err(|values: Vec<FlockF128>| {
            CommitError::invalid_configuration(format!(
                "Flock produced {} MLE ring-switch claims, expected {CLAIM_COUNT}",
                values.len(),
            ))
        })?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prover_and_verifier_reductions_have_the_same_target() {
        let point = vec![F128::from(2u64); LOG_PACKING + 2];
        let packed_witness = vec![FlockF128::ZERO; 4];
        let ring_switch = RingSwitch::new(&point, point.len()).unwrap();
        let prepared_claims = ring_switch
            .prepare_claims(&packed_witness, F128::default())
            .unwrap();
        let claims = *prepared_claims.claims();
        let challenge = core::array::from_fn(|index| FlockF128::new(index as u64 + 3, 0));

        let dense = prepared_claims.reduce(&challenge);
        let succinct = ring_switch.reduce_verifier(&claims, &challenge);

        assert_eq!(dense.packed_target, succinct.packed_target);
        assert_eq!(dense.packed_basis.len(), packed_witness.len());
    }
}
