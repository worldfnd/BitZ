//! Ring-switch reduction for multilinear openings.
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
    fold_1b_rows_naive, fold_b128_elems, inner_product,
};
use flock_core::pcs::{LOG_PACKING, pack::PACKING_WIDTH as CLAIM_COUNT};
use flock_core::zerocheck::univariate_skip::build_eq;

use crate::ProveError;
use crate::bridge::{as_flock_f128, as_flock_f128s};
use crate::ligerito::ReducedClaim;
use crate::opening::QueryError;
use crate::transpose::tensor_algebra_transpose;
pub(super) type BatchingPoint = [FlockF128; LOG_PACKING];

/// A validated MLE point split into packed and unpacked coordinates.
pub(super) struct RingSwitch<'a> {
    point: &'a [FlockF128],
}

/// Ring-switch claims with the prover data needed for challenge reduction.
pub(super) struct PreparedClaims {
    pub(super) claims: [FlockF128; CLAIM_COUNT],
    suffix_tensor: Vec<FlockF128>,
}

/// One packed-field claim with a succinct basis evaluator.
pub(super) struct SuccinctReduction<'a> {
    pub(super) packed_target: FlockF128,
    suffix_point: &'a [FlockF128],
    batching_weights: Vec<FlockF128>,
}

impl<'a> RingSwitch<'a> {
    /// Validates the point and records its field-compatible view.
    pub(super) fn new(point: &'a [F128], variable_count: usize) -> Result<Self, QueryError> {
        if point.len() != variable_count {
            return Err(QueryError::PointLengthMismatch);
        }
        if variable_count < LOG_PACKING {
            return Err(QueryError::Internal);
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
    ) -> Result<PreparedClaims, ProveError> {
        let (prefix_tensor, suffix_tensor) = build_eq_split(self.point, LOG_PACKING);
        if suffix_tensor.len() != packed_witness.len() {
            return Err(ProveError::PackedWitnessLengthMismatch);
        }

        let claims = claims_from_prover(fold_1b_rows_naive(packed_witness, &suffix_tensor))?;
        if claim_check(&prefix_tensor, &claims) != as_flock_f128(claimed_target) {
            return Err(ProveError::InvalidClaim);
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
    pub(super) fn reduce_succinct(
        &self,
        claims: &[FlockF128; CLAIM_COUNT],
        batching_point: &BatchingPoint,
    ) -> SuccinctReduction<'a> {
        let batching_weights = build_eq(batching_point);
        let packed_target = batch_claims(claims, &batching_weights);
        SuccinctReduction {
            packed_target,
            suffix_point: &self.point[LOG_PACKING..],
            batching_weights,
        }
    }
}

impl PreparedClaims {
    /// Materializes the packed basis and batches the partial evaluations.
    pub(super) fn reduce_dense(self, batching_point: &BatchingPoint) -> ReducedClaim {
        let batching_weights = build_eq(batching_point);
        let packed_target = batch_claims(&self.claims, &batching_weights);
        let packed_basis = fold_b128_elems(&self.suffix_tensor, &batching_weights);
        debug_assert_eq!(packed_basis.len(), self.suffix_tensor.len());

        ReducedClaim {
            packed_basis,
            packed_target,
        }
    }
}

impl SuccinctReduction<'_> {
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
                    &self.batching_weights,
                )
            })
            .collect()
    }
}

fn batch_claims(claims: &[FlockF128; CLAIM_COUNT], batching_weights: &[FlockF128]) -> FlockF128 {
    debug_assert_eq!(batching_weights.len(), CLAIM_COUNT);
    let transposed_claims = tensor_algebra_transpose(claims);
    inner_product(&transposed_claims, batching_weights)
}

/// Converts the prover result into the fixed protocol shape.
fn claims_from_prover(values: Vec<FlockF128>) -> Result<[FlockF128; CLAIM_COUNT], ProveError> {
    let values: [FlockF128; CLAIM_COUNT] = values
        .try_into()
        .map_err(|_: Vec<FlockF128>| ProveError::Internal)?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_succinct_reductions_match_the_witness_and_residual_basis() {
        const SUFFIX_DIMENSION: usize = 4;
        let point = (0..LOG_PACKING + SUFFIX_DIMENSION)
            .map(|index| F128::new(index as u64 + 2, index as u64 * 13 + 7))
            .collect::<Vec<_>>();
        let set_bits = [0, 1, 63, 64, 127, 128, 191, 255, 256, 513, 1027, 2047];
        let mut packed_witness = vec![FlockF128::ZERO; 1 << SUFFIX_DIMENSION];
        for index in set_bits {
            if index % 128 < 64 {
                packed_witness[index / 128].lo |= 1 << (index % 128);
            } else {
                packed_witness[index / 128].hi |= 1 << (index % 128 - 64);
            }
        }
        let target = set_bits
            .into_iter()
            .map(|index| {
                point.iter().copied().enumerate().fold(
                    F128::from(1u64),
                    |product, (coordinate, value)| {
                        product
                            * if (index >> coordinate) & 1 == 1 {
                                value
                            } else {
                                F128::from(1u64) + value
                            }
                    },
                )
            })
            .sum::<F128>();
        assert_ne!(target, F128::default());
        let ring_switch = RingSwitch::new(&point, point.len()).unwrap();
        let prepared_claims = ring_switch.prepare_claims(&packed_witness, target).unwrap();
        let claims = prepared_claims.claims;
        let batching_point =
            core::array::from_fn(|index| FlockF128::new(index as u64 + 3, index as u64 * 17 + 5));

        let dense_reduction = prepared_claims.reduce_dense(&batching_point);
        let succinct_reduction = ring_switch.reduce_succinct(&claims, &batching_point);
        let packed_target = dense_reduction
            .packed_basis
            .iter()
            .zip(&packed_witness)
            .fold(FlockF128::ZERO, |sum, (&basis, &witness)| {
                sum + basis * witness
            });

        assert_ne!(packed_target, FlockF128::ZERO);
        assert_eq!(dense_reduction.packed_target, packed_target);
        assert_eq!(
            dense_reduction.packed_target,
            succinct_reduction.packed_target
        );
        assert_eq!(dense_reduction.packed_basis.len(), packed_witness.len());

        let residual_points: [[FlockF128; SUFFIX_DIMENSION]; 2] = [
            core::array::from_fn(|index| FlockF128::new(index as u64 * 11 + 3, index as u64 + 5)),
            [
                FlockF128::ZERO,
                FlockF128::ONE,
                FlockF128::ONE,
                FlockF128::ZERO,
            ],
        ];
        for residual_point in residual_points {
            let mut expected_basis = dense_reduction.packed_basis.clone();
            for prefix_len in 0..=SUFFIX_DIMENSION {
                assert_eq!(
                    succinct_reduction.evaluate_basis(
                        &residual_point[..prefix_len],
                        SUFFIX_DIMENSION - prefix_len,
                    ),
                    expected_basis,
                );
                if prefix_len < SUFFIX_DIMENSION {
                    let challenge = residual_point[prefix_len];
                    expected_basis = expected_basis
                        .chunks_exact(2)
                        .map(|pair| pair[0] + challenge * (pair[0] + pair[1]))
                        .collect();
                }
            }
        }
    }
}
