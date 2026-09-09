//! Ring-switch reduction for factored inner-product openings.
//!
//! Generate each public weight as `column_weights[column] * row_weights[row]`.
//! The blocks follow column order, then row order.
//!
//! Split each public weight block `W_y` into binary coordinate polynomials:
//! `T_k(W_y) = sum_v bit_k(W_y[v]) X^v`.
//! The constant-coefficient dual `A` satisfies
//! `coeff_0(x * A(t)) = dot_bits(x, t)`.
//!
//! The prover coordinate claims are
//! `z_k = sum_y packed_witness[y] * A(T_k(W_y))`.
//! Their constant coefficients reconstruct the original `F128` target.
//!
//! For transcript challenges `rho_k`, the reduced basis is
//! `basis[y] = sum_k rho_k * A(T_k(W_y))`.
//! Therefore `inner_product(packed_witness, basis) = sum_k rho_k * z_k`.

use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::pack::PACKING_WIDTH as CLAIM_COUNT;
use flock_core::pcs::ring_switch::{inner_product, tensor_algebra_transpose};

use crate::bridge::as_flock_f128s;
use crate::ligerito::ReducedClaim;
use crate::opening::QueryError;
use crate::{ProveError, VerifyError};

pub(super) type BatchingPoint = [FlockF128; CLAIM_COUNT];
const _: () = assert!(CLAIM_COUNT == 128);

/// A validated reduction from factored bit weights to one packed-field claim.
pub(super) struct RingSwitch<'a> {
    row_weights: &'a [F128],
    column_weights: &'a [F128],
}

impl<'a> RingSwitch<'a> {
    /// Validates the factor lengths and their alignment with packed elements.
    pub(super) fn new(
        row_weights: &'a [F128],
        column_weights: &'a [F128],
        bit_len: usize,
    ) -> Result<Self, QueryError> {
        if row_weights.is_empty()
            || column_weights.is_empty()
            || !row_weights.len().is_multiple_of(CLAIM_COUNT)
            || row_weights.len().checked_mul(column_weights.len()) != Some(bit_len)
        {
            return Err(QueryError::WeightLengthMismatch);
        }
        Ok(Self {
            row_weights,
            column_weights,
        })
    }

    /// Returns the number of original bit weights.
    pub(super) fn bit_len(&self) -> usize {
        self.row_weights.len() * self.column_weights.len()
    }

    /// Generates one packed element's weights at a time, in column order.
    pub(super) fn weight_blocks(&self) -> impl Iterator<Item = [F128; CLAIM_COUNT]> + '_ {
        let blocks_per_column = self.row_weights.len() / CLAIM_COUNT;
        (0..self.bit_len() / CLAIM_COUNT).map(move |block_index| {
            let column_weight = self.column_weights[block_index / blocks_per_column];
            let row_start = (block_index % blocks_per_column) * CLAIM_COUNT;
            core::array::from_fn(|index| column_weight * self.row_weights[row_start + index])
        })
    }

    /// Derives 128 coordinate claims and checks their target.
    pub(super) fn prepare_claims(
        &self,
        packed_witness: &[FlockF128],
        claimed_target: F128,
    ) -> Result<[FlockF128; CLAIM_COUNT], ProveError> {
        if packed_witness.len() != self.bit_len() / CLAIM_COUNT {
            return Err(ProveError::PackedWitnessLengthMismatch);
        }
        let claims = compute_claims(packed_witness, self.weight_blocks());
        if reconstructed_target(&claims) != claimed_target {
            return Err(ProveError::InvalidClaim);
        }
        Ok(claims)
    }

    /// Reduces the coordinate claims to one dense Ligerito claim.
    pub(super) fn reduce_dense(
        &self,
        claims: &[FlockF128; CLAIM_COUNT],
        batching_point: &BatchingPoint,
    ) -> ReducedClaim {
        let packed_target = inner_product(claims, batching_point);
        let packed_basis = build_batched_basis(self.weight_blocks(), batching_point);
        ReducedClaim {
            packed_basis,
            packed_target,
        }
    }

    /// Checks proof claims and reduces them to one dense Ligerito claim.
    pub(super) fn reduce_verified(
        &self,
        claims: &[FlockF128; CLAIM_COUNT],
        claimed_target: F128,
        batching_point: &BatchingPoint,
    ) -> Result<ReducedClaim, VerifyError> {
        if reconstructed_target(claims) != claimed_target {
            return Err(VerifyError::VerificationFailed);
        }
        Ok(self.reduce_dense(claims, batching_point))
    }
}

/// Reconstructs the claimed inner product from constant coefficients.
fn reconstructed_target(claims: &[FlockF128; CLAIM_COUNT]) -> F128 {
    let mut words = [0u64; 2];
    for (index, value) in claims.iter().enumerate() {
        words[index >> 6] |= (value.lo & 1) << (index & 63);
    }
    F128::new(words[0], words[1])
}

fn compute_claims(
    packed_witness: &[FlockF128],
    weight_blocks: impl Iterator<Item = [F128; CLAIM_COUNT]>,
) -> [FlockF128; CLAIM_COUNT] {
    let mut claims = [FlockF128::ZERO; CLAIM_COUNT];
    for (&packed, weight_block) in packed_witness.iter().zip(weight_blocks) {
        if packed == FlockF128::ZERO {
            continue;
        }
        let coordinate_polynomials = tensor_algebra_transpose(as_flock_f128s(&weight_block));
        for (claim, polynomial) in claims.iter_mut().zip(coordinate_polynomials) {
            *claim += packed * constant_coefficient_dual(polynomial);
        }
    }
    claims
}

fn build_batched_basis(
    weight_blocks: impl Iterator<Item = [F128; CLAIM_COUNT]>,
    batching_point: &BatchingPoint,
) -> Vec<FlockF128> {
    let batching_table = BatchingPointByteTable::new(batching_point);
    let dual_monomials: [FlockF128; CLAIM_COUNT] =
        core::array::from_fn(|index| constant_coefficient_dual(monomial(index)));

    weight_blocks
        .map(|weight_block| {
            weight_block.iter().zip(&dual_monomials).fold(
                FlockF128::ZERO,
                |basis_value, (weight, dual_monomial)| {
                    if *weight == F128::default() {
                        basis_value
                    } else {
                        basis_value
                            + *dual_monomial * batching_table.combine_weight_coordinates(*weight)
                    }
                },
            )
        })
        .collect()
}

/// A byte lookup table for `sum_k bit_k(weight) * challenge[k]`.
struct BatchingPointByteTable {
    sums: Vec<FlockF128>,
}

impl BatchingPointByteTable {
    const BYTE_COUNT: usize = 16;
    const BYTE_VALUES: usize = 256;

    fn new(batching_point: &BatchingPoint) -> Self {
        let mut sums = vec![FlockF128::ZERO; Self::BYTE_COUNT * Self::BYTE_VALUES];
        for byte_index in 0..Self::BYTE_COUNT {
            let bit_base = 8 * byte_index;
            for byte_value in 0..Self::BYTE_VALUES {
                let mut sum = FlockF128::ZERO;
                for bit in 0..8 {
                    if (byte_value >> bit) & 1 == 1 {
                        sum += batching_point[bit_base + bit];
                    }
                }
                sums[byte_index * Self::BYTE_VALUES + byte_value] = sum;
            }
        }
        Self { sums }
    }

    fn combine_weight_coordinates(&self, weight: F128) -> FlockF128 {
        weight.to_bytes().iter().enumerate().fold(
            FlockF128::ZERO,
            |sum, (byte_index, byte_value)| {
                sum + self.sums[byte_index * Self::BYTE_VALUES + usize::from(*byte_value)]
            },
        )
    }
}

fn monomial(index: usize) -> FlockF128 {
    debug_assert!(index < CLAIM_COUNT);
    if index < 64 {
        FlockF128::new(1 << index, 0)
    } else {
        FlockF128::new(0, 1 << (index - 64))
    }
}

/// Applies the constant-coefficient dual map for the GHASH polynomial.
///
/// The modulus is `X^128 + X^7 + X^2 + X + 1`. Its dual basis is:
///
/// - `A(1) = 1`;
/// - `A(X^i) = X^(128-i) + X^(7-i) + [i=1]X` for `1 <= i <= 6`;
/// - `A(X^i) = X^(128-i)` for `7 <= i < 128`.
///
/// The bit operations below apply this basis formula to all coefficients.
#[inline]
fn constant_coefficient_dual(value: FlockF128) -> FlockF128 {
    let coefficients = u128::from(value.lo) | (u128::from(value.hi) << 64);
    let mut dual = (coefficients.reverse_bits() << 1) | (coefficients & 1);
    for input_degree in 1..=6usize {
        dual ^= ((coefficients >> input_degree) & 1) << (7 - input_degree);
    }
    dual ^= ((coefficients >> 1) & 1) << 1;
    FlockF128::new(dual as u64, (dual >> 64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_coefficient_dual_is_dual_to_natural_packing() {
        for left_index in 0..CLAIM_COUNT {
            let left = monomial(left_index);
            for right_index in 0..CLAIM_COUNT {
                assert_eq!(
                    (left * constant_coefficient_dual(monomial(right_index))).lo & 1,
                    u64::from(left_index == right_index),
                    "basis pair ({left_index}, {right_index})",
                );
            }
        }
    }

    #[test]
    fn boundary_bits_reconstruct_their_weights() {
        let column_weights = [F128::from(1u64)];
        for bit_index in [0, 63, 64, 127] {
            let packed_witness = [monomial(bit_index)];
            let weights = (0..CLAIM_COUNT)
                .map(|index| {
                    F128::new(
                        (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                        (index as u64).rotate_left(29) ^ 0xa5a5_5a5a_f0f0_0f0f,
                    )
                })
                .collect::<Vec<_>>();
            let ring_switch = RingSwitch::new(&weights, &column_weights, CLAIM_COUNT).unwrap();
            ring_switch
                .prepare_claims(&packed_witness, weights[bit_index])
                .unwrap();
        }
    }

    #[test]
    fn arbitrary_weights_reconstruct_the_direct_bit_inner_product() {
        let packed_witness = [
            FlockF128::new(0x8000_0000_0000_0003, 0x8000_0000_0000_0001),
            FlockF128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
        ];
        let weights = (0..256u64)
            .map(|index| {
                F128::new(
                    index.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    index.rotate_left(29) ^ 0xa5a5_5a5a_f0f0_0f0f,
                )
            })
            .collect::<Vec<_>>();
        let mut direct = F128::default();
        for (block_index, block) in packed_witness.iter().enumerate() {
            let words = [block.lo, block.hi];
            for bit in 0..CLAIM_COUNT {
                if (words[bit >> 6] >> (bit & 63)) & 1 == 1 {
                    direct += weights[CLAIM_COUNT * block_index + bit];
                }
            }
        }
        let column_weights = [F128::from(1u64)];
        let ring_switch = RingSwitch::new(
            &weights,
            &column_weights,
            packed_witness.len() * CLAIM_COUNT,
        )
        .unwrap();
        ring_switch.prepare_claims(&packed_witness, direct).unwrap();
    }

    #[test]
    fn optimized_batched_basis_matches_the_reference() {
        let weights = (0..CLAIM_COUNT * CLAIM_COUNT)
            .map(|index| {
                let coordinate = (index / CLAIM_COUNT + index % CLAIM_COUNT) % CLAIM_COUNT;
                if coordinate < 64 {
                    F128::new(1 << coordinate, 0)
                } else {
                    F128::new(0, 1 << (coordinate - 64))
                }
            })
            .collect::<Vec<_>>();
        let batching_point = core::array::from_fn(monomial);

        assert_eq!(
            build_batched_basis(flat_weight_blocks(&weights), &batching_point),
            reference_batched_basis(&weights, &batching_point),
        );
    }

    #[test]
    fn batching_point_byte_table_covers_every_byte_value_and_position() {
        let batching_point = core::array::from_fn(monomial);
        let table = BatchingPointByteTable::new(&batching_point);

        for byte_index in 0..BatchingPointByteTable::BYTE_COUNT {
            for byte_value in 0..BatchingPointByteTable::BYTE_VALUES {
                let mut bytes = [0u8; 16];
                bytes[byte_index] = byte_value as u8;
                let weight = F128::from_bytes(bytes);

                assert_eq!(
                    table.combine_weight_coordinates(weight),
                    FlockF128::new(weight.lo, weight.hi),
                    "byte position {byte_index}, value {byte_value}",
                );
            }
        }
    }

    #[test]
    fn reduced_claim_matches_batched_claims() {
        let packed_witness = [
            FlockF128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
            FlockF128::new(0x1357_9bdf_2468_ace0, 0x0f0f_f0f0_aaaa_5555),
        ];
        let weights = (0..256u64)
            .map(|index| F128::new(index.wrapping_mul(17), index.rotate_left(11)))
            .collect::<Vec<_>>();
        let batching_point = core::array::from_fn(|index| {
            let index = index as u64;
            FlockF128::new(index.wrapping_mul(31), index.rotate_left(7))
        });
        let column_weights = [F128::from(1u64)];
        let ring_switch = RingSwitch::new(
            &weights,
            &column_weights,
            packed_witness.len() * CLAIM_COUNT,
        )
        .unwrap();
        let claims = compute_claims(&packed_witness, flat_weight_blocks(&weights));
        let checked_claims = ring_switch
            .prepare_claims(&packed_witness, reconstructed_target(&claims))
            .unwrap();
        assert_eq!(checked_claims, claims);
        let dense_reduction = ring_switch.reduce_dense(&claims, &batching_point);

        assert_eq!(
            inner_product(&packed_witness, &dense_reduction.packed_basis),
            dense_reduction.packed_target,
        );
    }

    #[test]
    fn factored_weights_match_materialized_reduction() {
        let row_weights = (0..256u64)
            .map(|index| {
                if index % 17 == 0 {
                    F128::default()
                } else {
                    F128::new(
                        index.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                        index.rotate_left(29) ^ 0xa5a5_5a5a_f0f0_0f0f,
                    )
                }
            })
            .collect::<Vec<_>>();
        let column_weights = [
            F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
            F128::default(),
            F128::new(0x1357_9bdf_2468_ace0, 0x0f0f_f0f0_aaaa_5555),
            F128::from(1u64),
        ];
        let packed_witness = [
            FlockF128::new(0x8000_0000_0000_0001, 0x8000_0000_0000_0001),
            FlockF128::ZERO,
            FlockF128::new(u64::MAX, u64::MAX),
            FlockF128::new(1, 0),
            FlockF128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
            FlockF128::new(0x8000_0000_0000_0001, 0x8000_0000_0000_0001),
            FlockF128::ZERO,
            FlockF128::new(0x1357_9bdf_2468_ace0, 0x0f0f_f0f0_aaaa_5555),
        ];
        let weights = column_weights
            .iter()
            .flat_map(|&column| row_weights.iter().map(move |&row| column * row))
            .collect::<Vec<_>>();
        let ring_switch = RingSwitch::new(&row_weights, &column_weights, weights.len()).unwrap();
        assert_eq!(ring_switch.bit_len(), weights.len());
        assert_eq!(
            ring_switch.weight_blocks().flatten().collect::<Vec<_>>(),
            weights
        );

        let target = direct_bit_inner_product(&packed_witness, &weights);
        let claims = ring_switch.prepare_claims(&packed_witness, target).unwrap();
        assert_eq!(claims, reference_claims(&packed_witness, &weights));

        let batching_point = core::array::from_fn(|index| {
            FlockF128::new(
                (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                (index as u64).rotate_left(11),
            )
        });
        let reduced = ring_switch.reduce_dense(&claims, &batching_point);
        assert_eq!(
            reduced.packed_basis,
            reference_batched_basis(&weights, &batching_point),
        );
        assert_eq!(
            inner_product(&packed_witness, &reduced.packed_basis),
            reduced.packed_target,
        );
        let verified = ring_switch
            .reduce_verified(&claims, target, &batching_point)
            .unwrap();
        assert_eq!(verified.packed_basis, reduced.packed_basis);
        assert_eq!(verified.packed_target, reduced.packed_target);

        let wrong_target = target + F128::from(1u64);
        assert!(matches!(
            ring_switch.prepare_claims(&packed_witness, wrong_target),
            Err(ProveError::InvalidClaim),
        ));
        assert!(matches!(
            ring_switch.reduce_verified(&claims, wrong_target, &batching_point),
            Err(VerifyError::VerificationFailed),
        ));
    }

    #[test]
    fn rejects_invalid_factor_and_witness_lengths() {
        let row_weights = [F128::from(1u64); CLAIM_COUNT];
        let column_weights = [F128::from(1u64); 2];
        for (rows, columns, bit_len) in [
            (&[][..], &column_weights[..], 0),
            (&row_weights[..], &[][..], 0),
            (&[][..], &[][..], 0),
            (&row_weights[..127], &column_weights[..], 254),
            (&row_weights[..], &column_weights[..], 128),
            (&row_weights[..], &column_weights[..], usize::MAX),
        ] {
            assert!(matches!(
                RingSwitch::new(rows, columns, bit_len),
                Err(QueryError::WeightLengthMismatch),
            ));
        }

        let ring_switch = RingSwitch::new(&row_weights, &column_weights, 256).unwrap();
        for packed_witness in [&[][..], &[FlockF128::ZERO][..], &[FlockF128::ZERO; 3][..]] {
            assert!(matches!(
                ring_switch.prepare_claims(packed_witness, F128::default()),
                Err(ProveError::PackedWitnessLengthMismatch),
            ));
        }
    }

    fn flat_weight_blocks(weights: &[F128]) -> impl Iterator<Item = [F128; CLAIM_COUNT]> + '_ {
        weights
            .chunks_exact(CLAIM_COUNT)
            .map(|block| block.try_into().unwrap())
    }

    fn direct_bit_inner_product(packed_witness: &[FlockF128], weights: &[F128]) -> F128 {
        let mut target = F128::default();
        for (block_index, packed) in packed_witness.iter().enumerate() {
            let bits = u128::from(packed.lo) | (u128::from(packed.hi) << 64);
            for bit in 0..CLAIM_COUNT {
                if (bits >> bit) & 1 == 1 {
                    target += weights[block_index * CLAIM_COUNT + bit];
                }
            }
        }
        target
    }

    fn reference_claims(
        packed_witness: &[FlockF128],
        weights: &[F128],
    ) -> [FlockF128; CLAIM_COUNT] {
        let mut claims = [FlockF128::ZERO; CLAIM_COUNT];
        for (&packed, block) in packed_witness.iter().zip(weights.chunks_exact(CLAIM_COUNT)) {
            for (claim, polynomial) in claims
                .iter_mut()
                .zip(tensor_algebra_transpose(as_flock_f128s(block)))
            {
                *claim += packed * constant_coefficient_dual(polynomial);
            }
        }
        claims
    }

    fn reference_batched_basis(weights: &[F128], batching_point: &BatchingPoint) -> Vec<FlockF128> {
        weights
            .chunks_exact(CLAIM_COUNT)
            .map(|weight_block| {
                tensor_algebra_transpose(as_flock_f128s(weight_block))
                    .into_iter()
                    .zip(batching_point)
                    .fold(FlockF128::ZERO, |sum, (polynomial, coefficient)| {
                        sum + *coefficient * constant_coefficient_dual(polynomial)
                    })
            })
            .collect()
    }
}
