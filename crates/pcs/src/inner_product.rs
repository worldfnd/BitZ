//! Ring-switch reduction for explicit inner-product openings.
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

use crate::CommitError;
use crate::bridge::as_flock_f128s;
use crate::ligerito::ReducedClaim;

pub(super) type BatchingPoint = [FlockF128; CLAIM_COUNT];
const _: () = assert!(CLAIM_COUNT == 128);

/// A validated reduction from explicit bit weights to one packed-field claim.
pub(super) struct RingSwitch<'a> {
    weights: &'a [F128],
}

impl<'a> RingSwitch<'a> {
    /// Validates one weight for every bit in the packed witness.
    pub(super) fn new(weights: &'a [F128], bit_len: usize) -> Result<Self, CommitError> {
        if weights.len() != bit_len {
            return Err(CommitError::WeightLengthMismatch);
        }
        Ok(Self { weights })
    }

    /// Derives 128 coordinate claims and checks their target.
    pub(super) fn prepare_claims(
        &self,
        packed_witness: &[FlockF128],
        claimed_target: F128,
    ) -> Result<[FlockF128; CLAIM_COUNT], CommitError> {
        if packed_witness.len() != self.weights.len() / CLAIM_COUNT {
            return Err(CommitError::InvalidBitLength);
        }
        let claims = compute_claims(packed_witness, self.weights);
        if !self.target_matches(&claims, claimed_target) {
            return Err(CommitError::InvalidClaim);
        }
        Ok(claims)
    }

    /// Checks the original target against proof-provided coordinate claims.
    pub(super) fn target_matches(
        &self,
        claims: &[FlockF128; CLAIM_COUNT],
        claimed_target: F128,
    ) -> bool {
        reconstructed_target(claims) == claimed_target
    }

    /// Reduces the coordinate claims to one dense Ligerito claim.
    pub(super) fn reduce_dense(
        &self,
        claims: &[FlockF128; CLAIM_COUNT],
        batching_point: &BatchingPoint,
    ) -> ReducedClaim {
        let packed_target = inner_product(claims, batching_point);
        let packed_basis = build_batched_basis(self.weights, batching_point);
        ReducedClaim {
            packed_basis,
            packed_target,
        }
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

fn compute_claims(packed_witness: &[FlockF128], weights: &[F128]) -> [FlockF128; CLAIM_COUNT] {
    debug_assert_eq!(weights.len(), packed_witness.len() * CLAIM_COUNT);
    let mut claims = [FlockF128::ZERO; CLAIM_COUNT];
    for (&packed, weight_block) in packed_witness.iter().zip(weights.chunks_exact(CLAIM_COUNT)) {
        if packed == FlockF128::ZERO {
            continue;
        }
        let coordinate_polynomials = tensor_algebra_transpose(as_flock_f128s(weight_block));
        for (claim, polynomial) in claims.iter_mut().zip(coordinate_polynomials) {
            *claim += packed * constant_coefficient_dual(polynomial);
        }
    }
    claims
}

fn build_batched_basis(weights: &[F128], batching_point: &BatchingPoint) -> Vec<FlockF128> {
    debug_assert_eq!(weights.len() % CLAIM_COUNT, 0);
    let batching_table = BatchingPointByteTable::new(batching_point);
    let dual_monomials: [FlockF128; CLAIM_COUNT] =
        core::array::from_fn(|index| constant_coefficient_dual(monomial(index)));

    weights
        .chunks_exact(CLAIM_COUNT)
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
            let ring_switch = RingSwitch::new(&weights, CLAIM_COUNT).unwrap();
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
        let ring_switch = RingSwitch::new(&weights, packed_witness.len() * CLAIM_COUNT).unwrap();
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
            build_batched_basis(&weights, &batching_point),
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
        let ring_switch = RingSwitch::new(&weights, packed_witness.len() * CLAIM_COUNT).unwrap();
        let claims = compute_claims(&packed_witness, &weights);
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
