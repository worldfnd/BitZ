//! Arbitrary `F128` inner products over the original committed bits.
//!
//! Each public weight is split into 128 binary coordinate covectors. The
//! GHASH dual map packs each covector against the natural witness packing.
//! The prover sends 128 packed coordinate values before batching them with
//! independent transcript weights. One generic Ligerito basis opening proves
//! the resulting packed inner product.

use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ligerito::{recursive_prover_with_basis, recursive_verifier_with_basis};
use flock_core::pcs::ring_switch::tensor_algebra_transpose;
use flock_core::pcs::{BatchOpeningProofLigerito, LOG_PACKING};
use transcript::{ProverState, VerifierState};

use crate::bridge::{as_flock_f128s, into_flock_f128s};
use crate::challenger::{ProverChallenger, VerifierChallenger};
use crate::utils::{
    bind_inner_product_statement, observe_opening_target, read_inner_product_coordinates,
    read_opening_proof, sample_inner_product_batching_weights, write_inner_product_coordinates,
    write_opening_proof,
};
use crate::verify::validate_ligerito_proof_shape;
use crate::{CommitError, Commitment, LigeritoProfile, Pcs, ProverData, StatementBinding};

const COORDINATE_COUNT: usize = 1 << LOG_PACKING;
const _: () = assert!(COORDINATE_COUNT == 128);

pub(crate) fn open(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    weights: &[F128],
    claimed_target: F128,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    validate_query(pcs, weights)?;
    if packed_witness.len() != pcs.packed_len() {
        return Err(CommitError::InvalidBitLength);
    }
    if !params_match(pcs, data) {
        return Err(CommitError::invalid_configuration(format!(
            "prover data parameters do not match the active PCS: expected {:?}, got {:?}",
            pcs.params(),
            data.commitment().params,
        )));
    }
    let log_n = pcs.params().m.checked_sub(LOG_PACKING).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "PCS variable count {} is smaller than the packing width {LOG_PACKING}",
            pcs.params().m,
        ))
    })?;
    let ligerito_config = pcs
        .params()
        .ligerito_prover_config()
        .map_err(CommitError::InvalidConfiguration)?;

    if statement_binding == StatementBinding::Bind {
        bind_inner_product_statement(
            pcs,
            &data.commitment().root,
            weights,
            claimed_target,
            transcript,
        );
    }

    let packed_witness = into_flock_f128s(packed_witness);
    let coordinate_values = coordinates(&packed_witness, weights);
    if reconstruct_claim(&coordinate_values) != Some(claimed_target) {
        return Err(CommitError::InvalidClaim);
    }

    write_inner_product_coordinates(transcript, &coordinate_values)?;
    let batching_weights = sample_inner_product_batching_weights(transcript);
    let target = dot(&coordinate_values, &batching_weights);
    let basis = combined_basis(weights, &batching_weights);

    observe_opening_target(transcript, log_n, target)?;
    let mut challenger = ProverChallenger::new_ligerito(transcript, target);
    let flock_data = data.flock_data();
    let ligerito = recursive_prover_with_basis(
        &ligerito_config,
        packed_witness,
        basis,
        target,
        &flock_data.codeword,
        &flock_data.merkle_tree,
        &mut challenger,
    );
    if challenger.failed() {
        return Err(CommitError::invalid_configuration(
            "missing Ligerito opening-target prefix",
        ));
    }

    write_opening_proof(
        &BatchOpeningProofLigerito {
            ring_switches: Vec::new(),
            ligerito,
        },
        transcript,
    )
}

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Commitment,
    weights: &[F128],
    claimed_target: F128,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    validate_query(pcs, weights)?;
    let log_n = pcs.params().m.checked_sub(LOG_PACKING).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "PCS variable count {} is smaller than the packing width {LOG_PACKING}",
            pcs.params().m,
        ))
    })?;
    let ligerito_config = pcs
        .params()
        .ligerito_verifier_config()
        .map_err(CommitError::InvalidConfiguration)?;

    if statement_binding == StatementBinding::Bind {
        bind_inner_product_statement(pcs, commitment.root(), weights, claimed_target, transcript);
    }

    let proof = read_opening_proof(transcript)?;
    if !proof.ring_switches.is_empty() {
        return Err(CommitError::VerificationFailed);
    }
    validate_ligerito_proof_shape(
        &proof.ligerito,
        &ligerito_config,
        pcs.final_log_n(),
        commitment.root(),
    )?;

    let coordinate_values = read_inner_product_coordinates(transcript)?;
    if reconstruct_claim(&coordinate_values) != Some(claimed_target) {
        return Err(CommitError::VerificationFailed);
    }
    let batching_weights = sample_inner_product_batching_weights(transcript);
    let target = dot(&coordinate_values, &batching_weights);
    let basis = combined_basis(weights, &batching_weights);

    observe_opening_target(transcript, log_n, target)?;
    let mut challenger = VerifierChallenger::new_ligerito(transcript, target);
    let valid = recursive_verifier_with_basis(
        &ligerito_config,
        &proof.ligerito,
        &basis,
        target,
        commitment.root(),
        &mut challenger,
    );
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }
    if !valid {
        return Err(CommitError::VerificationFailed);
    }
    Ok(())
}

fn validate_query(pcs: &Pcs, weights: &[F128]) -> Result<(), CommitError> {
    if pcs.params().profile != LigeritoProfile::Secure {
        return Err(CommitError::UnsupportedInnerProductProfile);
    }
    if weights.len() != pcs.bit_len() {
        return Err(CommitError::WeightLengthMismatch);
    }
    Ok(())
}

fn params_match(pcs: &Pcs, data: &ProverData) -> bool {
    let expected = pcs.params();
    let actual = &data.commitment().params;

    expected.m == actual.m
        && expected.log_inv_rate == actual.log_inv_rate
        && expected.log_batch_size == actual.log_batch_size
        && expected.profile == actual.profile
        && expected.merkle_hash == actual.merkle_hash
}

fn coordinates(packed_witness: &[FlockF128], weights: &[F128]) -> Vec<FlockF128> {
    assert_eq!(
        weights.len(),
        packed_witness.len() * COORDINATE_COUNT,
        "one weight is required for each packed witness bit",
    );
    let mut result = vec![FlockF128::ZERO; COORDINATE_COUNT];
    for (&packed, block) in packed_witness
        .iter()
        .zip(weights.chunks_exact(COORDINATE_COUNT))
    {
        if packed == FlockF128::ZERO {
            continue;
        }
        let bitplanes = tensor_algebra_transpose(as_flock_f128s(block));
        for (coordinate, bitplane) in result.iter_mut().zip(bitplanes) {
            *coordinate += packed * ghash_dual_pack(bitplane);
        }
    }
    result
}

fn combined_basis(weights: &[F128], batching_weights: &[FlockF128]) -> Vec<FlockF128> {
    assert_eq!(batching_weights.len(), COORDINATE_COUNT);
    assert_eq!(weights.len() % COORDINATE_COUNT, 0);
    let batching_table = build_batching_table(batching_weights);
    let dual_basis = (0..COORDINATE_COUNT)
        .map(|index| {
            let monomial = if index < 64 {
                FlockF128::new(1 << index, 0)
            } else {
                FlockF128::new(0, 1 << (index - 64))
            };
            ghash_dual_pack(monomial)
        })
        .collect::<Vec<_>>();

    weights
        .chunks_exact(COORDINATE_COUNT)
        .map(|block| {
            block.iter().zip(&dual_basis).fold(
                FlockF128::ZERO,
                |acc, (weight, dual_basis_element)| {
                    if *weight == F128::default() {
                        acc
                    } else {
                        acc + *dual_basis_element * fold_weight_bits(*weight, &batching_table)
                    }
                },
            )
        })
        .collect()
}

fn build_batching_table(batching_weights: &[FlockF128]) -> Vec<FlockF128> {
    const BYTE_VALUES: usize = 256;
    let mut table = vec![FlockF128::ZERO; 16 * BYTE_VALUES];
    for byte_index in 0..16 {
        let bit_base = 8 * byte_index;
        for byte_value in 0..BYTE_VALUES {
            let mut sum = FlockF128::ZERO;
            for bit in 0..8 {
                if (byte_value >> bit) & 1 == 1 {
                    sum += batching_weights[bit_base + bit];
                }
            }
            table[byte_index * BYTE_VALUES + byte_value] = sum;
        }
    }
    table
}

fn fold_weight_bits(weight: F128, batching_table: &[FlockF128]) -> FlockF128 {
    const BYTE_VALUES: usize = 256;
    let bytes = weight.to_bytes();
    bytes
        .iter()
        .enumerate()
        .fold(FlockF128::ZERO, |sum, (index, byte)| {
            sum + batching_table[index * BYTE_VALUES + usize::from(*byte)]
        })
}

fn reconstruct_claim(coordinate_values: &[FlockF128]) -> Option<F128> {
    if coordinate_values.len() != COORDINATE_COUNT {
        return None;
    }
    let mut words = [0u64; 2];
    for (index, value) in coordinate_values.iter().enumerate() {
        words[index >> 6] |= (value.lo & 1) << (index & 63);
    }
    Some(F128::new(words[0], words[1]))
}

fn dot(left: &[FlockF128], right: &[FlockF128]) -> FlockF128 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(FlockF128::ZERO, |acc, (left, right)| acc + *left * *right)
}

/// Applies the constant-coefficient dual map for the GHASH polynomial.
#[inline]
fn ghash_dual_pack(value: FlockF128) -> FlockF128 {
    let input = u128::from(value.lo) | (u128::from(value.hi) << 64);
    let mut output = (input.reverse_bits() << 1) | (input & 1);
    for index in 1..=6usize {
        output ^= ((input >> (7 - index)) & 1) << index;
    }
    output ^= ((input >> 1) & 1) << 1;
    FlockF128::new(output as u64, (output >> 64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghash_dual_pack_is_dual_to_natural_packing() {
        for left_index in 0..COORDINATE_COUNT {
            let mut left_words = [0u64; 2];
            left_words[left_index >> 6] = 1u64 << (left_index & 63);
            let left = FlockF128::new(left_words[0], left_words[1]);
            for right_index in 0..COORDINATE_COUNT {
                let mut right_words = [0u64; 2];
                right_words[right_index >> 6] = 1u64 << (right_index & 63);
                let right = FlockF128::new(right_words[0], right_words[1]);
                assert_eq!(
                    (left * ghash_dual_pack(right)).lo & 1,
                    u64::from(left_index == right_index),
                    "basis pair ({left_index}, {right_index})",
                );
            }
        }
    }

    #[test]
    fn arbitrary_weights_reconstruct_the_direct_bit_inner_product() {
        let packed = [
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
        for (block_index, block) in packed.iter().enumerate() {
            let words = [block.lo, block.hi];
            for bit in 0..COORDINATE_COUNT {
                if (words[bit >> 6] >> (bit & 63)) & 1 == 1 {
                    direct += weights[COORDINATE_COUNT * block_index + bit];
                }
            }
        }

        assert_eq!(
            reconstruct_claim(&coordinates(&packed, &weights)),
            Some(direct),
        );
    }

    #[test]
    fn combined_basis_matches_batched_coordinate_values() {
        let packed = [
            FlockF128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
            FlockF128::new(0x1357_9bdf_2468_ace0, 0x0f0f_f0f0_aaaa_5555),
        ];
        let weights = (0..256u64)
            .map(|index| F128::new(index.wrapping_mul(17), index.rotate_left(11)))
            .collect::<Vec<_>>();
        let batching_weights = (0..COORDINATE_COUNT as u64)
            .map(|index| FlockF128::new(index.wrapping_mul(31), index.rotate_left(7)))
            .collect::<Vec<_>>();

        let coordinate_values = coordinates(&packed, &weights);
        let basis = combined_basis(&weights, &batching_weights);

        assert_eq!(
            dot(&packed, &basis),
            dot(&coordinate_values, &batching_weights)
        );
    }

    #[test]
    fn arbitrary_inner_products_require_the_secure_profile() {
        let shape = common::Shape::new(7, 15).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, crate::HashKind::Blake3).unwrap();
        assert_eq!(
            validate_query(&pcs, &[]),
            Err(CommitError::UnsupportedInnerProductProfile),
        );
    }
}
