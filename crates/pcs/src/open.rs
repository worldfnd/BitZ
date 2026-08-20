//! Proves batched multilinear openings against one Flock commitment.
//!
//! # Protocol
//!
//! This module implements the direct MLE claim transformations from the F2Z PCS specification.
//! See Section 2.7.1 on page 4 and Section 3.1, Groups 8 through 10, on pages 7 and 8.
//! Equation (34) defines each ring-switch basis and target.
//! Section 4, steps 36 through 50, gives the matching verifier order on page 9.
//!
//! Prover flow:
//! 1. Validate the packed witness, prover data, and query batch.
//! 2. Conditionally bind the commitment, parameters, queries, and targets.
//! 3. Compute and check 128 partial evaluations for each query.
//! 4. Bind every ring-switch vector, then sample one shared `r_dprime`.
//! 5. Derive each ring-switch target, then sample one `eta` per query.
//! 6. Combine the targets and form the `eta`-weighted Ligerito basis.
//! 7. Prove the combined claim with the retained codeword and Merkle tree.
//! 8. Serialize the complete Flock proof into one bounded compatibility hint.
//!
//! Steps 3 through 7 follow the claim flow in Groups 8 through 10.
//! Statement binding and proof transport belong to the local adapter.
//! The whole-proof hint differs from the specification's normative channel partition.
//!
//! For claim `ell`, split `r_ell` into `r_lo = r_ell[0..7]` and `r_hi = r_ell[7..m]`.
//! Then `s_ell[v] = q̂(r_hi, v)`.
//! The source gate requires `target_ell = Σ_v eq(v, r_lo) · s_ell[v]`.
//! Transposing the basis coordinates of `s_ell` gives `s_bar_ell`.
//! For shared `r_dprime`, define `Phi(z) = Σ_u eq(u, r_dprime) · [z]_u`.
//! Then `beta_ell = Σ_u eq(u, r_dprime) · s_bar_ell[u]`.
//! Also, `B_ell(y) = Phi(eq(r_hi, y))`.
//! Batching sets `B(y) = Σ_ell eta_ell · B_ell(y)` and `beta0 = Σ_ell eta_ell · beta_ell`.
//! Ligerito proves `Σ_y B(y) · q_pkd(y) = beta0`.
//! The code names the specification's batched target `beta` as `beta0`.

use crate::bridge::{as_flock_f128s, into_flock_f128s};
use crate::challenger::ProverChallenger;
use crate::utils::{
    bind_ring_switch_message, bind_statement, observe_opening_target, sample_batching_scalars,
    sample_shared_ring_switch_point, validate_batch, write_opening_proof,
};
use crate::{CommitError, Pcs, ProverData, ScopedOpeningQuery, StatementBinding};
use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ligerito::recursive_prover_with_basis;
use flock_core::pcs::ring_switch::{
    build_eq_split, claim_check, fold_1b_rows_split, fold_1b_rows_split_2way, inner_product,
    split_n_lo, tensor_algebra_transpose,
};
use flock_core::pcs::{BatchOpeningProofLigerito, RingSwitchProof};
use flock_core::{
    pcs::LOG_PACKING,
    zerocheck::{PaddingSpec, univariate_skip::build_eq},
};
use rayon::prelude::*;
use transcript::ProverState;

const FOLD_N_BYTES: usize = 16;
const FOLD_TABLE_SIZE: usize = 256;
const FOLD_TABLE_LEN: usize = FOLD_N_BYTES * FOLD_TABLE_SIZE;
type FoldTable = [FlockF128; FOLD_TABLE_LEN];

pub(crate) fn open_batch(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    queries: &[ScopedOpeningQuery<'_>],
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    // Step 1: Validate the batch, witness, prover data, and Ligerito configuration.
    let expected_m = pcs.params().m;
    validate_batch(queries, expected_m)?;
    let m_p = expected_m
        .checked_sub(LOG_PACKING)
        .ok_or_else(|| CommitError::invalid_configuration("m is below LOG_PACKING"))?;
    if packed_witness.len() != pcs.packed_len() {
        return Err(CommitError::InvalidBitLength);
    }
    if !params_match(pcs, data) {
        return Err(CommitError::invalid_configuration(
            "prover data parameters mismatch",
        ));
    }
    let ligerito_config = pcs
        .params()
        .ligerito_prover_config()
        .map_err(CommitError::InvalidConfiguration)?;

    // Step 2: Bind the statement unless the caller already bound it.
    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, &data.commitment().root, queries, transcript);
    }
    let packed_witness = into_flock_f128s(packed_witness);
    let flock_data = data.flock_data();

    // Step 3: Build split suffix factors and compute each set of 128 ring-switch values.
    // Keep balanced factors for the partial-evaluation fold and the basis fold.
    // At m=35, each factor pair uses 512 KiB. A dense table uses 4 GiB.
    let r_hi_eq_factors = queries
        .iter()
        .map(|scoped_query| {
            let r_hi = &scoped_query.query.point[LOG_PACKING..];
            build_eq_split(as_flock_f128s(r_hi), split_n_lo(r_hi.len()))
        })
        .collect::<Vec<_>>();

    #[cfg(debug_assertions)]
    for (eq_lo, eq_hi) in &r_hi_eq_factors {
        assert_eq!(eq_lo.len() * eq_hi.len(), packed_witness.len());
    }

    // Every committed bit is useful, so use Flock's dense padding descriptor.
    let padding = PaddingSpec::dense(expected_m);
    let s_hat_vs = fold_partial_evaluations(&packed_witness, &r_hi_eq_factors, &padding);

    for (scoped_query, s_hat_v) in queries.iter().zip(&s_hat_vs) {
        let query = scoped_query.query;
        let r_lo = &query.point[..LOG_PACKING];
        let eq_lo = build_eq(as_flock_f128s(r_lo));
        debug_assert_eq!(eq_lo.len(), 1 << r_lo.len());
        // s_hat_v[v] = Σ_y eq(r_hi, y) · q(y, v) = q̂(r_hi, v).
        debug_assert_eq!(s_hat_v.len(), 1 << LOG_PACKING);

        // query.target = Σ_v eq(r_lo, v) · s_hat_v[v].
        let evaluation = claim_check(&eq_lo, s_hat_v);
        let target = as_flock_f128s(core::slice::from_ref(&query.target))[0];
        if evaluation != target {
            return Err(CommitError::VerificationFailed);
        }
    }

    // Step 4: Bind all Group 8 vectors before one shared ring-switch challenge.
    for (scoped_query, s_hat_v) in queries.iter().zip(&s_hat_vs) {
        bind_ring_switch_message(transcript, scoped_query.scope, s_hat_v)?;
    }
    let r_dprime = sample_shared_ring_switch_point(transcript);
    let eq_r_dprime = build_eq(&r_dprime);
    debug_assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);

    // Step 5: Compute each beta_ell and combine it with a fresh eta_ell.
    let betas = s_hat_vs.iter().map(|s_hat_v| {
        let s_hat_u = tensor_algebra_transpose(s_hat_v);
        inner_product(&s_hat_u, &eq_r_dprime)
    });
    let etas = sample_batching_scalars(transcript, queries.iter().map(|query| query.scope));
    let beta0 = betas
        .zip(&etas)
        .fold(FlockF128::ZERO, |sum, (beta, eta)| sum + beta * *eta);

    // Step 6: Form B = Σ_ell eta_ell · B_ell without individual dense bases.
    let b_initial = fold_combined_basis(&r_hi_eq_factors, &eq_r_dprime, &etas);
    // Release the equality factors before Ligerito allocates its working buffers.
    drop(r_hi_eq_factors);

    // Step 7: Prove the Group 10 claim (root, B, beta) with Flock Ligerito.
    // The local wire profile binds m_p and beta0 before Flock observes the target.
    // Here b_initial is specification B, and beta0 is specification beta.
    observe_opening_target(transcript, m_p, beta0)?;
    let mut challenger = ProverChallenger::new_ligerito(transcript, beta0);
    let ligerito_proof = recursive_prover_with_basis(
        &ligerito_config,
        packed_witness,
        b_initial,
        beta0,
        &flock_data.codeword,
        &flock_data.merkle_tree,
        &mut challenger,
    );
    if challenger.failed() {
        return Err(CommitError::invalid_configuration(
            "missing Ligerito opening-target prefix",
        ));
    }

    // Step 8: Serialize the complete Flock proof into one bounded hint.
    // This compatibility transport differs from the normative NARG/hint partition.
    let opening_proof = BatchOpeningProofLigerito {
        ring_switches: s_hat_vs
            .into_iter()
            .map(|s_hat_v| RingSwitchProof { s_hat_v })
            .collect(),
        ligerito: ligerito_proof,
    };
    write_opening_proof(&opening_proof, transcript)?;

    Ok(())
}

/// Folds split equality tensors against the witness.
///
/// It uses one Flock scan for each claim pair and one scan for a final unpaired claim.
/// See Flock's [split row folds] in the pinned dependency.
///
/// [split row folds]: https://github.com/succinctlabs/flock/blob/879072249e52b8b9054bf0c6a034cec20f8f6fc7/crates/flock-core/src/pcs/ring_switch.rs#L930-L1225
fn fold_partial_evaluations(
    packed_witness: &[FlockF128],
    eq_factors: &[(Vec<FlockF128>, Vec<FlockF128>)],
    padding: &PaddingSpec,
) -> Vec<Vec<FlockF128>> {
    let mut results = Vec::with_capacity(eq_factors.len());
    let mut pairs = eq_factors.chunks_exact(2);
    for pair in &mut pairs {
        let [first, second] = pair else {
            unreachable!("chunks_exact returned a non-pair")
        };
        let (first_result, second_result) = fold_1b_rows_split_2way(
            packed_witness,
            &first.0,
            &first.1,
            &second.0,
            &second.1,
            padding,
        );
        results.push(first_result);
        results.push(second_result);
    }
    if let [last] = pairs.remainder() {
        results.push(fold_1b_rows_split(
            packed_witness,
            &last.0,
            &last.1,
            padding,
        ));
    }
    results
}

/// Builds the combined basis without materializing any individual `B_ell`.
///
/// This is the deferred-dense fold from Flock, specialized for F2Z's shared challenge.
/// Only the combined basis is dense. Each claim adds one 4,096-element lookup table.
/// Each output block accumulates every `eta`-scaled claim directly into the combined basis.
///
/// Adapted from Flock's [byte-table fold] and [fused combine].
/// Copyright 2025 The Binius Developers
/// Copyright 2025 Irreducible, Inc.
/// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
/// SPDX-License-Identifier: Apache-2.0 OR MIT
///
/// [byte-table fold]: https://github.com/succinctlabs/flock/blob/879072249e52b8b9054bf0c6a034cec20f8f6fc7/crates/flock-core/src/pcs/ring_switch.rs#L1564-L1675
/// [fused combine]: https://github.com/succinctlabs/flock/blob/879072249e52b8b9054bf0c6a034cec20f8f6fc7/crates/flock-core/src/pcs.rs#L264-L333
pub(super) fn fold_combined_basis(
    eq_factors: &[(Vec<FlockF128>, Vec<FlockF128>)],
    eq_r_dprime: &[FlockF128],
    etas: &[FlockF128],
) -> Vec<FlockF128> {
    assert!(!eq_factors.is_empty());
    assert_eq!(eq_factors.len(), etas.len());
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);

    let block_len = eq_factors[0].0.len();
    let block_count = eq_factors[0].1.len();
    assert!(
        eq_factors
            .iter()
            .all(|(eq_lo, eq_hi)| eq_lo.len() == block_len && eq_hi.len() == block_count)
    );

    let tables = etas
        .iter()
        .map(|&eta| build_scaled_fold_table(eq_r_dprime, eta))
        .collect::<Vec<_>>();
    let mut combined = flock_core::scratch::take_f128(block_len * block_count);
    combined
        .par_chunks_mut(block_len)
        .enumerate()
        .for_each(|(hi, output)| {
            let (first_eq_lo, first_eq_hi) = &eq_factors[0];
            let first_hi_factor = first_eq_hi[hi];
            for (slot, &lo_factor) in output.iter_mut().zip(first_eq_lo) {
                *slot = fold_slot(lo_factor * first_hi_factor, &tables[0]);
            }
            for ((eq_lo, eq_hi), table) in eq_factors[1..].iter().zip(&tables[1..]) {
                let hi_factor = eq_hi[hi];
                for (slot, &lo_factor) in output.iter_mut().zip(eq_lo) {
                    *slot += fold_slot(lo_factor * hi_factor, table);
                }
            }
        });
    combined
}

fn build_scaled_fold_table(eq_r_dprime: &[FlockF128], eta: FlockF128) -> Box<FoldTable> {
    let mut table = Box::new([FlockF128::ZERO; FOLD_TABLE_LEN]);
    for byte_index in 0..FOLD_N_BYTES {
        let scaled: [FlockF128; 8] =
            core::array::from_fn(|bit| eta * eq_r_dprime[byte_index * 8 + bit]);
        let offset = byte_index * FOLD_TABLE_SIZE;
        for value in 1usize..FOLD_TABLE_SIZE {
            let bit = value.trailing_zeros() as usize;
            table[offset + value] = table[offset + (value & (value - 1))] + scaled[bit];
        }
    }
    table
}

#[inline(always)]
fn fold_slot(value: FlockF128, table: &FoldTable) -> FlockF128 {
    let lo = value.lo.to_le_bytes();
    let hi = value.hi.to_le_bytes();
    let mut sums = [FlockF128::ZERO; 4];
    for byte_index in 0..8 {
        sums[byte_index & 3] += table[byte_index * FOLD_TABLE_SIZE + lo[byte_index] as usize];
        sums[byte_index & 3] += table[(byte_index + 8) * FOLD_TABLE_SIZE + hi[byte_index] as usize];
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

/// Checks that retained prover data uses the exact PCS parameters.
fn params_match(pcs: &Pcs, data: &ProverData) -> bool {
    let expected = pcs.params();
    let actual = &data.commitment().params;

    expected.m == actual.m
        && expected.log_inv_rate == actual.log_inv_rate
        && expected.log_batch_size == actual.log_batch_size
        && expected.profile == actual.profile
        && expected.merkle_hash == actual.merkle_hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::pcs::ring_switch::fold_b128_elems_split;

    #[test]
    fn fused_basis_matches_flock_for_every_byte_value() {
        let weights = (0..128)
            .map(|bit| {
                if bit < 64 {
                    FlockF128::new(1 << bit, 0)
                } else {
                    FlockF128::new(0, 1 << (bit - 64))
                }
            })
            .collect::<Vec<_>>();
        let values = (0..FOLD_N_BYTES)
            .flat_map(|byte_index| {
                (0..FOLD_TABLE_SIZE).map(move |value| {
                    let word = (value as u64) << (8 * (byte_index & 7));
                    if byte_index < 8 {
                        FlockF128::new(word, 0)
                    } else {
                        FlockF128::new(0, word)
                    }
                })
            })
            .collect::<Vec<_>>();
        let factors = vec![(values.clone(), vec![FlockF128::ONE])];
        let etas = vec![FlockF128::ONE];

        let actual = fold_combined_basis(&factors, &weights, &etas);
        assert_eq!(actual, materialized_flock_basis(&factors, &weights, &etas));
        assert_eq!(actual, values);
    }

    #[test]
    fn fused_basis_scales_and_combines_split_claims() {
        let zero = FlockF128::ZERO;
        let one = FlockF128::ONE;
        let x = FlockF128::generator();
        let x2 = x * x;
        let x3 = x2 * x;
        let factors = vec![
            (vec![one; 4], vec![one; 2]),
            (vec![one, zero, one, zero], vec![one, one]),
            (vec![zero, one, one, zero], vec![one, zero]),
            (vec![zero, zero, one, one], vec![zero, one]),
        ];
        let mut weights = vec![zero; 1 << LOG_PACKING];
        weights[0] = one;
        let etas = vec![zero, x, x2, x3];
        let expected = vec![x, x2, x + x2, zero, x, zero, x + x3, x3];

        let actual = fold_combined_basis(&factors, &weights, &etas);
        assert_eq!(actual, materialized_flock_basis(&factors, &weights, &etas));
        assert_eq!(actual, expected);
    }

    fn materialized_flock_basis(
        factors: &[(Vec<FlockF128>, Vec<FlockF128>)],
        weights: &[FlockF128],
        etas: &[FlockF128],
    ) -> Vec<FlockF128> {
        let mut combined = vec![FlockF128::ZERO; factors[0].0.len() * factors[0].1.len()];
        for ((eq_lo, eq_hi), &eta) in factors.iter().zip(etas) {
            let scaled = weights
                .iter()
                .map(|&weight| eta * weight)
                .collect::<Vec<_>>();
            let basis = fold_b128_elems_split(eq_lo, eq_hi, &scaled);
            for (combined, &value) in combined.iter_mut().zip(&basis) {
                *combined += value;
            }
            flock_core::scratch::give_f128(basis);
        }
        combined
    }
}
