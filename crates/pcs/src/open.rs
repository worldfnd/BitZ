//! Batched multilinear openings over one Flock commitment.
//!
//! Prover steps:
//! 1. Validate the packed witness, prover data, and all query points.
//! 2. Bind the commitment root, trusted parameters, queries, and targets to the transcript.
//! 3. Compute and check the 128 partial evaluations for every query.
//! 4. Absorb all partial evaluations and sample one shared ring-switch challenge.
//! 5. Compute each packed target and sample one batching challenge per query.
//! 6. Build and combine the packed Ligerito bases with the batching challenges.
//! 7. Call `recursive_prover_with_basis` with the retained codeword and Merkle tree.
//! 8. Write all ring-switch messages and one Ligerito proof to the transcript.
//!
//! Ring-switch equations, with `r_lo = r[0..7]` and `r_hi = r[7..m]`:
//! `s_v = q̂(r_hi, v)` and `target = Σ_v eq(r_lo, v) · s_v`.
//! Write `eq(r_hi, y) = Σ_u A(y, u) · basis[u]` and transpose `(s_v)` into `(s_u)`.
//! For sampled `r_dprime`, set `B(y) = Σ_u eq(r_dprime, u) · A(y, u)`.
//! The final packed claim is `Σ_y B(y) · q_pkd(y) = beta0`.

use crate::bridge::{as_flock_f128s, into_flock_f128s};
use crate::challenger::ProverChallenger;
use crate::protocol::{
    bind_ring_switch_message, bind_statement, observe_opening_target, sample_batching_scalars,
    sample_shared_ring_switch_point, validate_batch, write_opening_proof,
};
use crate::{CommitError, Pcs, ProverData, ScopedOpeningQuery, StatementBinding};
use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ligerito::recursive_prover_with_basis;
use flock_core::pcs::ring_switch::{
    build_eq_split, claim_check, fold_1b_rows_split, fold_1b_rows_split_2way,
    fold_b128_elems_split, inner_product, split_n_lo, tensor_algebra_transpose,
};
use flock_core::pcs::{BatchOpeningProofLigerito, RingSwitchProof};
use flock_core::{
    pcs::LOG_PACKING,
    zerocheck::{PaddingSpec, univariate_skip::build_eq},
};
use transcript::ProverState;

pub(crate) fn open_batch(
    pcs: &Pcs,
    data: ProverData,
    packed_witness: Vec<F128>,
    queries: &[ScopedOpeningQuery<'_>],
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    // 1. Input Validation
    let expected_m = pcs.params().m;
    validate_batch(queries, expected_m)?;
    let m_p = expected_m
        .checked_sub(LOG_PACKING)
        .ok_or_else(|| CommitError::invalid_configuration("m is below LOG_PACKING"))?;
    if packed_witness.len() != pcs.packed_len() {
        return Err(CommitError::InvalidBitLength);
    }
    if !params_match(pcs, &data) {
        return Err(CommitError::invalid_configuration(
            "prover data parameters mismatch",
        ));
    }
    let ligerito_config = pcs
        .params()
        .ligerito_prover_config()
        .map_err(CommitError::InvalidConfiguration)?;

    // 2. Bind Statement
    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, &data.commitment().root, queries, transcript);
    }
    let packed_witness = into_flock_f128s(packed_witness);
    let flock_data = data.into_flock_data();

    // 3. Compute and Check Partial Evaluations
    let padding = PaddingSpec::dense(expected_m);
    // Retain balanced equality factors for both folding phases.
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

    // 4. Record Ring-Switch Messages and Sample the Shared Challenge
    // Normative ring-switch, batching, and opening schedule:
    // https://github.com/worldfnd/f2z-benchmark/blob/5014c717e88ab5e54e70e7a1099caaca5c41a926/docs/f2z-pcs-spec/part3-interaction.tex#L454-L534
    for (scoped_query, s_hat_v) in queries.iter().zip(&s_hat_vs) {
        bind_ring_switch_message(transcript, scoped_query.scope, s_hat_v)?;
    }
    let r_dprime = sample_shared_ring_switch_point(transcript);
    let eq_r_dprime = build_eq(&r_dprime);
    debug_assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);

    // 5. Compute and Batch the Ligerito Targets
    let betas = s_hat_vs.iter().map(|s_hat_v| {
        let s_hat_u = tensor_algebra_transpose(s_hat_v);
        inner_product(&s_hat_u, &eq_r_dprime)
    });
    let etas = sample_batching_scalars(transcript, queries.iter().map(|query| query.scope));
    let beta0 = betas
        .zip(&etas)
        .fold(FlockF128::ZERO, |sum, (beta, eta)| sum + beta * *eta);

    // 6. Build and Combine the Ligerito Bases
    let mut factors_and_etas = r_hi_eq_factors.iter().zip(&etas);
    let ((first_eq_lo, first_eq_hi), &first_eta) = factors_and_etas
        .next()
        .expect("validated batch is nonempty");
    let mut b_initial = fold_scaled_basis(first_eq_lo, first_eq_hi, &eq_r_dprime, first_eta);
    for ((eq_lo, eq_hi), &eta) in factors_and_etas {
        let basis = fold_scaled_basis(eq_lo, eq_hi, &eq_r_dprime, eta);
        debug_assert_eq!(basis.len(), b_initial.len());
        for (combined, &value) in b_initial.iter_mut().zip(&basis) {
            *combined += value;
        }
        flock_core::scratch::give_f128(basis);
    }
    // Release the equality factors before Ligerito allocates its working buffers.
    drop(r_hi_eq_factors);

    // 7. Prove the Ligerito Claim
    // Prove Σ_y b_initial[y] · packed_witness[y] = beta0.
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

    // 8. Write the Bounded Opening Proof
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

/// Folds split equality tensors against the witness, with one scan for each claim pair.
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

/// Builds one eta-scaled Ligerito basis without materializing a dense equality tensor.
fn fold_scaled_basis(
    eq_lo: &[FlockF128],
    eq_hi: &[FlockF128],
    eq_r_dprime: &[FlockF128],
    eta: FlockF128,
) -> Vec<FlockF128> {
    // The fold is F128-linear in eq_r_dprime, but not in the equality tensor.
    // Therefore, scale eq_r_dprime instead of either equality factor.
    let scaled_eq_r_dprime = eq_r_dprime
        .iter()
        .map(|&value| eta * value)
        .collect::<Vec<_>>();
    fold_b128_elems_split(eq_lo, eq_hi, &scaled_eq_r_dprime)
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
