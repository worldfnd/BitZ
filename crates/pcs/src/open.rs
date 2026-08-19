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
    ETA_SQUEEZE_LABEL, RING_SWITCH_CLAIM_LABEL, RING_SWITCH_LABEL, bind_statement, validate_batch,
    write_opening_proof,
};
use crate::{CommitError, Pcs, ProverData, ScopedOpeningQuery, StatementBinding};
use field::F128;
use flock_core::challenger::Challenger;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ligerito::recursive_prover_with_basis;
use flock_core::pcs::ring_switch::{
    build_eq_split, claim_check, fold_1b_rows_naive, fold_b128_elems, inner_product,
    tensor_algebra_transpose,
};
use flock_core::pcs::{BatchOpeningProofLigerito, RingSwitchProof};
use flock_core::{pcs::LOG_PACKING, zerocheck::univariate_skip::build_eq};
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
    if packed_witness.len() != pcs.packed_len() {
        return Err(CommitError::InvalidBitLength);
    }
    if !params_match(pcs, &data) {
        return Err(CommitError::invalid_configuration(format!(
            "prover data parameters do not match the active PCS: expected {:?}, got {:?}",
            pcs.params(),
            data.commitment().params,
        )));
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
    let mut s_hat_vs = Vec::with_capacity(queries.len());
    for scoped_query in queries {
        let query = scoped_query.query;
        let (r_lo, r_hi) = query.point.split_at(LOG_PACKING);
        let (eq_lo, eq_hi) = build_eq_split(as_flock_f128s(&query.point), r_lo.len());
        debug_assert_eq!(eq_lo.len(), 1 << r_lo.len());
        debug_assert_eq!(eq_hi.len(), 1 << r_hi.len());
        debug_assert_eq!(eq_hi.len(), packed_witness.len());

        // s_hat_v[v] = Σ_y eq(r_hi, y) · q(y, v) = q̂(r_hi, v).
        let s_hat_v = fold_1b_rows_naive(&packed_witness, &eq_hi);
        debug_assert_eq!(s_hat_v.len(), 1 << LOG_PACKING);

        // query.target = Σ_v eq(r_lo, v) · s_hat_v[v].
        let evaluation = claim_check(&eq_lo, &s_hat_v);
        let target = as_flock_f128s(core::slice::from_ref(&query.target))[0];
        if evaluation != target {
            return Err(CommitError::VerificationFailed);
        }
        s_hat_vs.push(s_hat_v);
    }

    // 4. Record Ring-Switch Messages and Sample the Shared Challenge
    let mut challenger = ProverChallenger::new(transcript);
    challenger.observe_label(RING_SWITCH_LABEL);
    for (scoped_query, s_hat_v) in queries.iter().zip(&s_hat_vs) {
        challenger.observe_label(RING_SWITCH_CLAIM_LABEL);
        challenger.public_message(&scoped_query.scope);
        challenger.observe_f128_slice(s_hat_v);
    }
    let r_dprime = challenger.sample_f128_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);
    debug_assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);

    // 5. Compute and Batch the Ligerito Targets
    let betas = s_hat_vs.iter().map(|s_hat_v| {
        let s_hat_u = tensor_algebra_transpose(s_hat_v);
        inner_product(&s_hat_u, &eq_r_dprime)
    });
    challenger.observe_label(ETA_SQUEEZE_LABEL);
    let etas = challenger.sample_f128_vec(queries.len());
    let beta0 = betas
        .zip(&etas)
        .fold(FlockF128::ZERO, |sum, (beta, eta)| sum + beta * *eta);

    // 6. Build and Combine the Ligerito Bases
    let mut b_initial = vec![FlockF128::ZERO; packed_witness.len()];
    for (scoped_query, eta) in queries.iter().zip(&etas) {
        let query = scoped_query.query;
        let r_hi = &query.point[LOG_PACKING..];
        let eq_hi = build_eq(as_flock_f128s(r_hi));
        let basis = fold_b128_elems(&eq_hi, &eq_r_dprime);
        debug_assert_eq!(basis.len(), b_initial.len());
        for (combined, value) in b_initial.iter_mut().zip(basis) {
            *combined += *eta * value;
        }
    }

    // 7. Prove the Ligerito Claim
    // Prove Σ_y b_initial[y] · packed_witness[y] = beta0.
    let ligerito_proof = recursive_prover_with_basis(
        &ligerito_config,
        packed_witness,
        b_initial,
        beta0,
        &flock_data.codeword,
        &flock_data.merkle_tree,
        &mut challenger,
    );

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

fn params_match(pcs: &Pcs, data: &ProverData) -> bool {
    let expected = pcs.params();
    let actual = &data.commitment().params;

    expected.m == actual.m
        && expected.log_inv_rate == actual.log_inv_rate
        && expected.log_batch_size == actual.log_batch_size
        && expected.profile == actual.profile
        && expected.merkle_hash == actual.merkle_hash
}
