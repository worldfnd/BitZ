//! Standard multilinear openings over the Flock commitment.
//!
//! Prover steps:
//! 1. Validate the packed witness, prover data, and query point.
//! 2. Bind the commitment root, trusted parameters, point, and target to the transcript.
//! 3. Split the point into seven low coordinates and `m - 7` high coordinates.
//! 4. Build the low and high equality tables with Flock's `build_eq_split`.
//! 5. Compute the 128 partial evaluations with `fold_1b_rows_naive`.
//! 6. Check the target against the low-coordinate equality table.
//! 7. Bind the tag-4001 ring-switch message.
//! 8. Sample seven tag-4101 ring-switch challenges and build their equality table.
//! 9. Transpose the partial evaluations and compute the packed target `beta0`.
//! 10. Build the packed Ligerito basis with `fold_b128_elems`.
//! 11. Bind tag 5001 and call `recursive_prover_with_basis`.
//! 12. Write a bounded opening proof to the transcript.
//!
//! Ring-switch equations, with `r_lo = r[0..7]` and `r_hi = r[7..m]`:
//! `s_v = q̂(r_hi, v)` and `target = Σ_v eq(r_lo, v) · s_v`.
//! Write `eq(r_hi, y) = Σ_u A(y, u) · basis[u]` and transpose `(s_v)` into `(s_u)`.
//! For sampled `r_dprime`, set `B(y) = Σ_u eq(r_dprime, u) · A(y, u)`.
//! The final packed claim is `Σ_y B(y) · q_pkd(y) = beta0`.

use crate::bridge::{as_flock_f128s, into_flock_f128s};
use crate::challenger::ProverChallenger;
use crate::utils::{
    bind_ring_switch_message, bind_statement, observe_opening_target, sample_ring_switch_point,
    write_opening_proof,
};
use crate::{CommitError, OpeningQuery, Pcs, ProverData, StatementBinding};
use field::F128;
use flock_core::pcs::ligerito::recursive_prover_with_basis;
use flock_core::pcs::ring_switch::{
    build_eq_split, claim_check, fold_1b_rows_naive, fold_b128_elems, inner_product,
    tensor_algebra_transpose,
};
use flock_core::pcs::{BatchOpeningProofLigerito, RingSwitchProof};
use flock_core::{pcs::LOG_PACKING, zerocheck::univariate_skip::build_eq};
use transcript::ProverState;

pub(crate) fn open(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    query: &OpeningQuery,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    // 1. Input Validation
    let expected_m = pcs.params().m;
    if query.point.len() != expected_m {
        return Err(CommitError::PointLengthMismatch);
    }
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
    let ligerito_config = pcs
        .params()
        .ligerito_prover_config()
        .map_err(CommitError::InvalidConfiguration)?;

    // 2. Bind Statement
    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, &data.commitment().root, query, transcript);
    }
    let packed_witness = into_flock_f128s(packed_witness);
    let flock_data = data.flock_data();

    // 3. Split Point
    let (r_lo, r_hi) = query.point.split_at(LOG_PACKING);

    // 4. Build eq Tables
    let (eq_lo, eq_hi) = build_eq_split(as_flock_f128s(&query.point), r_lo.len());
    debug_assert_eq!(eq_lo.len(), 1 << r_lo.len());
    debug_assert_eq!(eq_hi.len(), 1 << r_hi.len());
    debug_assert_eq!(eq_hi.len(), packed_witness.len());

    // 5. Compute Partial Evaluations
    // s_hat_v[v] = Σ_y eq(r_hi, y) · q(y, v) = q̂(r_hi, v).
    let s_hat_v = fold_1b_rows_naive(&packed_witness, &eq_hi);
    debug_assert_eq!(s_hat_v.len(), 1 << LOG_PACKING);

    // 6. Check Target
    // query.target = Σ_v eq(r_lo, v) · s_hat_v[v].
    let evaluation = claim_check(&eq_lo, &s_hat_v);
    let target = as_flock_f128s(core::slice::from_ref(&query.target))[0];
    if evaluation != target {
        return Err(CommitError::InvalidClaim);
    }

    // 7. Bind Ring-Switch Message
    bind_ring_switch_message(transcript, &s_hat_v)?;

    // 8. Sample Ring-Switch Challenges
    // eq_r_dprime[u] = eq(r_dprime, u).
    let r_dprime = sample_ring_switch_point(transcript);
    let eq_r_dprime = build_eq(&r_dprime);
    debug_assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);

    // 9. Compute the Ligerito Target
    // beta0 = Σ_u eq(r_dprime, u) · s_hat_u[u].
    let s_hat_u = tensor_algebra_transpose(&s_hat_v);
    let beta0 = inner_product(&s_hat_u, &eq_r_dprime);

    // 10. Build the Ligerito Basis
    // b_initial[y] = B(y) = Σ_u eq(r_dprime, u) · A(y, u).
    let b_initial = fold_b128_elems(&eq_hi, &eq_r_dprime);
    debug_assert_eq!(b_initial.len(), packed_witness.len());
    drop(eq_hi);
    drop(eq_lo);

    // 11. Prove the Ligerito Claim
    // Prove Σ_y b_initial[y] · packed_witness[y] = beta0.
    observe_opening_target(transcript, r_hi.len(), beta0)?;
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

    // 12. Write the Bounded Opening Proof
    let opening_proof = BatchOpeningProofLigerito {
        ring_switches: vec![RingSwitchProof { s_hat_v }],
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
