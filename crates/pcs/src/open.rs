//! Standard multilinear openings over the FLoCK commitment.
//!
//! Prover steps:
//! 1. Validate the prover data and require `query.point.len() == params.m`.
//! 2. Bind the commitment root, trusted parameters, point, and target to the transcript.
//! 3. Split the point into seven low coordinates and `m - 7` high coordinates.
//! 4. Build the high-coordinate equality table with FLoCK's `build_eq`.
//! 5. Compute the 128 partial evaluations with `fold_1b_rows_naive`.
//! 6. Check the target against the low-coordinate equality table.
//! 7. Absorb the ring-switch domain label and all partial evaluations.
//! 8. Sample seven ring-switch challenges and build their equality table.
//! 9. Transpose the partial evaluations and compute the packed target `beta0`.
//! 10. Build the packed Ligerito basis with `fold_b128_elems`.
//! 11. Call `recursive_prover_with_basis` with the retained codeword and Merkle tree.
//! 12. Write a bounded opening proof to the transcript.

use crate::bridge::as_flock_f128s;
use crate::challenger::ProverChallenger;
use crate::protocol::{RING_SWITCH_LABEL, bind_statement_prover, write_opening_proof};
use crate::{CommitError, OpeningQuery, Pcs, ProverData};
use flock_core::challenger::Challenger;
use flock_core::pcs::ligerito::recursive_prover_with_basis;
use flock_core::pcs::ring_switch::{
    claim_check, fold_1b_rows_naive, fold_b128_elems, inner_product, tensor_algebra_transpose,
};
use flock_core::pcs::{BatchOpeningProofLigerito, RingSwitchProof};
use flock_core::{pcs::LOG_PACKING, zerocheck::univariate_skip::build_eq};
use transcript::ProverState;

pub(crate) fn open(
    pcs: &Pcs,
    data: ProverData,
    query: &OpeningQuery,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    // 1. Input Validation
    let expected_m = pcs.params().m;
    if query.point.len() != expected_m {
        return Err(CommitError::PointLengthMismatch);
    }
    if data.bit_len() != pcs.bit_len() || !params_match(pcs, &data) {
        return Err(CommitError::InvalidConfiguration);
    }
    let ligerito_config = pcs
        .params()
        .ligerito_prover_config()
        .map_err(|_| CommitError::InvalidConfiguration)?;
    // 2. Bind Statement
    bind_statement_prover(pcs, &data.commitment().root, query, transcript);
    let (packed_witness, flock_data) = data.into_opening_parts();
    // 3. Split Point
    let (r_lo, r_hi) = query.point.split_at(LOG_PACKING);
    // 4. Build eq table
    let eq_hi = build_eq(as_flock_f128s(r_hi));
    debug_assert_eq!(eq_hi.len(), packed_witness.len());
    // 5. Compute Partial Evaluations
    let s_hat_v = fold_1b_rows_naive(&packed_witness, &eq_hi);
    debug_assert_eq!(s_hat_v.len(), 1 << LOG_PACKING);
    // 6. Check Target
    let eq_lo = build_eq(as_flock_f128s(r_lo));
    let evaluation = claim_check(&eq_lo, &s_hat_v);
    let target = as_flock_f128s(core::slice::from_ref(&query.target))[0];
    if evaluation != target {
        return Err(CommitError::VerificationFailed);
    }
    // 7. Record Ring-Switch Message
    let mut challenger = ProverChallenger::new(transcript);
    challenger.observe_label(RING_SWITCH_LABEL);
    challenger.observe_f128_slice(&s_hat_v);
    // 8. Sample Ring-Switch Challenges
    let r_dprime = challenger.sample_f128_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);
    debug_assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    // 9. Compute the Ligerito Target
    let s_hat_u = tensor_algebra_transpose(&s_hat_v);
    let beta0 = inner_product(&s_hat_u, &eq_r_dprime);
    // 10. Build the Ligerito Basis
    let b_initial = fold_b128_elems(&eq_hi, &eq_r_dprime);
    debug_assert_eq!(b_initial.len(), packed_witness.len());
    // 11. Prove the Ligerito Claim
    let ligerito_proof = recursive_prover_with_basis(
        &ligerito_config,
        packed_witness,
        b_initial,
        beta0,
        &flock_data.codeword,
        &flock_data.merkle_tree,
        &mut challenger,
    );
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
