//! Standard multilinear verification over the FLoCK commitment.
//!
//! Verifier steps:
//! 1. Validate the evaluation point and derive the Ligerito verifier configuration.
//! 2. Bind the commitment root, trusted parameters, point, and target to the transcript.
//! 3. Read and deserialize the bounded opening proof.
//! 4. Validate the proof shape and require its initial root to match the commitment.
//! 5. Extract one ring-switch proof and split the point into low and high coordinates.
//! 6. Replay the ring-switch label and partial evaluations through `VerifierChallenger`.
//! 7. Check the target against the low-coordinate equality table.
//! 8. Sample seven ring-switch challenges and compute the packed target `beta0`.
//! 9. Build the succinct Ligerito basis evaluator from the high coordinates.
//! 10. Call `recursive_verifier_with_basis_succinct` against the commitment root.
//! 11. Reject FLoCK failures and transcript mismatches.
//!
//! The verifier checks `target = Σ_v eq(r_lo, v) · s_v`.
//! It samples `r_dprime` and computes `beta0 = Σ_u eq(r_dprime, u) · s_u`.
//! The succinct basis evaluates `B_hat`, where
//! `B(y) = Σ_u eq(r_dprime, u) · A(y, u)`.
//! Ligerito then verifies `Σ_y B(y) · q_pkd(y) = beta0` against the root.

use flock_core::challenger::Challenger;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ligerito::{VerifierConfig, recursive_verifier_with_basis_succinct};
use flock_core::pcs::ring_switch::{
    claim_check, eval_rs_eq_finish_from_prefix_binary_q, eval_rs_eq_prefix, inner_product,
    tensor_algebra_transpose,
};
use flock_core::pcs::{BatchOpeningProofLigerito, LOG_PACKING};
use flock_core::zerocheck::univariate_skip::build_eq;
use transcript::VerifierState;

use crate::bridge::as_flock_f128s;
use crate::challenger::VerifierChallenger;
use crate::protocol::{RING_SWITCH_LABEL, bind_statement_verifier, read_opening_proof};
use crate::{CommitError, Commitment, OpeningQuery, Pcs};

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Commitment,
    query: &OpeningQuery,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    // 1. Input Validation
    let m = pcs.params().m;
    if query.point.len() != m {
        return Err(CommitError::PointLengthMismatch);
    }
    let log_n = m
        .checked_sub(LOG_PACKING)
        .ok_or(CommitError::InvalidConfiguration)?;
    let ligerito_config = pcs
        .params()
        .ligerito_verifier_config()
        .map_err(|_| CommitError::InvalidConfiguration)?;
    let final_log_n = validate_config(&ligerito_config, log_n, pcs.params().log_batch_size)?;

    // 2. Bind Statement
    bind_statement_verifier(pcs, commitment.root(), query, transcript);

    // 3. Read Opening Proof
    let proof = read_opening_proof(transcript)?;

    // 4. Validate Proof Shape
    validate_proof_shape(&proof, &ligerito_config, final_log_n, commitment.root())?;

    // 5. Extract Ring-Switch Claim
    let ring_switch = proof
        .ring_switches
        .first()
        .ok_or(CommitError::MalformedProof)?;
    let (r_lo, r_hi) = query.point.split_at(LOG_PACKING);
    let r_hi = as_flock_f128s(r_hi);

    // 6. Replay Ring-Switch Message
    let mut challenger = VerifierChallenger::new(transcript);
    challenger.observe_label(RING_SWITCH_LABEL);
    challenger.observe_f128_slice(&ring_switch.s_hat_v);
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }

    // 7. Check Target
    // query.target = Σ_v eq(r_lo, v) · s_hat_v[v].
    let eq_lo = build_eq(as_flock_f128s(r_lo));
    let target = as_flock_f128s(core::slice::from_ref(&query.target))[0];
    if claim_check(&eq_lo, &ring_switch.s_hat_v) != target {
        return Err(CommitError::VerificationFailed);
    }

    // 8. Compute the Ligerito Target
    // beta0 = Σ_u eq(r_dprime, u) · s_hat_u[u].
    let r_dprime = challenger.sample_f128_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);
    let s_hat_u = tensor_algebra_transpose(&ring_switch.s_hat_v);
    let beta0 = inner_product(&s_hat_u, &eq_r_dprime);

    // 9. Build the Succinct Basis Evaluator
    // result[y] = B_hat(ris || bits(y)).
    let eval_b_residual = |ris: &[FlockF128], yr_log_n: usize| {
        if yr_log_n > 32 || ris.len().checked_add(yr_log_n) != Some(r_hi.len()) {
            return Vec::new();
        }
        let Some(yr_len) = 1usize.checked_shl(yr_log_n as u32) else {
            return Vec::new();
        };
        let prefix = eval_rs_eq_prefix(r_hi, ris);
        let suffix = &r_hi[ris.len()..];
        (0..yr_len)
            .map(|y| {
                eval_rs_eq_finish_from_prefix_binary_q(&prefix, suffix, y as u32, &eq_r_dprime)
            })
            .collect()
    };

    // 10. Verify the Ligerito Claim
    // Verify Σ_y B(y) · q_pkd(y) = beta0 without materializing B.
    let valid = recursive_verifier_with_basis_succinct(
        &ligerito_config,
        &proof.ligerito,
        log_n,
        beta0,
        commitment.root(),
        eval_b_residual,
        &mut challenger,
    );

    // 11. Check Verification Results
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }
    if !valid {
        return Err(CommitError::VerificationFailed);
    }
    Ok(())
}

fn validate_config(
    config: &VerifierConfig,
    log_n: usize,
    expected_initial_k: usize,
) -> Result<usize, CommitError> {
    let r = config.recursive_steps;
    let level_count = r.checked_add(1).ok_or(CommitError::InvalidConfiguration)?;
    if r == 0
        || config.recursive_ks.len() != r
        || config.recursive_log_msg_cols.len() != r
        || config.log_inv_rates.len() != level_count
        || config.queries.len() != level_count
        || config.grinding_bits.len() != level_count
        || config.fold_grinding_bits.len() != level_count
        || config.ood_samples.len() != level_count
        || config.initial_k != expected_initial_k
        || config.initial_log_num_interleaved != config.initial_k
        || config.ood_samples[0] != 0
        || config.log_inv_rates.contains(&0)
        || config
            .grinding_bits
            .iter()
            .chain(&config.fold_grinding_bits)
            .any(|&bits| u32::try_from(bits).is_err())
    {
        return Err(CommitError::InvalidConfiguration);
    }

    let mut remaining = log_n
        .checked_sub(config.initial_k)
        .ok_or(CommitError::InvalidConfiguration)?;
    if config.initial_log_msg_cols != remaining
        || checked_pow2(config.initial_k).is_none()
        || !valid_query_shape(remaining, config.log_inv_rates[0], config.queries[0])
    {
        return Err(CommitError::InvalidConfiguration);
    }

    for level in 0..r {
        let k = config.recursive_ks[level];
        if k == 0 || checked_pow2(k).is_none() {
            return Err(CommitError::InvalidConfiguration);
        }
        remaining = remaining
            .checked_sub(k)
            .ok_or(CommitError::InvalidConfiguration)?;
        if config.recursive_log_msg_cols[level] != remaining
            || !valid_query_shape(
                remaining,
                config.log_inv_rates[level + 1],
                config.queries[level + 1],
            )
        {
            return Err(CommitError::InvalidConfiguration);
        }
    }

    if remaining > 32 || checked_pow2(remaining).is_none() {
        return Err(CommitError::InvalidConfiguration);
    }
    Ok(remaining)
}

fn valid_query_shape(log_columns: usize, log_rate: usize, queries: usize) -> bool {
    queries > 0
        && log_columns
            .checked_add(log_rate)
            .and_then(checked_pow2)
            .is_some_and(|block_len| queries <= block_len)
}

fn checked_pow2(log: usize) -> Option<usize> {
    u32::try_from(log)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
}

fn validate_proof_shape(
    proof: &BatchOpeningProofLigerito,
    config: &VerifierConfig,
    final_log_n: usize,
    expected_root: &[u8; 32],
) -> Result<(), CommitError> {
    if proof.ring_switches.len() != 1
        || proof.ring_switches[0].s_hat_v.len() != 1usize << LOG_PACKING
        || &proof.ligerito.initial_root != expected_root
    {
        return Err(CommitError::MalformedProof);
    }

    let lig = &proof.ligerito;
    let r = config.recursive_steps;
    if lig.recursive_roots.len() != r
        || lig.recursive_proofs.len() != r - 1
        || lig.grinding_nonces.len() != r + 1
    {
        return Err(CommitError::MalformedProof);
    }

    let expected_ood = config
        .ood_samples
        .iter()
        .skip(1)
        .try_fold(0usize, |sum, &count| sum.checked_add(count))
        .ok_or(CommitError::InvalidConfiguration)?;
    let expected_fold_nonces = positive_fold_nonce_count(config)?;
    let expected_sumchecks = config
        .recursive_ks
        .iter()
        .try_fold(1usize + config.initial_k, |sum, &k| sum.checked_add(k))
        .and_then(|sum| sum.checked_add(r))
        .and_then(|sum| sum.checked_add(expected_ood))
        .ok_or(CommitError::InvalidConfiguration)?;
    if lig.ood_values.len() != expected_ood
        || lig.fold_grinding_nonces.len() != expected_fold_nonces
        || lig.sumcheck_transcript.len() != expected_sumchecks
    {
        return Err(CommitError::MalformedProof);
    }

    let initial_width = checked_pow2(config.initial_k).ok_or(CommitError::InvalidConfiguration)?;
    if !rows_match(
        &lig.initial_proof.opened_rows,
        config.queries[0],
        initial_width,
    ) {
        return Err(CommitError::MalformedProof);
    }
    for (level, recursive) in lig.recursive_proofs.iter().enumerate() {
        let width =
            checked_pow2(config.recursive_ks[level]).ok_or(CommitError::InvalidConfiguration)?;
        if !rows_match(&recursive.opened_rows, config.queries[level + 1], width) {
            return Err(CommitError::MalformedProof);
        }
    }

    let last_k = *config
        .recursive_ks
        .last()
        .ok_or(CommitError::InvalidConfiguration)?;
    let final_width = checked_pow2(last_k).ok_or(CommitError::InvalidConfiguration)?;
    let final_yr_len = checked_pow2(final_log_n).ok_or(CommitError::InvalidConfiguration)?;
    if !rows_match(&lig.final_proof.opened_rows, config.queries[r], final_width)
        || lig.final_proof.yr.len() != final_yr_len
    {
        return Err(CommitError::MalformedProof);
    }
    Ok(())
}

fn positive_fold_nonce_count(config: &VerifierConfig) -> Result<usize, CommitError> {
    let initial = config.initial_k.min(config.fold_grinding_bits[0]);
    config
        .recursive_ks
        .iter()
        .zip(config.fold_grinding_bits.iter().skip(1))
        .try_fold(initial, |sum, (&k, &bits)| sum.checked_add(k.min(bits)))
        .ok_or(CommitError::InvalidConfiguration)
}

fn rows_match(rows: &[Vec<FlockF128>], expected_rows: usize, expected_width: usize) -> bool {
    rows.len() == expected_rows && rows.iter().all(|row| row.len() == expected_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HashKind, LigeritoProfile};

    fn registered_config() -> (VerifierConfig, usize, usize) {
        let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
        let config = pcs.params().ligerito_verifier_config().unwrap();
        (
            config,
            pcs.params().m - LOG_PACKING,
            pcs.params().log_batch_size,
        )
    }

    #[test]
    fn registered_verifier_config_passes_local_validation() {
        let (config, log_n, initial_k) = registered_config();
        assert!(validate_config(&config, log_n, initial_k).is_ok());
    }

    #[test]
    fn config_validation_rejects_invalid_shapes_and_values() {
        let (valid, log_n, initial_k) = registered_config();

        let mut config = valid.clone();
        config.log_inv_rates.pop();
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::InvalidConfiguration)
        );

        let mut config = valid.clone();
        config.log_inv_rates[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::InvalidConfiguration)
        );

        let mut config = valid.clone();
        config.queries[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::InvalidConfiguration)
        );

        let mut config = valid.clone();
        config.recursive_ks[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::InvalidConfiguration)
        );

        let mut config = valid;
        config.queries[0] = usize::MAX;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::InvalidConfiguration)
        );
    }

    #[test]
    fn checked_pow2_and_rows_match_reject_bad_shapes() {
        assert_eq!(checked_pow2(0), Some(1));
        assert_eq!(checked_pow2(3), Some(8));
        assert_eq!(checked_pow2(usize::BITS as usize), None);

        let zero = FlockF128::ZERO;
        assert!(rows_match(&[vec![zero; 4], vec![zero; 4]], 2, 4));
        assert!(!rows_match(&[vec![zero; 4]], 2, 4));
        assert!(!rows_match(&[vec![zero; 3], vec![zero; 4]], 2, 4));
    }
}
