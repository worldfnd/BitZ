//! Batched multilinear verification over the Flock commitment.
//!
//! Verifier steps:
//! 1. Validate every evaluation point and derive the Ligerito verifier configuration.
//! 2. Bind the commitment root, trusted parameters, and queries to the transcript.
//! 3. Read and deserialize the bounded opening proof.
//! 4. Validate the proof shape and require its initial root to match the commitment.
//! 5. Replay and check every ring-switch message before sampling challenges.
//! 6. Sample one shared ring-switch point and one batching scalar per query.
//! 7. Combine the targets and succinct basis evaluators.
//! 8. Call `recursive_verifier_with_basis_succinct` against the commitment root.
//! 9. Reject Flock failures and transcript mismatches.
//!
//! Each query produces one packed claim under the shared ring-switch point.
//! The verifier combines these claims with independent batching scalars.
//! Ligerito verifies the resulting claim against the commitment root.

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
use crate::utils::{
    bind_ring_switch_message, bind_statement, observe_opening_target, read_opening_proof,
    sample_batching_scalars, sample_shared_ring_switch_point, validate_batch,
};
use crate::{CommitError, Commitment, Pcs, ScopedOpeningQuery, StatementBinding};

pub(crate) fn verify_batch(
    pcs: &Pcs,
    commitment: &Commitment,
    queries: &[ScopedOpeningQuery<'_>],
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    // 1. Input Validation
    let m = pcs.params().m;
    validate_batch(queries, m)?;
    let log_n = m
        .checked_sub(LOG_PACKING)
        .ok_or_else(|| CommitError::invalid_configuration("m is below LOG_PACKING"))?;
    let ligerito_config = pcs
        .params()
        .ligerito_verifier_config()
        .map_err(CommitError::InvalidConfiguration)?;
    let final_log_n = validate_config(&ligerito_config, log_n, pcs.params().log_batch_size)?;

    // 2. Bind Statement
    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, commitment.root(), queries, transcript);
    }

    // 3. Read Opening Proof
    let proof = read_opening_proof(transcript)?;

    // 4. Validate Proof Shape
    validate_proof_shape(
        &proof,
        &ligerito_config,
        final_log_n,
        commitment.root(),
        queries.len(),
    )?;

    // 5. Replay Ring-Switch Messages and Check Targets
    // Normative verifier replay through the batched opening:
    // https://github.com/worldfnd/f2z-benchmark/blob/5014c717e88ab5e54e70e7a1099caaca5c41a926/docs/f2z-pcs-spec/part4-verifier.tex#L205-L250
    let mut r_his = Vec::with_capacity(queries.len());
    for (scoped_query, ring_switch) in queries.iter().zip(&proof.ring_switches) {
        let query = scoped_query.query;
        let (r_lo, r_hi) = query.point.split_at(LOG_PACKING);
        r_his.push(as_flock_f128s(r_hi));

        bind_ring_switch_message(transcript, scoped_query.scope, &ring_switch.s_hat_v)?;

        // query.target = Σ_v eq(r_lo, v) · s_hat_v[v].
        let eq_lo = build_eq(as_flock_f128s(r_lo));
        let target = as_flock_f128s(core::slice::from_ref(&query.target))[0];
        if claim_check(&eq_lo, &ring_switch.s_hat_v) != target {
            return Err(CommitError::VerificationFailed);
        }
    }
    let r_dprime = sample_shared_ring_switch_point(transcript);

    // 6. Compute the Batched Ligerito Target
    let eq_r_dprime = build_eq(&r_dprime);
    let etas = sample_batching_scalars(transcript, queries.iter().map(|query| query.scope));
    let beta =
        proof
            .ring_switches
            .iter()
            .zip(&etas)
            .fold(FlockF128::ZERO, |sum, (ring_switch, &eta)| {
                let s_hat_u = tensor_algebra_transpose(&ring_switch.s_hat_v);
                sum + eta * inner_product(&s_hat_u, &eq_r_dprime)
            });

    // 7. Build the Batched Succinct Basis Evaluator
    let eval_b_residual = |ris: &[FlockF128], yr_log_n: usize| {
        if yr_log_n > 32
            || r_his
                .iter()
                .any(|r_hi| ris.len().checked_add(yr_log_n) != Some(r_hi.len()))
        {
            return Vec::new();
        }
        let Some(yr_len) = 1usize.checked_shl(yr_log_n as u32) else {
            return Vec::new();
        };
        let mut result = vec![FlockF128::ZERO; yr_len];
        for (r_hi, &eta) in r_his.iter().zip(&etas) {
            let prefix = eval_rs_eq_prefix(r_hi, ris);
            let suffix = &r_hi[ris.len()..];
            for (y, value) in result.iter_mut().enumerate() {
                *value += eta
                    * eval_rs_eq_finish_from_prefix_binary_q(
                        &prefix,
                        suffix,
                        y as u32,
                        &eq_r_dprime,
                    );
            }
        }
        result
    };

    // 8. Verify the Batched Ligerito Claim
    observe_opening_target(transcript, log_n, beta)?;
    let mut challenger = VerifierChallenger::new_ligerito(transcript, beta);
    let valid = recursive_verifier_with_basis_succinct(
        &ligerito_config,
        &proof.ligerito,
        log_n,
        beta,
        commitment.root(),
        eval_b_residual,
        &mut challenger,
    );

    // 9. Check Verification Results
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }
    if !valid {
        return Err(CommitError::VerificationFailed);
    }
    Ok(())
}

pub(crate) fn validate_config(
    config: &VerifierConfig,
    log_n: usize,
    expected_initial_k: usize,
) -> Result<usize, CommitError> {
    let r = config.recursive_steps;
    let level_count = r
        .checked_add(1)
        .ok_or_else(|| CommitError::invalid_configuration("recursive_steps is too large"))?;
    if r == 0 {
        return Err(CommitError::invalid_configuration(
            "recursive_steps is zero",
        ));
    }
    if config.recursive_ks.len() != r {
        return Err(CommitError::invalid_configuration(
            "recursive_ks length mismatch",
        ));
    }
    if config.recursive_log_msg_cols.len() != r {
        return Err(CommitError::invalid_configuration(
            "recursive_log_msg_cols length mismatch",
        ));
    }
    if config.log_inv_rates.len() != level_count {
        return Err(CommitError::invalid_configuration(
            "log_inv_rates length mismatch",
        ));
    }
    if config.queries.len() != level_count {
        return Err(CommitError::invalid_configuration(
            "queries length mismatch",
        ));
    }
    if config.grinding_bits.len() != level_count {
        return Err(CommitError::invalid_configuration(
            "grinding_bits length mismatch",
        ));
    }
    if config.fold_grinding_bits.len() != level_count {
        return Err(CommitError::invalid_configuration(
            "fold_grinding_bits length mismatch",
        ));
    }
    if config.ood_samples.len() != level_count {
        return Err(CommitError::invalid_configuration(
            "ood_samples length mismatch",
        ));
    }
    if config.initial_k != expected_initial_k {
        return Err(CommitError::invalid_configuration("initial_k mismatch"));
    }
    if config.initial_log_num_interleaved != config.initial_k {
        return Err(CommitError::invalid_configuration(
            "initial_log_num_interleaved mismatch",
        ));
    }
    if config.ood_samples[0] != 0 {
        return Err(CommitError::invalid_configuration(
            "ood_samples[0] is nonzero",
        ));
    }
    if let Some(level) = config.log_inv_rates.iter().position(|&rate| rate == 0) {
        return Err(CommitError::invalid_configuration(format!(
            "log_inv_rates[{level}] is zero"
        )));
    }
    if let Some((level, _)) = config
        .grinding_bits
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, bits)| u32::try_from(bits).is_err())
    {
        return Err(CommitError::invalid_configuration(format!(
            "grinding_bits[{level}] exceeds u32"
        )));
    }
    if let Some((level, _)) = config
        .fold_grinding_bits
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, bits)| u32::try_from(bits).is_err())
    {
        return Err(CommitError::invalid_configuration(format!(
            "fold_grinding_bits[{level}] exceeds u32"
        )));
    }

    let mut remaining = log_n
        .checked_sub(config.initial_k)
        .ok_or_else(|| CommitError::invalid_configuration("initial_k exceeds log_n"))?;
    if config.initial_log_msg_cols != remaining {
        return Err(CommitError::invalid_configuration(
            "initial_log_msg_cols mismatch",
        ));
    }
    if checked_pow2(config.initial_k).is_none() {
        return Err(CommitError::invalid_configuration("initial_k is too large"));
    }
    if !valid_query_shape(remaining, config.log_inv_rates[0], config.queries[0]) {
        return Err(CommitError::invalid_configuration(
            "invalid initial query shape",
        ));
    }

    for level in 0..r {
        let k = config.recursive_ks[level];
        if k == 0 {
            return Err(CommitError::invalid_configuration(format!(
                "recursive_ks[{level}] is zero"
            )));
        }
        if checked_pow2(k).is_none() {
            return Err(CommitError::invalid_configuration(format!(
                "recursive_ks[{level}] is too large"
            )));
        }
        remaining = remaining.checked_sub(k).ok_or_else(|| {
            CommitError::invalid_configuration(format!(
                "recursive_ks[{level}] exceeds remaining columns"
            ))
        })?;
        if config.recursive_log_msg_cols[level] != remaining {
            return Err(CommitError::invalid_configuration(format!(
                "recursive_log_msg_cols[{level}] mismatch"
            )));
        }
        if !valid_query_shape(
            remaining,
            config.log_inv_rates[level + 1],
            config.queries[level + 1],
        ) {
            return Err(CommitError::invalid_configuration(format!(
                "invalid recursive query shape at level {level}"
            )));
        }
    }

    if remaining > 32 {
        return Err(CommitError::invalid_configuration(
            "final log columns exceed 32",
        ));
    }
    if checked_pow2(remaining).is_none() {
        return Err(CommitError::invalid_configuration(
            "final log columns are too large",
        ));
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
    expected_ring_switches: usize,
) -> Result<(), CommitError> {
    if proof.ring_switches.len() != expected_ring_switches
        || proof
            .ring_switches
            .iter()
            .any(|ring_switch| ring_switch.s_hat_v.len() != 1usize << LOG_PACKING)
        || &proof.ligerito.initial_root != expected_root
    {
        return Err(CommitError::VerificationFailed);
    }

    let lig = &proof.ligerito;
    let r = config.recursive_steps;
    if lig.recursive_roots.len() != r
        || lig.recursive_proofs.len() != r - 1
        || lig.grinding_nonces.len() != r + 1
    {
        return Err(CommitError::VerificationFailed);
    }

    let expected_ood = config
        .ood_samples
        .iter()
        .skip(1)
        .try_fold(0usize, |sum, &count| sum.checked_add(count))
        .ok_or_else(|| CommitError::invalid_configuration("ood_samples sum is too large"))?;
    let expected_fold_nonces = positive_fold_nonce_count(config)?;
    let initial_sumchecks = 1usize.checked_add(config.initial_k).ok_or_else(|| {
        CommitError::invalid_configuration("initial sumcheck length is too large")
    })?;
    let expected_sumchecks = config
        .recursive_ks
        .iter()
        .try_fold(initial_sumchecks, |sum, &k| sum.checked_add(k))
        .and_then(|sum| sum.checked_add(r))
        .and_then(|sum| sum.checked_add(expected_ood))
        .ok_or_else(|| CommitError::invalid_configuration("sumcheck length is too large"))?;
    if lig.ood_values.len() != expected_ood
        || lig.fold_grinding_nonces.len() != expected_fold_nonces
        || lig.sumcheck_transcript.len() != expected_sumchecks
    {
        return Err(CommitError::VerificationFailed);
    }

    let initial_width = checked_pow2(config.initial_k)
        .ok_or_else(|| CommitError::invalid_configuration("initial_k is too large"))?;
    if !rows_match(
        &lig.initial_proof.opened_rows,
        config.queries[0],
        initial_width,
    ) {
        return Err(CommitError::VerificationFailed);
    }
    for (level, recursive) in lig.recursive_proofs.iter().enumerate() {
        let width = checked_pow2(config.recursive_ks[level]).ok_or_else(|| {
            CommitError::invalid_configuration(format!("recursive_ks[{level}] is too large"))
        })?;
        if !rows_match(&recursive.opened_rows, config.queries[level + 1], width) {
            return Err(CommitError::VerificationFailed);
        }
    }

    let last_k = *config
        .recursive_ks
        .last()
        .ok_or_else(|| CommitError::invalid_configuration("recursive_ks is empty"))?;
    let final_width = checked_pow2(last_k)
        .ok_or_else(|| CommitError::invalid_configuration("last recursive_k is too large"))?;
    let final_yr_len = checked_pow2(final_log_n)
        .ok_or_else(|| CommitError::invalid_configuration("final_log_n is too large"))?;
    if !rows_match(&lig.final_proof.opened_rows, config.queries[r], final_width)
        || lig.final_proof.yr.len() != final_yr_len
    {
        return Err(CommitError::VerificationFailed);
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
        .ok_or_else(|| CommitError::invalid_configuration("fold nonce count is too large"))
}

fn rows_match(rows: &[Vec<FlockF128>], expected_rows: usize, expected_width: usize) -> bool {
    rows.len() == expected_rows && rows.iter().all(|row| row.len() == expected_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HashKind, LigeritoProfile};

    fn registered_config() -> (VerifierConfig, usize, usize) {
        let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
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
            Err(CommitError::invalid_configuration(
                "log_inv_rates length mismatch"
            ))
        );

        let mut config = valid.clone();
        config.log_inv_rates[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "log_inv_rates[0] is zero"
            ))
        );

        let mut config = valid.clone();
        config.queries[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "invalid initial query shape"
            ))
        );

        let mut config = valid.clone();
        config.recursive_ks[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "recursive_ks[0] is zero"
            ))
        );

        let mut config = valid;
        config.queries[0] = usize::MAX;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "invalid initial query shape"
            ))
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
