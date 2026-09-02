//! Standard multilinear verification over the Flock commitment.
//!
//! Verifier steps:
//! 1. Validate the evaluation point and derive the Ligerito verifier configuration.
//! 2. Bind the commitment root, trusted parameters, point, and target to the transcript.
//! 3. Read and deserialize the bounded opening proof.
//! 4. Validate the proof shape and require its initial root to match the commitment.
//! 5. Extract one ring-switch proof and split the point into low and high coordinates.
//! 6. Replay the tag-4001 ring-switch message.
//! 7. Check the target against the low-coordinate equality table.
//! 8. Sample seven tag-4101 challenges and compute the packed target `beta0`.
//! 9. Build the succinct Ligerito basis evaluator from the high coordinates.
//! 10. Bind tag 5001 and call `recursive_verifier_with_basis_succinct`.
//! 11. Reject Flock failures and transcript mismatches.
//!
//! The verifier checks `target = Σ_v eq(r_lo, v) · s_v`.
//! It samples `r_dprime` and computes `beta0 = Σ_u eq(r_dprime, u) · s_u`.
//! The succinct basis evaluates `B_hat`, where
//! `B(y) = Σ_u eq(r_dprime, u) · A(y, u)`.
//! Ligerito then verifies `Σ_y B(y) · q_pkd(y) = beta0` against the root.

use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ligerito::{
    LigeritoProof, VerifierConfig, recursive_verifier_with_basis_succinct,
};
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
    sample_ring_switch_point,
};
use crate::{CommitError, Commitment, Pcs, StatementBinding};

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Commitment,
    point: &[field::F128],
    target: field::F128,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    // 1. Input Validation
    let m = pcs.params().m;
    if point.len() != m {
        return Err(CommitError::PointLengthMismatch);
    }
    let log_n = m.checked_sub(LOG_PACKING).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "PCS variable count {m} is smaller than the packing width {LOG_PACKING}"
        ))
    })?;
    let ligerito_config = pcs
        .params()
        .ligerito_verifier_config()
        .map_err(CommitError::InvalidConfiguration)?;
    let final_log_n = pcs.final_log_n();

    // 2. Bind Statement
    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, commitment.root(), point, target, transcript);
    }

    // 3. Read Opening Proof
    let proof = read_opening_proof(transcript)?;

    // 4. Validate Proof Shape
    validate_proof_shape(&proof, &ligerito_config, final_log_n, commitment.root())?;

    // 5. Extract Ring-Switch Claim
    let ring_switch = proof
        .ring_switches
        .first()
        .ok_or(CommitError::MalformedProof)?;
    let (r_lo, r_hi) = point.split_at(LOG_PACKING);
    let r_hi = as_flock_f128s(r_hi);

    // 6. Replay Ring-Switch Message
    bind_ring_switch_message(transcript, &ring_switch.s_hat_v)?;

    // 7. Check Target
    // target = Σ_v eq(r_lo, v) · s_hat_v[v].
    let eq_lo = build_eq(as_flock_f128s(r_lo));
    let target = as_flock_f128s(core::slice::from_ref(&target))[0];
    if claim_check(&eq_lo, &ring_switch.s_hat_v) != target {
        return Err(CommitError::VerificationFailed);
    }

    // 8. Compute the Ligerito Target
    // beta0 = Σ_u eq(r_dprime, u) · s_hat_u[u].
    let r_dprime = sample_ring_switch_point(transcript);
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
    observe_opening_target(transcript, log_n, beta0)?;
    let mut challenger = VerifierChallenger::new_ligerito(transcript, beta0);
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

pub(crate) fn validate_config(
    config: &VerifierConfig,
    log_n: usize,
    expected_initial_k: usize,
) -> Result<usize, CommitError> {
    let r = config.recursive_steps;
    let level_count = r.checked_add(1).ok_or_else(|| {
        CommitError::invalid_configuration("recursive step count overflows the level count")
    })?;
    if r == 0 {
        return Err(CommitError::invalid_configuration(
            "verifier configuration has no recursive steps",
        ));
    }
    if config.recursive_ks.len() != r {
        return Err(CommitError::invalid_configuration(format!(
            "recursive_ks length {} does not match recursive_steps {r}",
            config.recursive_ks.len(),
        )));
    }
    if config.recursive_log_msg_cols.len() != r {
        return Err(CommitError::invalid_configuration(format!(
            "recursive_log_msg_cols length {} does not match recursive_steps {r}",
            config.recursive_log_msg_cols.len(),
        )));
    }
    if config.log_inv_rates.len() != level_count {
        return Err(CommitError::invalid_configuration(format!(
            "log_inv_rates length {} does not match level count {level_count}",
            config.log_inv_rates.len(),
        )));
    }
    if config.queries.len() != level_count {
        return Err(CommitError::invalid_configuration(format!(
            "queries length {} does not match level count {level_count}",
            config.queries.len(),
        )));
    }
    if config.grinding_bits.len() != level_count {
        return Err(CommitError::invalid_configuration(format!(
            "grinding_bits length {} does not match level count {level_count}",
            config.grinding_bits.len(),
        )));
    }
    if config.fold_grinding_bits.len() != level_count {
        return Err(CommitError::invalid_configuration(format!(
            "fold_grinding_bits length {} does not match level count {level_count}",
            config.fold_grinding_bits.len(),
        )));
    }
    if config.ood_samples.len() != level_count {
        return Err(CommitError::invalid_configuration(format!(
            "ood_samples length {} does not match level count {level_count}",
            config.ood_samples.len(),
        )));
    }
    if config.initial_k != expected_initial_k {
        return Err(CommitError::invalid_configuration(format!(
            "initial_k {} does not match expected value {expected_initial_k}",
            config.initial_k,
        )));
    }
    if config.initial_log_num_interleaved != config.initial_k {
        return Err(CommitError::invalid_configuration(format!(
            "initial_log_num_interleaved {} does not match initial_k {}",
            config.initial_log_num_interleaved, config.initial_k,
        )));
    }
    if config.ood_samples[0] != 0 {
        return Err(CommitError::invalid_configuration(format!(
            "ood_samples[0] must be zero, got {}",
            config.ood_samples[0],
        )));
    }
    if let Some(level) = config.log_inv_rates.iter().position(|&rate| rate == 0) {
        return Err(CommitError::invalid_configuration(format!(
            "log_inv_rates[{level}] must be positive"
        )));
    }
    if let Some((level, bits)) = config
        .grinding_bits
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, bits)| u32::try_from(bits).is_err())
    {
        return Err(CommitError::invalid_configuration(format!(
            "grinding_bits at level {level} exceeds u32: {bits}"
        )));
    }
    if let Some((level, bits)) = config
        .fold_grinding_bits
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, bits)| u32::try_from(bits).is_err())
    {
        return Err(CommitError::invalid_configuration(format!(
            "fold_grinding_bits at level {level} exceeds u32: {bits}"
        )));
    }

    let mut remaining = log_n.checked_sub(config.initial_k).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "initial_k {} exceeds log message length {log_n}",
            config.initial_k,
        ))
    })?;
    if config.initial_log_msg_cols != remaining {
        return Err(CommitError::invalid_configuration(format!(
            "initial_log_msg_cols {} does not match expected value {remaining}",
            config.initial_log_msg_cols,
        )));
    }
    if checked_pow2(config.initial_k).is_none() {
        return Err(CommitError::invalid_configuration(format!(
            "initial_k {} cannot be represented as a usize power of two",
            config.initial_k,
        )));
    }
    if !valid_query_shape(remaining, config.log_inv_rates[0], config.queries[0]) {
        return Err(CommitError::invalid_configuration(format!(
            "initial query shape is invalid: log_columns={remaining}, log_inv_rate={}, queries={}",
            config.log_inv_rates[0], config.queries[0],
        )));
    }

    for level in 0..r {
        let k = config.recursive_ks[level];
        if k == 0 {
            return Err(CommitError::invalid_configuration(format!(
                "recursive_ks[{level}] must be positive"
            )));
        }
        if checked_pow2(k).is_none() {
            return Err(CommitError::invalid_configuration(format!(
                "recursive_ks[{level}] cannot be represented as a usize power of two: {k}"
            )));
        }
        remaining = remaining.checked_sub(k).ok_or_else(|| {
            CommitError::invalid_configuration(format!(
                "recursive_ks[{level}] exceeds the remaining log columns: {k} > {remaining}"
            ))
        })?;
        if config.recursive_log_msg_cols[level] != remaining {
            return Err(CommitError::invalid_configuration(format!(
                "recursive_log_msg_cols[{level}] is {}, expected {remaining}",
                config.recursive_log_msg_cols[level],
            )));
        }
        if !valid_query_shape(
            remaining,
            config.log_inv_rates[level + 1],
            config.queries[level + 1],
        ) {
            return Err(CommitError::invalid_configuration(format!(
                "query shape at recursive level {level} is invalid: log_columns={remaining}, log_inv_rate={}, queries={}",
                config.log_inv_rates[level + 1],
                config.queries[level + 1],
            )));
        }
    }

    if remaining > 32 {
        return Err(CommitError::invalid_configuration(format!(
            "final log message columns {remaining} exceed the supported maximum 32"
        )));
    }
    if checked_pow2(remaining).is_none() {
        return Err(CommitError::invalid_configuration(format!(
            "final log message columns {remaining} cannot be represented as a usize power of two"
        )));
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
    {
        return Err(CommitError::VerificationFailed);
    }

    validate_ligerito_proof_shape(&proof.ligerito, config, final_log_n, expected_root)
}

pub(crate) fn validate_ligerito_proof_shape(
    lig: &LigeritoProof,
    config: &VerifierConfig,
    final_log_n: usize,
    expected_root: &[u8; 32],
) -> Result<(), CommitError> {
    if &lig.initial_root != expected_root {
        return Err(CommitError::VerificationFailed);
    }

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
        .ok_or_else(|| {
            CommitError::invalid_configuration("sum of ood_samples[1..] overflows usize")
        })?;
    let expected_fold_nonces = positive_fold_nonce_count(config)?;
    let initial_sumchecks = 1usize.checked_add(config.initial_k).ok_or_else(|| {
        CommitError::invalid_configuration("initial sumcheck transcript length overflows usize")
    })?;
    let expected_sumchecks = config
        .recursive_ks
        .iter()
        .try_fold(initial_sumchecks, |sum, &k| sum.checked_add(k))
        .and_then(|sum| sum.checked_add(r))
        .and_then(|sum| sum.checked_add(expected_ood))
        .ok_or_else(|| {
            CommitError::invalid_configuration(
                "expected sumcheck transcript length overflows usize",
            )
        })?;
    if lig.ood_values.len() != expected_ood
        || lig.fold_grinding_nonces.len() != expected_fold_nonces
        || lig.sumcheck_transcript.len() != expected_sumchecks
    {
        return Err(CommitError::VerificationFailed);
    }

    let initial_width = checked_pow2(config.initial_k).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "initial_k {} cannot be represented as a usize power of two",
            config.initial_k,
        ))
    })?;
    if !rows_match(
        &lig.initial_proof.opened_rows,
        config.queries[0],
        initial_width,
    ) {
        return Err(CommitError::VerificationFailed);
    }
    for (level, recursive) in lig.recursive_proofs.iter().enumerate() {
        let width = checked_pow2(config.recursive_ks[level]).ok_or_else(|| {
            CommitError::invalid_configuration(format!(
                "recursive_ks[{level}] cannot be represented as a usize power of two: {}",
                config.recursive_ks[level],
            ))
        })?;
        if !rows_match(&recursive.opened_rows, config.queries[level + 1], width) {
            return Err(CommitError::VerificationFailed);
        }
    }

    let last_k = *config
        .recursive_ks
        .last()
        .ok_or_else(|| CommitError::invalid_configuration("recursive_ks is empty"))?;
    let final_width = checked_pow2(last_k).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "last recursive_ks value cannot be represented as a usize power of two: {last_k}"
        ))
    })?;
    let final_yr_len = checked_pow2(final_log_n).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "final_log_n cannot be represented as a usize power of two: {final_log_n}"
        ))
    })?;
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
        .ok_or_else(|| {
            CommitError::invalid_configuration("expected fold grinding nonce count overflows usize")
        })
}

fn rows_match(rows: &[Vec<FlockF128>], expected_rows: usize, expected_width: usize) -> bool {
    rows.len() == expected_rows && rows.iter().all(|row| row.len() == expected_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommitScheme, HashKind, LigeritoProfile, OpeningQuery};
    use common::Shape;
    use field::F128 as LocalF128;
    use transcript::{build_prover, build_verifier};

    fn registered_config() -> (VerifierConfig, usize, usize) {
        let shape = Shape::new(7, 15).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
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
        let expected_levels = config.recursive_steps + 1;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(format!(
                "log_inv_rates length {} does not match level count {expected_levels}",
                config.log_inv_rates.len(),
            )))
        );

        let mut config = valid.clone();
        config.log_inv_rates[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "log_inv_rates[0] must be positive"
            ))
        );

        let mut config = valid.clone();
        config.queries[0] = 0;
        let initial_log_columns = log_n - config.initial_k;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(format!(
                "initial query shape is invalid: log_columns={initial_log_columns}, log_inv_rate={}, queries=0",
                config.log_inv_rates[0],
            )))
        );

        let mut config = valid.clone();
        config.recursive_ks[0] = 0;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "recursive_ks[0] must be positive"
            ))
        );

        let mut config = valid;
        config.queries[0] = usize::MAX;
        let initial_log_columns = log_n - config.initial_k;
        assert_eq!(
            validate_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(format!(
                "initial query shape is invalid: log_columns={initial_log_columns}, log_inv_rate={}, queries={}",
                config.log_inv_rates[0],
                usize::MAX,
            )))
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

    #[test]
    fn proof_shape_validation_rejects_prover_supplied_dimension_mismatches() {
        const SESSION: &[u8] = b"pcs-proof-shape-test";
        const INSTANCE: &[u8] = b"zero-polynomial";
        let shape = Shape::new(7, 15).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let packed_witness = vec![LocalF128::default(); pcs.packed_len()];
        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let query = OpeningQuery::Mle {
            point: vec![LocalF128::from(2u64); 22],
            target: LocalF128::default(),
        };
        let mut prover = build_prover(SESSION, INSTANCE);
        pcs.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        )
        .unwrap();
        let transcript_proof = prover.finish();
        let mut verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        let valid = read_opening_proof(&mut verifier).unwrap();
        let config = pcs.params().ligerito_verifier_config().unwrap();
        let final_log_n =
            validate_config(&config, 22 - LOG_PACKING, pcs.params().log_batch_size).unwrap();
        let root = *commitment.root();
        assert_eq!(
            validate_proof_shape(&valid, &config, final_log_n, &root),
            Ok(())
        );

        type ProofMutation = fn(&mut BatchOpeningProofLigerito);
        let mutations: [(&str, ProofMutation); 3] = [
            ("ring-switch count", |proof| proof.ring_switches.clear()),
            ("recursive-root count", |proof| {
                proof.ligerito.recursive_roots.pop();
            }),
            ("opened-row width", |proof| {
                proof.ligerito.initial_proof.opened_rows[0].pop();
            }),
        ];

        for (case, mutate) in mutations {
            let mut proof = valid.clone();
            mutate(&mut proof);
            assert_eq!(
                validate_proof_shape(&proof, &config, final_log_n, &root),
                Err(CommitError::VerificationFailed),
                "{case}",
            );
        }
    }
}
