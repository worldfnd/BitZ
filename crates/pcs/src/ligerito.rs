//! Shared Ligerito for reduced linear claims.
//!
//! Query modules derive one packed basis and target through their ring switch.
//! This module validates the prover input and runs the common opening protocol.

use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::LOG_PACKING;
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::{
    LigeritoProof, ProverConfig, VerifierConfig, recursive_prover_with_basis,
    recursive_verifier_with_basis, recursive_verifier_with_basis_succinct,
};
use transcript::{ProverState, VerifierState};

use crate::bridge::into_flock_f128s;
use crate::challenger::{ProverChallenger, VerifierChallenger};
use crate::utils::{observe_opening_target, read_opening_proof, write_opening_proof};
use crate::{CommitError, Pcs, ProverData, Root};

#[derive(Clone, Debug)]
pub(crate) struct CheckedLigerito {
    prover_config: ProverConfig,
    verifier_config: VerifierConfig,
    log_n_u32: u32,
    final_log_n: usize,
}

impl CheckedLigerito {
    pub(crate) fn new(params: &PcsParams) -> Result<Self, CommitError> {
        let log_n = params.m.checked_sub(LOG_PACKING).ok_or_else(|| {
            CommitError::invalid_configuration(format!(
                "PCS variable count {} is smaller than the packing width {LOG_PACKING}",
                params.m,
            ))
        })?;
        let log_n_u32 = u32::try_from(log_n).map_err(|_| {
            CommitError::invalid_configuration("packed witness variable count exceeds u32")
        })?;
        let prover_config = params
            .ligerito_prover_config()
            .map_err(CommitError::InvalidConfiguration)?;
        let verifier_config = params
            .ligerito_verifier_config()
            .map_err(CommitError::InvalidConfiguration)?;
        validate_pcs_verifier_prover(params, &prover_config, &verifier_config)?;
        let final_log_n = validate_verifier_config(&verifier_config, log_n, params.log_batch_size)?;

        Ok(Self {
            prover_config,
            verifier_config,
            log_n_u32,
            final_log_n,
        })
    }

    pub(crate) fn prover_config(&self) -> &ProverConfig {
        &self.prover_config
    }

    pub(crate) fn verifier_config(&self) -> &VerifierConfig {
        &self.verifier_config
    }

    pub(crate) fn log_n_u32(&self) -> u32 {
        self.log_n_u32
    }

    pub(crate) fn final_log_n(&self) -> usize {
        self.final_log_n
    }
}

pub(crate) struct ReducedClaim {
    pub(crate) packed_basis: Vec<FlockF128>,
    pub(crate) packed_target: FlockF128,
}

pub(crate) struct ReducedProver<'a> {
    pcs: &'a Pcs,
    data: &'a ProverData,
    packed_witness: Vec<FlockF128>,
}

impl<'a> ReducedProver<'a> {
    /// Validates and converts the packed witness before transcript mutation.
    pub(crate) fn new(
        pcs: &'a Pcs,
        data: &'a ProverData,
        packed_witness: Vec<F128>,
    ) -> Result<Self, CommitError> {
        if packed_witness.len() != pcs.packed_len() {
            return Err(CommitError::InvalidBitLength);
        }
        validate_prover_data(pcs, data)?;
        Ok(Self {
            pcs,
            data,
            packed_witness: into_flock_f128s(packed_witness),
        })
    }

    /// Returns the packed witness for the query-specific ring switch.
    pub(crate) fn witness(&self) -> &[FlockF128] {
        &self.packed_witness
    }

    /// Proves one reduced claim and writes the completed opening proof.
    pub(crate) fn prove(
        self,
        claim: ReducedClaim,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError> {
        let ReducedClaim {
            packed_basis,
            packed_target,
        } = claim;
        observe_opening_target(transcript, self.pcs.opening_log_n(), packed_target);
        let mut challenger = ProverChallenger::new_ligerito(transcript, packed_target);
        let flock_data = self.data.flock_data();
        let ligerito = recursive_prover_with_basis(
            self.pcs.prover_config(),
            self.packed_witness,
            packed_basis,
            packed_target,
            &flock_data.codeword,
            &flock_data.merkle_tree,
            &mut challenger,
        );
        if challenger.failed() {
            return Err(CommitError::invalid_configuration(
                "missing Ligerito opening-target prefix",
            ));
        }

        write_opening_proof(&ligerito, transcript)
    }
}

pub(crate) fn validate_verifier_config(
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

pub(crate) fn validate_prover_data(pcs: &Pcs, data: &ProverData) -> Result<(), CommitError> {
    let expected = pcs.params();
    let actual = &data.commitment().params;
    if expected.m != actual.m
        || expected.log_inv_rate != actual.log_inv_rate
        || expected.log_batch_size != actual.log_batch_size
        || expected.profile != actual.profile
        || expected.merkle_hash != actual.merkle_hash
    {
        return Err(CommitError::invalid_configuration(format!(
            "prover data parameters do not match the active PCS: expected {:?}, got {:?}",
            expected, actual,
        )));
    }
    Ok(())
}

pub(crate) fn read_proof(
    pcs: &Pcs,
    commitment: &Root,
    transcript: &mut VerifierState<'_>,
) -> Result<LigeritoProof, CommitError> {
    let proof = read_opening_proof(transcript)?;
    validate_ligerito_proof_shape(
        &proof,
        pcs.verifier_config(),
        pcs.final_log_n(),
        &commitment.0,
    )?;
    Ok(proof)
}

pub(crate) fn verify_dense(
    pcs: &Pcs,
    commitment: &Root,
    proof: &LigeritoProof,
    claim: ReducedClaim,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    let ReducedClaim {
        packed_basis,
        packed_target,
    } = claim;
    finish_verification(
        pcs.opening_log_n(),
        packed_target,
        transcript,
        |challenger| {
            recursive_verifier_with_basis(
                pcs.verifier_config(),
                proof,
                &packed_basis,
                packed_target,
                &commitment.0,
                challenger,
            )
        },
    )
}

pub(crate) fn verify_succinct<F>(
    pcs: &Pcs,
    commitment: &Root,
    proof: &LigeritoProof,
    log_n: usize,
    packed_target: FlockF128,
    evaluate_basis: F,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError>
where
    F: Fn(&[FlockF128], usize) -> Vec<FlockF128>,
{
    finish_verification(
        pcs.opening_log_n(),
        packed_target,
        transcript,
        |challenger| {
            recursive_verifier_with_basis_succinct(
                pcs.verifier_config(),
                proof,
                log_n,
                packed_target,
                &commitment.0,
                evaluate_basis,
                challenger,
            )
        },
    )
}

fn finish_verification<'proof>(
    opening_log_n: u32,
    packed_target: FlockF128,
    transcript: &mut VerifierState<'proof>,
    verify: impl FnOnce(&mut VerifierChallenger<'_, 'proof>) -> bool,
) -> Result<(), CommitError> {
    observe_opening_target(transcript, opening_log_n, packed_target);
    let mut challenger = VerifierChallenger::new_ligerito(transcript, packed_target);
    let valid = verify(&mut challenger);
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }
    if !valid {
        return Err(CommitError::VerificationFailed);
    }
    Ok(())
}

fn validate_pcs_verifier_prover(
    params: &PcsParams,
    prover: &ProverConfig,
    verifier: &VerifierConfig,
) -> Result<(), CommitError> {
    let shared_fields_match = prover.log_inv_rates == verifier.log_inv_rates
        && prover.recursive_steps == verifier.recursive_steps
        && prover.initial_log_msg_cols == verifier.initial_log_msg_cols
        && prover.initial_log_num_interleaved == verifier.initial_log_num_interleaved
        && prover.initial_k == verifier.initial_k
        && prover.recursive_log_msg_cols == verifier.recursive_log_msg_cols
        && prover.recursive_ks == verifier.recursive_ks
        && prover.queries == verifier.queries
        && prover.grinding_bits == verifier.grinding_bits
        && prover.fold_grinding_bits == verifier.fold_grinding_bits
        && prover.ood_samples == verifier.ood_samples
        && prover.merkle_hash == verifier.merkle_hash;
    if !shared_fields_match {
        return Err(CommitError::invalid_configuration(
            "Ligerito prover and verifier configurations do not match",
        ));
    }
    if verifier.log_inv_rates.first().copied() != Some(params.log_inv_rate) {
        return Err(CommitError::invalid_configuration(format!(
            "initial Ligerito rate does not match PCS rate {}",
            params.log_inv_rate,
        )));
    }
    if verifier.merkle_hash != params.merkle_hash {
        return Err(CommitError::invalid_configuration(
            "Ligerito Merkle hash does not match the PCS Merkle hash",
        ));
    }
    Ok(())
}

fn checked_pow2(log: usize) -> Option<usize> {
    u32::try_from(log)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
}

fn valid_query_shape(log_columns: usize, log_rate: usize, queries: usize) -> bool {
    queries > 0
        && log_columns
            .checked_add(log_rate)
            .and_then(checked_pow2)
            .is_some_and(|block_len| queries <= block_len)
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

fn rows_match(rows: &[Vec<FlockF128>], expected_rows: usize, expected_width: usize) -> bool {
    rows.len() == expected_rows && rows.iter().all(|row| row.len() == expected_width)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommitScheme, HashKind, LigeritoProfile, OpeningQuery, StatementBinding};
    use common::Shape;
    use flock_core::pcs::LOG_PACKING;
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
        assert!(validate_verifier_config(&config, log_n, initial_k).is_ok());
    }

    #[test]
    fn config_validation_rejects_invalid_shapes_and_values() {
        let (valid, log_n, initial_k) = registered_config();

        let mut config = valid.clone();
        config.log_inv_rates.pop();
        let expected_levels = config.recursive_steps + 1;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(format!(
                "log_inv_rates length {} does not match level count {expected_levels}",
                config.log_inv_rates.len(),
            )))
        );

        let mut config = valid.clone();
        config.log_inv_rates[0] = 0;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "log_inv_rates[0] must be positive"
            ))
        );

        let mut config = valid.clone();
        config.queries[0] = 0;
        let initial_log_columns = log_n - config.initial_k;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(format!(
                "initial query shape is invalid: log_columns={initial_log_columns}, log_inv_rate={}, queries=0",
                config.log_inv_rates[0],
            )))
        );

        let mut config = valid.clone();
        config.recursive_ks[0] = 0;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(CommitError::invalid_configuration(
                "recursive_ks[0] must be positive"
            ))
        );

        let mut config = valid;
        config.queries[0] = usize::MAX;
        let initial_log_columns = log_n - config.initial_k;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
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
        let packed_witness = vec![F128::default(); pcs.packed_len()];
        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let query = OpeningQuery::Mle {
            point: vec![F128::from(2u64); 22],
            target: F128::default(),
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
        assert_eq!(
            validate_ligerito_proof_shape(
                &valid,
                pcs.verifier_config(),
                pcs.final_log_n(),
                &commitment.0,
            ),
            Ok(())
        );

        type ProofMutation = fn(&mut LigeritoProof);
        let mutations: [(&str, ProofMutation); 2] = [
            ("recursive-root count", |proof| {
                proof.recursive_roots.pop();
            }),
            ("opened-row width", |proof| {
                proof.initial_proof.opened_rows[0].pop();
            }),
        ];

        for (case, mutate) in mutations {
            let mut proof = valid.clone();
            mutate(&mut proof);
            assert_eq!(
                validate_ligerito_proof_shape(
                    &proof,
                    pcs.verifier_config(),
                    pcs.final_log_n(),
                    &commitment.0,
                ),
                Err(CommitError::VerificationFailed),
                "{case}",
            );
        }
    }
}
