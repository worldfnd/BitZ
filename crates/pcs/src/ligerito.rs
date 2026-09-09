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
use crate::utils::{read_opening_proof, write_opening_proof};
use crate::{ConfigError, Pcs, ProveError, ProverData, Root, VerifyError};

#[derive(Clone, Debug)]
pub(crate) struct CheckedLigerito {
    prover_config: ProverConfig,
    verifier_config: VerifierConfig,
    log_n_u32: u32,
    final_log_n: usize,
}

impl CheckedLigerito {
    pub(crate) fn new(params: &PcsParams) -> Result<Self, ConfigError> {
        let log_n = params
            .m
            .checked_sub(LOG_PACKING)
            .ok_or(ConfigError::Invalid("m below packing width"))?;
        let log_n_u32 =
            u32::try_from(log_n).map_err(|_| ConfigError::Invalid("log_n exceeds u32"))?;
        let prover_config = params
            .ligerito_prover_config()
            .map_err(|_| ConfigError::Invalid("prover config"))?;
        let verifier_config = params
            .ligerito_verifier_config()
            .map_err(|_| ConfigError::Invalid("verifier config"))?;
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
    ) -> Result<Self, ProveError> {
        if packed_witness.len() != pcs.packed_len() {
            return Err(ProveError::PackedWitnessLengthMismatch);
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
    ) -> Result<(), ProveError> {
        let ReducedClaim {
            packed_basis,
            packed_target,
        } = claim;
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
            return Err(ProveError::Internal);
        }

        write_opening_proof(&ligerito, transcript)
    }
}

pub(crate) fn validate_verifier_config(
    config: &VerifierConfig,
    log_n: usize,
    expected_initial_k: usize,
) -> Result<usize, ConfigError> {
    let r = config.recursive_steps;
    let level_count = r
        .checked_add(1)
        .ok_or(ConfigError::Invalid("recursive_steps overflow"))?;
    if r == 0 {
        return Err(ConfigError::Invalid("recursive_steps is zero"));
    }
    if config.recursive_ks.len() != r {
        return Err(ConfigError::Invalid("recursive_ks length"));
    }
    if config.recursive_log_msg_cols.len() != r {
        return Err(ConfigError::Invalid("recursive_log_msg_cols length"));
    }
    if config.log_inv_rates.len() != level_count {
        return Err(ConfigError::Invalid("log_inv_rates length"));
    }
    if config.queries.len() != level_count {
        return Err(ConfigError::Invalid("queries length"));
    }
    if config.grinding_bits.len() != level_count {
        return Err(ConfigError::Invalid("grinding_bits length"));
    }
    if config.fold_grinding_bits.len() != level_count {
        return Err(ConfigError::Invalid("fold_grinding_bits length"));
    }
    if config.ood_samples.len() != level_count {
        return Err(ConfigError::Invalid("ood_samples length"));
    }
    if config.initial_k != expected_initial_k {
        return Err(ConfigError::Invalid("initial_k mismatch"));
    }
    if config.initial_log_num_interleaved != config.initial_k {
        return Err(ConfigError::Invalid("initial_log_num_interleaved mismatch"));
    }
    if config.ood_samples[0] != 0 {
        return Err(ConfigError::Invalid("ood_samples[0] is nonzero"));
    }
    if config.log_inv_rates.contains(&0) {
        return Err(ConfigError::Invalid("log_inv_rates contains zero"));
    }
    if config
        .grinding_bits
        .iter()
        .any(|&bits| u32::try_from(bits).is_err())
    {
        return Err(ConfigError::Invalid("grinding_bits exceeds u32"));
    }
    if config
        .fold_grinding_bits
        .iter()
        .any(|&bits| u32::try_from(bits).is_err())
    {
        return Err(ConfigError::Invalid("fold_grinding_bits exceeds u32"));
    }

    let mut remaining = log_n
        .checked_sub(config.initial_k)
        .ok_or(ConfigError::Invalid("initial_k exceeds log_n"))?;
    if config.initial_log_msg_cols != remaining {
        return Err(ConfigError::Invalid("initial_log_msg_cols mismatch"));
    }
    if checked_pow2(config.initial_k).is_none() {
        return Err(ConfigError::Invalid("initial_k exceeds platform width"));
    }
    if !valid_query_shape(remaining, config.log_inv_rates[0], config.queries[0]) {
        return Err(ConfigError::Invalid("queries[0]"));
    }

    for level in 0..r {
        let k = config.recursive_ks[level];
        if k == 0 {
            return Err(ConfigError::Invalid("recursive_ks contains zero"));
        }
        if checked_pow2(k).is_none() {
            return Err(ConfigError::Invalid("recursive_ks exceeds platform width"));
        }
        remaining = remaining
            .checked_sub(k)
            .ok_or(ConfigError::Invalid("recursive_ks exceed log_n"))?;
        if config.recursive_log_msg_cols[level] != remaining {
            return Err(ConfigError::Invalid("recursive_log_msg_cols mismatch"));
        }
        if !valid_query_shape(
            remaining,
            config.log_inv_rates[level + 1],
            config.queries[level + 1],
        ) {
            return Err(ConfigError::Invalid("queries"));
        }
    }

    if remaining > 32 {
        return Err(ConfigError::Invalid("final_log_n exceeds 32"));
    }
    if checked_pow2(remaining).is_none() {
        return Err(ConfigError::Invalid("final_log_n exceeds platform width"));
    }
    Ok(remaining)
}

pub(crate) fn validate_prover_data(pcs: &Pcs, data: &ProverData) -> Result<(), ProveError> {
    let expected = pcs.params();
    let actual = &data.commitment().params;
    if expected.m != actual.m
        || expected.log_inv_rate != actual.log_inv_rate
        || expected.log_batch_size != actual.log_batch_size
        || expected.profile != actual.profile
        || expected.merkle_hash != actual.merkle_hash
    {
        return Err(ProveError::ProverDataMismatch);
    }
    Ok(())
}

pub(crate) fn read_proof(
    pcs: &Pcs,
    commitment: &Root,
    transcript: &mut VerifierState<'_>,
) -> Result<LigeritoProof, VerifyError> {
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
) -> Result<(), VerifyError> {
    let ReducedClaim {
        packed_basis,
        packed_target,
    } = claim;
    finish_verification(packed_target, transcript, |challenger| {
        recursive_verifier_with_basis(
            pcs.verifier_config(),
            proof,
            &packed_basis,
            packed_target,
            &commitment.0,
            challenger,
        )
    })
}

pub(crate) fn verify_succinct<F>(
    pcs: &Pcs,
    commitment: &Root,
    proof: &LigeritoProof,
    log_n: usize,
    packed_target: FlockF128,
    evaluate_basis: F,
    transcript: &mut VerifierState<'_>,
) -> Result<(), VerifyError>
where
    F: Fn(&[FlockF128], usize) -> Vec<FlockF128>,
{
    finish_verification(packed_target, transcript, |challenger| {
        recursive_verifier_with_basis_succinct(
            pcs.verifier_config(),
            proof,
            log_n,
            packed_target,
            &commitment.0,
            evaluate_basis,
            challenger,
        )
    })
}

fn finish_verification<'proof>(
    packed_target: FlockF128,
    transcript: &mut VerifierState<'proof>,
    verify: impl FnOnce(&mut VerifierChallenger<'_, 'proof>) -> bool,
) -> Result<(), VerifyError> {
    let mut challenger = VerifierChallenger::new_ligerito(transcript, packed_target);
    let valid = verify(&mut challenger);
    if challenger.failed() {
        return Err(VerifyError::MalformedProof);
    }
    if !valid {
        return Err(VerifyError::VerificationFailed);
    }
    Ok(())
}

fn validate_pcs_verifier_prover(
    params: &PcsParams,
    prover: &ProverConfig,
    verifier: &VerifierConfig,
) -> Result<(), ConfigError> {
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
        return Err(ConfigError::Invalid("prover and verifier differ"));
    }
    if verifier.log_inv_rates.first().copied() != Some(params.log_inv_rate) {
        return Err(ConfigError::Invalid("log_inv_rate mismatch"));
    }
    if verifier.merkle_hash != params.merkle_hash {
        return Err(ConfigError::Invalid("merkle_hash mismatch"));
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
) -> Result<(), VerifyError> {
    if &lig.initial_root != expected_root {
        return Err(VerifyError::VerificationFailed);
    }

    let r = config.recursive_steps;
    if lig.recursive_roots.len() != r
        || lig.recursive_proofs.len() != r - 1
        || lig.grinding_nonces.len() != r + 1
    {
        return Err(VerifyError::VerificationFailed);
    }

    let expected_ood = config
        .ood_samples
        .iter()
        .skip(1)
        .try_fold(0usize, |sum, &count| sum.checked_add(count))
        .ok_or(VerifyError::Internal)?;
    let expected_fold_nonces = positive_fold_nonce_count(config)?;
    let initial_sumchecks = 1usize
        .checked_add(config.initial_k)
        .ok_or(VerifyError::Internal)?;
    let expected_sumchecks = config
        .recursive_ks
        .iter()
        .try_fold(initial_sumchecks, |sum, &k| sum.checked_add(k))
        .and_then(|sum| sum.checked_add(r))
        .and_then(|sum| sum.checked_add(expected_ood))
        .ok_or(VerifyError::Internal)?;
    if lig.ood_values.len() != expected_ood
        || lig.fold_grinding_nonces.len() != expected_fold_nonces
        || lig.sumcheck_transcript.len() != expected_sumchecks
    {
        return Err(VerifyError::VerificationFailed);
    }

    let initial_width = checked_pow2(config.initial_k).ok_or(VerifyError::Internal)?;
    if !rows_match(
        &lig.initial_proof.opened_rows,
        config.queries[0],
        initial_width,
    ) {
        return Err(VerifyError::VerificationFailed);
    }
    for (level, recursive) in lig.recursive_proofs.iter().enumerate() {
        let width = checked_pow2(config.recursive_ks[level]).ok_or(VerifyError::Internal)?;
        if !rows_match(&recursive.opened_rows, config.queries[level + 1], width) {
            return Err(VerifyError::VerificationFailed);
        }
    }

    let last_k = *config.recursive_ks.last().ok_or(VerifyError::Internal)?;
    let final_width = checked_pow2(last_k).ok_or(VerifyError::Internal)?;
    let final_yr_len = checked_pow2(final_log_n).ok_or(VerifyError::Internal)?;
    if !rows_match(&lig.final_proof.opened_rows, config.queries[r], final_width)
        || lig.final_proof.yr.len() != final_yr_len
    {
        return Err(VerifyError::VerificationFailed);
    }
    Ok(())
}

fn rows_match(rows: &[Vec<FlockF128>], expected_rows: usize, expected_width: usize) -> bool {
    rows.len() == expected_rows && rows.iter().all(|row| row.len() == expected_width)
}

fn positive_fold_nonce_count(config: &VerifierConfig) -> Result<usize, VerifyError> {
    let initial = config.initial_k.min(config.fold_grinding_bits[0]);
    config
        .recursive_ks
        .iter()
        .zip(config.fold_grinding_bits.iter().skip(1))
        .try_fold(initial, |sum, (&k, &bits)| sum.checked_add(k.min(bits)))
        .ok_or(VerifyError::Internal)
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
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(ConfigError::Invalid("log_inv_rates length"))
        );

        let mut config = valid.clone();
        config.log_inv_rates[0] = 0;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(ConfigError::Invalid("log_inv_rates contains zero"))
        );

        let mut config = valid.clone();
        config.queries[0] = 0;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(ConfigError::Invalid("queries[0]"))
        );

        let mut config = valid.clone();
        config.recursive_ks[0] = 0;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(ConfigError::Invalid("recursive_ks contains zero"))
        );

        let mut config = valid;
        config.queries[0] = usize::MAX;
        assert_eq!(
            validate_verifier_config(&config, log_n, initial_k),
            Err(ConfigError::Invalid("queries[0]"))
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
                Err(VerifyError::VerificationFailed),
                "{case}",
            );
        }
    }
}
