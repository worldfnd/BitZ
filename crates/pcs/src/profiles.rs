//! Calculates PCS parameters from the padded witness size and security target.
//!
//! Input `m` is log2(padded witness bits). Packing gives `2^(m - 7)` field elements.
//! We use GF(2^128), 128 witness bits per element, and BLAKE3 for Merkle commitments and PoW.
//!
//! Each level groups binary folds. The first level uses four folds; later levels use up to three.
//! Each fold halves the message size. We stop at five variables: 32 field elements.
//! Successive levels use code rates 1/2, 1/4, 1/8, and so on.
//!
//! The 100-bit profile uses Johnson list decoding, OOD checks, and no query grinding.
//! The 128-bit profile uses unique decoding, no OOD checks, and query grinding.
//! Flock supplies folding, query, and multilinear OOD estimates. We add our combined error costs.
//!
//! These targets concern classical challenge blocks. They assume uniform transcript challenges and classical PoW costs.
//! Sources: BitZ Remark A.2; Flock Appendix C.3; BCHKS25 Corollary 1.4.

use flock_core::pcs::LOG_PACKING;
use flock_core::pcs::ligerito::{
    FinalBlockConfig, GrindingStep, LigeritoLevelConfig, LigeritoSecurityConfig, SoundnessRegime,
};

use crate::{ConfigError, SecurityLevel};

const INITIAL_K: usize = 4;
const RECURSIVE_K: usize = 3;
const FINAL_LOG_N: usize = 5;
const JOHNSON_ETA: f64 = 0.02;
const MAX_GRINDING_BITS: usize = transcript::pow::MAX_GRINDING_BITS as usize;

/// Derives every level from `m` and the selected target.
///
/// Per level: choose the shape, ask Flock for base estimates, then select queries and grinding.
/// We use one query initially to measure its contribution. We store only the completed configuration.
pub(crate) fn security_config(
    m: usize,
    security_level: SecurityLevel,
) -> Result<LigeritoSecurityConfig, ConfigError> {
    if !(20..=35).contains(&m) {
        return Err(ConfigError::Invalid("unsupported PCS size"));
    }
    let johnson = security_level == SecurityLevel::Bits100;
    let log_n = m - LOG_PACKING;
    let mut remaining = log_n;
    let mut levels = Vec::new();

    while remaining > FINAL_LOG_N {
        // 1. Choose folds. Leave five variables for the explicit final message.
        let first = levels.is_empty();
        let folds = if first {
            INITIAL_K
        } else {
            RECURSIVE_K.min(remaining - FINAL_LOG_N)
        };
        remaining -= folds;

        // 2. Choose the code rate: message columns / encoded columns.
        // log_inv_rate = log2(encoded columns / message columns).
        // levels.len() counts completed levels: 0, 1, 2, ...
        // Adding 1 selects our fixed expansion schedule: 2x, 4x, 8x, ...
        // The corresponding code rates are 1/2, 1/4, 1/8, ...
        // More redundancy improves query detection bounds, so later levels need fewer queries.
        let log_inv_rate = levels.len() + 1;

        // 3. Give Flock the shape and decoding policy. One query measures the contribution per query.
        let mut level = LigeritoLevelConfig {
            log_inv_rate,
            log_msg_cols: remaining,
            log_num_interleaved: folds,
            k_recursive: folds,
            regime: if johnson {
                SoundnessRegime::JohnsonOod
            } else {
                SoundnessRegime::Udr
            },
            eta: johnson.then_some(JOHNSON_ETA),
            // Zero is the fixed backend policy for unique decoding.
            proximity_loss: (!johnson).then_some(0.0),
            queries: 1,
            // Starting values. The selected profile sets the final query count and grinding below.
            grinding_bits: 0,
            fold_grinding_bits: 0,
            // Later Johnson levels use one OOD sample. The commitment performs the initial check.
            ood_samples: usize::from(johnson && !first),
            target_security_bits: security_level.bits() as usize,
            expected_eps_pg_bits: 0.0,
            expected_eps_query_bits: 0.0,
            expected_eps_ood_bits: None,
        };

        // 4. Select parameters using Flock's estimates and our combined error costs.
        match security_level {
            SecurityLevel::Bits100 => configure_100(&mut level, first, remaining == FINAL_LOG_N)?,
            SecurityLevel::Bits128 => configure_128(&mut level, first, remaining == FINAL_LOG_N)?,
        }
        if level.queries > 1usize << (remaining + log_inv_rate) {
            return Err(ConfigError::Invalid("queries exceed codeword length"));
        }

        // 5. Refresh backend diagnostics with the final query count.
        let (fold_bits, query_bits) = level.paper_predicted_bits();
        level.expected_eps_pg_bits = fold_bits;
        level.expected_eps_query_bits = query_bits;
        level.expected_eps_ood_bits = level.paper_predicted_ood_bits();
        levels.push(level);
    }

    Ok(LigeritoSecurityConfig {
        m,
        log_n,
        initial_k: INITIAL_K,
        target_security_bits: security_level.bits() as usize,
        analysis_version: if johnson {
            "bitz_johnson_combined_blocks_v2"
        } else {
            "bitz_udr_combined_blocks_v1"
        }
        .into(),
        field: "f128".into(),
        hash: "blake3".into(),
        grinding_step: GrindingStep::PostCommitPreQueries,
        levels,
        final_block: FinalBlockConfig {
            yr_log_n: remaining,
        },
    })
}

/// Selects the 100-bit parameters with Johnson list decoding.
fn configure_100(
    level: &mut LigeritoLevelConfig,
    first: bool,
    final_level: bool,
) -> Result<(), ConfigError> {
    // Error coefficient C represents probability C / 2^128. The 100-bit target permits C <= 2^28.
    let allowed_coefficient = 2f64.powi(28);

    // 1. Reuse Flock's folding and per-query estimates. The fold estimate already includes the row multiplier.
    let (fold_bits, per_query_bits) = level.paper_predicted_bits();
    let fold_coefficient = (128.0 - fold_bits).exp2();
    let rate = 2f64.powi(-(level.log_inv_rate as i32));
    let list_size = 1.0 / (2.0 * JOHNSON_ETA * rate.sqrt());

    // 2. Choose the smallest fold grinding that covers every round and its sumcheck.
    // Johnson halves the fold coefficient and decreases grinding by one bit after each round.
    level.fold_grinding_bits = (0..=MAX_GRINDING_BITS)
        .find(|&bits| {
            (0..level.k_recursive).all(|round| {
                let grinding = bits.saturating_sub(round);
                let claim_batching = f64::from(!first && round == 0 && grinding == 0);
                let total_coefficient = fold_coefficient * 2f64.powi(-(round as i32))
                    + (2.0 + claim_batching) * list_size;
                total_coefficient <= 2f64.powi(28 + grinding as i32)
            })
        })
        .ok_or(ConfigError::Invalid("Johnson fold grinding exceeds cap"))?;

    // 3. Check OOD selection and batching. Their challenges precede fold grinding.
    // Independent coordinates give degree at most mu. Pair collisions cost at most L^2 * mu / (2 * |F|).
    // The commitment performs the initial check. Its backend sample count stays zero.
    let explicit_ood = LigeritoLevelConfig {
        ood_samples: 1,
        ..level.clone()
    };
    if explicit_ood.paper_predicted_ood_bits().unwrap() < 100.0 {
        return Err(ConfigError::Invalid("Johnson OOD bound"));
    }
    let batching_challenges = if first { 8.0 } else { 1.0 };
    if batching_challenges * list_size > allowed_coefficient {
        return Err(ConfigError::Invalid("Johnson unground batching bound"));
    }

    // 4. Add queries until query misses and batching together meet 2^-100, without query grinding.
    // The next commitment halves the rate, increasing its list bound by sqrt(2).
    // The explicit final message has only one candidate.
    let next_list_size = if final_level {
        1.0
    } else {
        list_size * 2f64.sqrt()
    };
    level.queries = (100.0 / per_query_bits).ceil() as usize;
    while query_and_batching_error(
        (-per_query_bits * level.queries as f64).exp2(),
        level.queries,
        next_list_size,
        true,
    ) > 2f64.powi(-100)
    {
        level.queries += 1;
    }
    level.grinding_bits = 0;
    Ok(())
}

/// Selects the 128-bit parameters with unique decoding and no OOD checks.
fn configure_128(
    level: &mut LigeritoLevelConfig,
    first: bool,
    final_level: bool,
) -> Result<(), ConfigError> {
    // 1. Check the theorem range, then reuse Flock's folding and per-query estimates.
    let length = 1u128 << (level.log_msg_cols + level.log_inv_rate);
    let inverse_rate = 1u128 << level.log_inv_rate;
    let scaled_distance_squared = (inverse_rate - 1).pow(2) * length;
    // BCHKS25 Corollary 1.4 requires distance^2 * n >= 18.
    if scaled_distance_squared < 18 * inverse_rate.pow(2) {
        return Err(ConfigError::Invalid("UDR theorem range"));
    }
    let (fold_bits, per_query_bits) = level.paper_predicted_bits();
    let fold_coefficient = (128.0 - fold_bits).exp2();

    // 2. Choose constant fold grinding. Add the sumcheck's coefficient of two to Flock's fold coefficient.
    // An unground first recursive fold also shares the preceding claim-batching challenge.
    level.fold_grinding_bits = (0..=MAX_GRINDING_BITS)
        .find(|&bits| {
            let claim_batching = f64::from(!first && bits == 0);
            fold_coefficient + 2.0 + claim_batching <= 2f64.powi(bits as i32)
        })
        .ok_or(ConfigError::Invalid("UDR fold grinding exceeds cap"))?;

    // 3. Reuse Flock's per-query estimate. Queries alone must meet 2^-128.
    level.queries = (128.0 / per_query_bits).ceil() as usize;
    let all_queries_miss = (-per_query_bits * level.queries as f64).exp2();

    // 4. Choose query grinding to cover query misses and batching together.
    // Unique decoding has one candidate. Only the final level includes the extra claim-batching challenge.
    let combined_error =
        query_and_batching_error(all_queries_miss, level.queries, 1.0, final_level);
    level.grinding_bits = (0..=MAX_GRINDING_BITS)
        .find(|&bits| combined_error <= 2f64.powi(bits as i32 - 128))
        .ok_or(ConfigError::Invalid("UDR query grinding exceeds cap"))?;
    Ok(())
}

/// Adds query misses and batching error. Each batching challenge costs list_size / 2^128.
fn query_and_batching_error(
    all_queries_miss: f64,
    queries: usize,
    list_size: f64,
    include_claim_batching: bool,
) -> f64 {
    let batching_challenges =
        queries.next_power_of_two().ilog2() + u32::from(include_claim_batching);
    all_queries_miss + f64::from(batching_challenges) * list_size * 2f64.powi(-128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_batching_covers_every_candidate() {
        // Eight query-batching challenges and one claim-batching challenge each cover 64 candidates.
        let error = query_and_batching_error(0.0, 256, 64.0, true);
        assert!(error >= 576.0 * 2f64.powi(-128));
        assert!(error < 577.0 * 2f64.powi(-128));
    }

    #[test]
    fn udr_m22_matches_audited_parameters() {
        let config = security_config(22, SecurityLevel::Bits128).unwrap();
        assert_eq!(config.initial_k, 4);
        assert_eq!(config.final_block.yr_log_n, 5);
        assert_eq!(config.hash, "blake3");
        assert_eq!(config.target_security_bits, 128);
        for (index, level) in config.levels.iter().enumerate() {
            assert_eq!(level.queries, [311, 192, 161][index]);
            assert_eq!(level.fold_grinding_bits, [10, 9, 7][index]);
            assert_eq!(level.grinding_bits, 4);
            assert_eq!(level.ood_samples, 0);
        }
    }

    #[test]
    fn udr_all_supported_sizes_cover_combined_errors() {
        for m in 20..=35 {
            let config = security_config(m, SecurityLevel::Bits128).unwrap();
            assert_eq!(config.hash, "blake3");
            config.to_prover_verifier_configs().unwrap();
            for (index, level) in config.levels.iter().enumerate() {
                // Reconstruct the published formulas independently of the bound helpers.
                let n = 2f64.powi((level.log_msg_cols + level.log_inv_rate) as i32);
                let rho = 2f64.powi(-(level.log_inv_rate as i32));
                let delta = 1.0 - rho;
                let gamma = delta / 2.0 - 3.0 / (delta * n);
                assert!(delta >= 3.0 * (2.0 / n).sqrt());
                assert!(gamma >= delta / 3.0 && gamma < delta / 2.0);
                let fold_error = |grinding| {
                    let beta = f64::from(index > 0 && grinding == 0);
                    (gamma * n + 3.0 + beta) * 2f64.powi(-128 - grinding as i32)
                };
                assert!(fold_error(level.fold_grinding_bits) <= 2f64.powi(-128));
                if level.fold_grinding_bits > 0 {
                    assert!(fold_error(level.fold_grinding_bits - 1) > 2f64.powi(-128));
                }
                let final_beta = usize::from(index + 1 == config.levels.len());
                let alpha = level.queries.next_power_of_two().ilog2() as usize;
                let miss = (1.0 - gamma).powi(level.queries as i32);
                assert!(miss <= 2f64.powi(-128));
                assert!((1.0 - gamma).powi(level.queries as i32 - 1) > 2f64.powi(-128));
                let query = (miss + (alpha + final_beta) as f64 * 2f64.powi(-128))
                    * 2f64.powi(-(level.grinding_bits as i32));
                assert!(query <= 2f64.powi(-128), "m={m}, level={index}");
                if level.grinding_bits > 0 {
                    assert!(query * 2.0 > 2f64.powi(-128));
                }
                assert!(level.queries <= n as usize);
                assert!(level.fold_grinding_bits <= 32 && level.grinding_bits <= 32);
                assert!(
                    level.fold_grinding_bits == 0 || level.fold_grinding_bits >= level.k_recursive,
                    "every positive fold schedule must retain all native grinding hooks"
                );
            }
        }
    }

    #[test]
    fn johnson_queries_replace_query_grinding() {
        for (m, folds) in [(20, [7, 4, 1]), (22, [9, 6, 3])] {
            let config = security_config(m, SecurityLevel::Bits100).unwrap();
            assert_eq!(config.target_security_bits, 100);
            for (index, level) in config.levels.iter().enumerate() {
                assert_eq!(level.queries, [218, 106, 71][index]);
                assert_eq!(level.fold_grinding_bits, folds[index]);
                assert_eq!(level.grinding_bits, 0);
                assert_eq!(level.ood_samples, usize::from(index > 0));
            }
        }
    }

    #[test]
    fn johnson_all_sizes_cover_combined_blocks() {
        for m in 20..=35 {
            let config = security_config(m, SecurityLevel::Bits100).unwrap();
            config.to_prover_verifier_configs().unwrap();
            for (index, level) in config.levels.iter().enumerate() {
                // Reconstruct the published formulas independently of the bound helpers.
                let rho = 2f64.powi(-(level.log_inv_rate as i32));
                let sqrt_rho = rho.sqrt();
                let list = 1.0 / (0.04 * sqrt_rho);
                let gamma = 1.0 - sqrt_rho - 0.02;
                let half = (sqrt_rho / 0.04).ceil().max(3.0) + 0.5;
                let n = 2f64.powi((level.log_msg_cols + level.log_inv_rate) as i32);
                let base = (2.0 * half.powi(5) + 3.0 * half * gamma * rho) / (3.0 * rho.powf(1.5))
                    * n
                    + half / sqrt_rho;
                let k = level.k_recursive;
                let folds_fit = |grinding: usize| {
                    (0..k).all(|round| {
                        let row_union = 2f64.powi((k - 1 - round) as i32);
                        let effective = grinding.saturating_sub(round);
                        let beta = f64::from(index > 0 && round == 0 && effective == 0);
                        base * row_union + (2.0 + beta) * list <= 2f64.powi(28 + effective as i32)
                    })
                };
                assert!(folds_fit(level.fold_grinding_bits));
                if level.fold_grinding_bits > 0 {
                    assert!(!folds_fit(level.fold_grinding_bits - 1));
                }
                let next_list = config.levels.get(index + 1).map_or(1.0, |next| {
                    1.0 / (0.04 * 2f64.powi(-(next.log_inv_rate as i32)).sqrt())
                });
                let alpha = level.queries.next_power_of_two().ilog2();
                let query = (sqrt_rho + 0.02).powi(level.queries as i32)
                    + f64::from(alpha + 1) * next_list * 2f64.powi(-128);
                assert!(query <= 2f64.powi(-100));
                let previous_query = (sqrt_rho + 0.02).powi(level.queries as i32 - 1)
                    + f64::from((level.queries - 1).next_power_of_two().ilog2() + 1)
                        * next_list
                        * 2f64.powi(-128);
                assert!(previous_query > 2f64.powi(-100));
                let mu = (level.log_msg_cols + level.log_num_interleaved) as f64;
                // Initial and recursive OOD points use independent coordinates, with total degree at most mu.
                let ood = list * list * mu * 0.5;
                assert!(ood <= 2f64.powi(28));
                assert!((if index == 0 { 8.0 } else { 1.0 }) * list <= 2f64.powi(28));
                assert!(level.queries <= n as usize);
                assert_eq!(level.grinding_bits, 0);
                assert_eq!(level.ood_samples, usize::from(index > 0));
                assert!(level.fold_grinding_bits <= 32);
                assert!(level.fold_grinding_bits >= k);
            }
        }
    }

    #[test]
    fn both_profiles_keep_a_five_variable_residual() {
        for security in [SecurityLevel::Bits100, SecurityLevel::Bits128] {
            for m in 20..=35 {
                let config = security_config(m, security).unwrap();
                assert_eq!(config.levels[0].k_recursive, 4);
                assert_eq!(config.final_block.yr_log_n, 5);
                let total_folds: usize = config.levels.iter().map(|level| level.k_recursive).sum();
                assert_eq!(total_folds + 5, m - 7);
                for (index, level) in config.levels.iter().enumerate() {
                    assert_eq!(level.log_inv_rate, index + 1);
                    assert_eq!(level.log_num_interleaved, level.k_recursive);
                }
                if m <= 21 {
                    assert_eq!(config.levels[1].k_recursive, 3);
                    assert_eq!(config.levels[2].k_recursive, m - 19);
                }
            }
        }
    }

    #[test]
    fn both_profiles_reject_unsupported_sizes() {
        for security in [SecurityLevel::Bits100, SecurityLevel::Bits128] {
            for m in [0, 19, 36, usize::MAX] {
                assert!(security_config(m, security).is_err());
            }
        }
    }
}
