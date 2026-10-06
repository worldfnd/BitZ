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
//! Both profiles calculate fold grinding from the level size.
//!
//! These targets concern classical challenge blocks. They assume uniform transcript challenges and classical PoW costs.
//! Sources: BitZ Remark A.2 and Lemma B.2; Flock Appendix C.3; BCHKS25 Corollary 1.4.

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
/// Per level: choose folds, set the shape, calculate queries and grinding, then build the backend configuration.
/// Only diagnostic fields start as placeholders. Backend methods fill them before we store the level.
/// The separate initial OOD result uses `None` for no check and `Some(0)` for a check without grinding.
pub(crate) fn security_config(
    m: usize,
    security_level: SecurityLevel,
) -> Result<(LigeritoSecurityConfig, Option<u32>), ConfigError> {
    if !(20..=35).contains(&m) {
        return Err(ConfigError::Invalid("unsupported PCS size"));
    }
    let johnson = security_level == SecurityLevel::Bits100;
    let log_n = m - LOG_PACKING;
    let mut remaining = log_n;
    let mut levels = Vec::new();
    let mut initial_ood = None;

    while remaining > FINAL_LOG_N {
        // 1. Choose folds. Leave five variables for the explicit final message.
        let first = levels.is_empty();
        let folds = if first {
            INITIAL_K
        } else {
            RECURSIVE_K.min(remaining - FINAL_LOG_N)
        };
        remaining -= folds;

        // 2. Set the shape: 2^folds rows, 2^remaining message columns, and rate 2^-log_inv_rate.
        let log_inv_rate = levels.len() + 1;

        // 3. Calculate all query and grinding parameters before creating the backend configuration.
        let parameters = match security_level {
            SecurityLevel::Bits100 => parameters_100(remaining, log_inv_rate, folds)?,
            SecurityLevel::Bits128 => parameters_128(remaining, log_inv_rate)?,
        };
        if first {
            initial_ood = parameters.initial_ood_grinding;
        }

        // 4. Combine the shape, fixed decoding policy, and calculated parameters.
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
            queries: parameters.queries,
            grinding_bits: parameters.query_grinding_bits,
            fold_grinding_bits: parameters.fold_grinding_bits,
            // Later Johnson levels use one OOD sample. The initial OOD check has separate configuration.
            ood_samples: usize::from(johnson && !first),
            target_security_bits: security_level.bits() as usize,
            expected_eps_pg_bits: 0.0,
            expected_eps_query_bits: 0.0,
            expected_eps_ood_bits: None,
        };

        // 5. Fill backend diagnostics. These estimates do not select the parameters.
        let (fold_bits, query_bits) = level.paper_predicted_bits();
        level.expected_eps_pg_bits = fold_bits;
        level.expected_eps_query_bits = query_bits;
        level.expected_eps_ood_bits = level.paper_predicted_ood_bits();
        levels.push(level);
    }

    Ok((
        LigeritoSecurityConfig {
            m,
            log_n,
            initial_k: INITIAL_K,
            target_security_bits: security_level.bits() as usize,
            analysis_version: if johnson {
                "bitz_johnson_combined_blocks_v1"
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
        },
        initial_ood,
    ))
}

/// Calculated values for one level. The caller supplies geometry and fixed policy separately.
struct LevelParameters {
    queries: usize,
    fold_grinding_bits: usize,
    query_grinding_bits: usize,
    initial_ood_grinding: Option<u32>,
}

/// Selects the 100-bit parameters with Johnson list decoding.
fn parameters_100(
    log_msg_cols: usize,
    log_inv_rate: usize,
    folds: usize,
) -> Result<LevelParameters, ConfigError> {
    let first = log_inv_rate == 1;
    let final_level = log_msg_cols == FINAL_LOG_N;
    let codeword_length = 1usize << (log_msg_cols + log_inv_rate);
    let variables = log_msg_cols + folds;
    // Error coefficient C represents probability C / 2^128. The 100-bit target permits C <= 2^28.
    let allowed_coefficient = 2f64.powi(28);

    // 1. Calculate the Johnson list size, query miss probability, and folding bound.
    let rate = 2f64.powi(-(log_inv_rate as i32));
    let (sqrt_rate_lower, sqrt_rate_upper) = sqrt_bounds(rate);
    let eta_lower = JOHNSON_ETA.next_down();
    let eta_upper = JOHNSON_ETA.next_up();
    let radius_upper = ((1.0 - sqrt_rate_lower).next_up() - eta_lower).next_up();
    let list_size = div_up(1.0, (2.0 * eta_lower * sqrt_rate_lower).next_down());
    let query_miss = add_up(sqrt_rate_upper, eta_upper);

    // Flock C.3: h = max(ceil(sqrt(rate)/(2*eta)), 3) + 1/2.
    // The fold coefficient is n*(2*h^5 + 3*h*radius*rate)/(3*rate^(3/2)) + h/sqrt(rate).
    let h = div_up(sqrt_rate_upper, 2.0 * eta_lower).ceil().max(3.0) + 0.5;
    let h_fifth = (0..5).fold(1.0, |power, _| mul_up(power, h));
    let numerator = add_up(2.0 * h_fifth, mul_up(mul_up(3.0 * h, radius_upper), rate));
    let denominator = (3.0 * (rate * sqrt_rate_lower).next_down()).next_down();
    let per_position = div_up(numerator, denominator);
    let fold_coefficient = add_up(
        mul_up(per_position, codeword_length as f64),
        div_up(h, sqrt_rate_lower),
    );
    // Retain the pinned backend's conservative multiplier for interleaved rows.
    let fold_coefficient = mul_up(fold_coefficient, 2f64.powi(folds as i32 - 1));

    // 2. Choose the smallest fold grinding that covers every round and its sumcheck.
    // Johnson halves the fold coefficient and decreases grinding by one bit after each round.
    let fold_grinding_bits = (0..=MAX_GRINDING_BITS)
        .find(|&bits| {
            (0..folds).all(|round| {
                let grinding = bits.saturating_sub(round);
                let claim_batching = f64::from(!first && round == 0 && grinding == 0);
                let total_coefficient = add_up(
                    fold_coefficient * 2f64.powi(-(round as i32)),
                    mul_up(2.0 + claim_batching, list_size),
                );
                total_coefficient <= 2f64.powi(28 + grinding as i32)
            })
        })
        .ok_or(ConfigError::Invalid("Johnson fold grinding exceeds cap"))?;

    // 3. Check OOD selection and batching. Their challenges precede fold grinding.
    let selection_coefficient = if first {
        mul_up(list_size, variables as f64)
    } else {
        mul_up(mul_up(list_size, list_size), variables as f64 * 0.5)
    };
    if selection_coefficient > allowed_coefficient {
        return Err(ConfigError::Invalid("Johnson recursive OOD bound"));
    }
    let batching_challenges = if first { 8.0 } else { 1.0 };
    if mul_up(batching_challenges, list_size) > allowed_coefficient {
        return Err(ConfigError::Invalid("Johnson unground batching bound"));
    }
    let initial_ood_grinding = if first {
        // BitZ Lemma B.2: each candidate pair can collide at at most degree points.
        let candidate_pairs = mul_up(list_size, (list_size - 1.0).next_up()) * 0.5;
        let degree = ((1u64 << variables) - 1) as f64;
        let collision_coefficient = mul_up(candidate_pairs, degree);
        let bits = (0..=MAX_GRINDING_BITS)
            .find(|&bits| collision_coefficient <= 2f64.powi(28 + bits as i32))
            .ok_or(ConfigError::Invalid(
                "Johnson initial OOD grinding exceeds cap",
            ))?;
        Some(bits as u32)
    } else {
        None
    };

    // 4. Add queries until query misses and batching together meet 2^-100, without query grinding.
    // The next commitment halves the rate, increasing its list bound by sqrt(2).
    // The explicit final message has only one candidate.
    let next_list_size = if final_level {
        1.0
    } else {
        mul_up(list_size, sqrt_bounds(2.0).1)
    };
    let mut queries = 0;
    let mut all_queries_miss = 1.0;
    while query_and_batching_error(all_queries_miss, queries, next_list_size, true)
        > 2f64.powi(-100)
    {
        if queries == codeword_length {
            return Err(ConfigError::Invalid("queries exceed codeword length"));
        }
        all_queries_miss = mul_up(all_queries_miss, query_miss);
        queries += 1;
    }
    Ok(LevelParameters {
        queries,
        fold_grinding_bits,
        query_grinding_bits: 0,
        initial_ood_grinding,
    })
}

/// Selects the 128-bit parameters with unique decoding and no OOD checks.
fn parameters_128(
    log_msg_cols: usize,
    log_inv_rate: usize,
) -> Result<LevelParameters, ConfigError> {
    let first = log_inv_rate == 1;
    let final_level = log_msg_cols == FINAL_LOG_N;
    let codeword_length = 1usize << (log_msg_cols + log_inv_rate);

    // 1. Calculate radius = distance/2 - 3/(distance*n), where distance = 1 - 1/inverse_rate.
    // Keep radius*n as an exact fraction for the fold check.
    let length = codeword_length as u128;
    let inverse_rate = 1u128 << log_inv_rate;
    let scaled_distance_squared = (inverse_rate - 1).pow(2) * length;
    // BCHKS25 Corollary 1.4 requires distance^2 * n >= 18.
    if scaled_distance_squared < 18 * inverse_rate.pow(2) {
        return Err(ConfigError::Invalid("UDR theorem range"));
    }
    let denominator = 2 * inverse_rate * (inverse_rate - 1);
    let radius_times_length_numerator = scaled_distance_squared - 6 * inverse_rate.pow(2);

    // 2. Choose constant fold grinding. Each fold and its sumcheck cost (radius*n + 3) / 2^128.
    // An unground first recursive fold also shares the preceding claim-batching challenge.
    let fold_grinding_bits = (0..=MAX_GRINDING_BITS)
        .find(|&bits| {
            let claim_batching = u128::from(!first && bits == 0);
            let error_numerator =
                radius_times_length_numerator + (3 + claim_batching) * denominator;
            error_numerator <= denominator * (1u128 << bits)
        })
        .ok_or(ConfigError::Invalid("UDR fold grinding exceeds cap"))?;

    // 3. Add queries until their miss probability alone meets 2^-128.
    let miss_numerator = denominator * length - radius_times_length_numerator;
    let query_miss = div_up(
        (miss_numerator as f64).next_up(),
        ((denominator * length) as f64).next_down(),
    );
    let mut queries = 0;
    let mut all_queries_miss = 1.0;
    while all_queries_miss > 2f64.powi(-128) {
        if queries == codeword_length {
            return Err(ConfigError::Invalid("queries exceed codeword length"));
        }
        all_queries_miss = mul_up(all_queries_miss, query_miss);
        queries += 1;
    }

    // 4. Choose query grinding to cover query misses and batching together.
    // Unique decoding has one candidate. Only the final level includes the extra claim-batching challenge.
    let combined_error = query_and_batching_error(all_queries_miss, queries, 1.0, final_level);
    let query_grinding_bits = (0..=MAX_GRINDING_BITS)
        .find(|&bits| combined_error <= 2f64.powi(bits as i32 - 128))
        .ok_or(ConfigError::Invalid("UDR query grinding exceeds cap"))?;
    Ok(LevelParameters {
        queries,
        fold_grinding_bits,
        query_grinding_bits,
        initial_ood_grinding: None,
    })
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
    let batching_error = mul_up(f64::from(batching_challenges), list_size) * 2f64.powi(-128);
    add_up(all_queries_miss, batching_error)
}

// Round error bounds upward and divisors downward. Rounding must never weaken a bound.
fn add_up(left: f64, right: f64) -> f64 {
    (left + right).next_up()
}

fn mul_up(left: f64, right: f64) -> f64 {
    (left * right).next_up()
}

fn div_up(numerator: f64, denominator: f64) -> f64 {
    (numerator / denominator).next_up()
}

fn sqrt_bounds(value: f64) -> (f64, f64) {
    // Check both endpoints independently of the platform's sqrt rounding.
    let root = value.sqrt();
    let mut lower = root.next_down();
    while mul_up(lower, lower) > value {
        lower = lower.next_down();
    }
    let mut upper = root.next_up();
    while (upper * upper).next_down() < value {
        upper = upper.next_up();
    }
    (lower, upper)
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
        let (config, initial_ood) = security_config(22, SecurityLevel::Bits128).unwrap();
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
        assert_eq!(initial_ood, None);
    }

    #[test]
    fn udr_all_supported_sizes_cover_combined_errors() {
        for m in 20..=35 {
            let (config, initial_ood) = security_config(m, SecurityLevel::Bits128).unwrap();
            assert_eq!(initial_ood, None);
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
            let (config, initial_ood) = security_config(m, SecurityLevel::Bits100).unwrap();
            assert_eq!(config.target_security_bits, 100);
            assert_eq!(initial_ood, Some(0));
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
            let (config, initial_ood) = security_config(m, SecurityLevel::Bits100).unwrap();
            config.to_prover_verifier_configs().unwrap();
            let initial_grinding = initial_ood.unwrap();
            let list = 1.0 / (0.04 * 0.5f64.sqrt());
            let initial = list * (list - 1.0) * 0.5 * (2f64.powi(m as i32 - 7) - 1.0);
            assert!(initial <= 2f64.powi(28 + initial_grinding as i32));
            assert!(initial_grinding <= 32);
            if initial_grinding > 0 {
                assert!(initial > 2f64.powi(27 + initial_grinding as i32));
            }
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
                let ood = if index == 0 {
                    list * mu
                } else {
                    list * list * mu * 0.5
                };
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
                let (config, _) = security_config(m, security).unwrap();
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
