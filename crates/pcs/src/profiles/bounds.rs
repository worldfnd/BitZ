//! Conservative probability bounds and per-level parameter selection.
//!
//! BitZ Remark A.2 and Lemma B.2 give the list and initial OOD bounds.
//! Flock Appendix C.3 gives folding, sumcheck, batching, and query bounds.

use flock_core::pcs::ligerito::{LigeritoLevelConfig, SoundnessRegime};

use super::JOHNSON_ETA;
use crate::ConfigError;

const MAX_GRINDING_BITS: usize = transcript::pow::MAX_GRINDING_BITS as usize;
const JOHNSON_SLACK_BITS: i32 = 128 - 100;

/// Selects query and grinding parameters for the supplied canonical level.
/// Only the first Johnson level returns an initial OOD grinding requirement.
pub(super) fn configure_level(
    level: &mut LigeritoLevelConfig,
    first: bool,
    final_level: bool,
) -> Result<Option<u32>, ConfigError> {
    let target = level.target_security_bits;
    let johnson = matches!(level.regime, SoundnessRegime::JohnsonOod);
    let positions = 1usize << (level.log_msg_cols + level.log_inv_rate);
    let mut initial_ood = None;
    let (miss_upper, query_list) = match level.regime {
        SoundnessRegime::JohnsonOod => {
            let probability = JohnsonProbability::new(level, positions);
            level.fold_grinding_bits = (0..=MAX_GRINDING_BITS)
                .find(|&bits| probability.folds_meet_target(level.k_recursive, bits, first))
                .ok_or(ConfigError::Invalid("Johnson fold grinding exceeds cap"))?;
            let variables = level.log_msg_cols + level.log_num_interleaved;
            probability.check_ood(variables, first)?;
            if first {
                initial_ood = Some(probability.initial_ood_grinding(variables)?);
            }
            // Nonfinal batching concerns the next commitment, whose rate halves rho.
            // Its list bound grows by sqrt(2). The final residual has one candidate.
            let query_list = if final_level {
                1.0
            } else {
                mul_up(probability.list_upper, sqrt_bounds(2.0).1)
            };
            (probability.miss_upper, query_list)
        }
        SoundnessRegime::Udr => (configure_udr(level, positions, first)?, 1.0),
    };
    let target_error = 2f64.powi(-(target as i32));
    let mut query_miss = 1.0;
    // Queries cover the target. Johnson also covers alpha and OOD beta without grinding.
    loop {
        let error = if johnson {
            combined_query_error(query_miss, level.queries, true, query_list)
        } else {
            query_miss
        };
        if error <= target_error {
            break;
        }
        if level.queries == positions {
            return Err(ConfigError::Invalid("queries exceed codeword length"));
        }
        query_miss = mul_up(query_miss, miss_upper);
        level.queries += 1;
    }
    if !johnson {
        // UDR grinding covers the combined query, alpha, and final beta errors.
        let error = combined_query_error(query_miss, level.queries, final_level, query_list);
        level.grinding_bits = (0..=MAX_GRINDING_BITS)
            .find(|&bits| error <= 2f64.powi(bits as i32 - target as i32))
            .ok_or(ConfigError::Invalid("UDR query grinding exceeds cap"))?;
    }
    let (pg, query) = level.paper_predicted_bits();
    level.expected_eps_pg_bits = pg;
    level.expected_eps_query_bits = query;
    level.expected_eps_ood_bits = level.paper_predicted_ood_bits();
    Ok(initial_ood)
}

struct JohnsonProbability {
    fold_coefficient_upper: f64,
    miss_upper: f64,
    list_upper: f64,
}

impl JohnsonProbability {
    fn new(level: &LigeritoLevelConfig, positions: usize) -> Self {
        // All inputs have the checked canonical geometry and eta=0.02.
        let rho = 2f64.powi(-(level.log_inv_rate as i32));
        let (sqrt_lower, sqrt_upper) = sqrt_bounds(rho);
        let eta_lower = JOHNSON_ETA.next_down();
        let eta_upper = JOHNSON_ETA.next_up();
        let gamma_upper = ((1.0 - sqrt_lower).next_up() - eta_lower).next_up();
        let half = div_up(sqrt_upper, 2.0 * eta_lower).ceil().max(3.0) + 0.5;
        let half5 = (0..5).fold(1.0, |value, _| mul_up(value, half));
        let numerator = add_up(2.0 * half5, mul_up(mul_up(3.0 * half, gamma_upper), rho));
        let denominator = (3.0 * (rho * sqrt_lower).next_down()).next_down();
        let base = add_up(
            mul_up(div_up(numerator, denominator), positions as f64),
            div_up(half, sqrt_lower),
        );
        // Retain the pinned backend's row multiplier as a conservative bound.
        let row_union = 2f64.powi(level.log_num_interleaved as i32 - 1);
        Self {
            fold_coefficient_upper: mul_up(base, row_union),
            miss_upper: add_up(sqrt_upper, eta_upper),
            list_upper: div_up(1.0, (2.0 * eta_lower * sqrt_lower).next_down()),
        }
    }

    fn check_ood(&self, variables: usize, first: bool) -> Result<(), ConfigError> {
        // One sample covers each recursive selection. Its response separates selection from beta.
        // The next query block covers beta. Level zero uses its implicit OOD bound.
        let mu = variables as f64;
        let coefficient = if first {
            mul_up(self.list_upper, mu)
        } else {
            mul_up(mul_up(self.list_upper, self.list_upper), mu * 0.5)
        };
        if coefficient > 2f64.powi(JOHNSON_SLACK_BITS) {
            return Err(ConfigError::Invalid("Johnson recursive OOD bound"));
        }
        // Initial ring switching and OOD mixing cost at most 8L/F.
        // Later introduction beta costs L/F before any fold grinding starts.
        let batching = if first { 8.0 } else { 1.0 };
        if mul_up(batching, self.list_upper) > 2f64.powi(JOHNSON_SLACK_BITS) {
            return Err(ConfigError::Invalid("Johnson unground batching bound"));
        }
        Ok(())
    }

    fn initial_ood_grinding(&self, log_n: usize) -> Result<u32, ConfigError> {
        let pairs = mul_up(self.list_upper, (self.list_upper - 1.0).next_up()) * 0.5;
        let degree = ((1u64 << log_n) - 1) as f64;
        let coefficient = mul_up(pairs, degree);
        (0..=MAX_GRINDING_BITS)
            .find(|&bits| coefficient <= 2f64.powi(JOHNSON_SLACK_BITS + bits as i32))
            .map(|bits| bits as u32)
            .ok_or(ConfigError::Invalid(
                "Johnson initial OOD grinding exceeds cap",
            ))
    }

    fn folds_meet_target(&self, folds: usize, grinding: usize, first: bool) -> bool {
        (0..folds).all(|round| {
            let effective = grinding.saturating_sub(round);
            // Flock C.3 takes a list union for sumcheck and claim batching.
            let extra = mul_up(
                2.0 + f64::from(!first && round == 0 && effective == 0),
                self.list_upper,
            );
            let coefficient = add_up(
                self.fold_coefficient_upper * 2f64.powi(-(round as i32)),
                extra,
            );
            coefficient <= 2f64.powi(JOHNSON_SLACK_BITS + effective as i32)
        })
    }
}

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
    // The product checks certify the bracket independently of sqrt rounding.
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

/// Selects constant UDR fold grinding and returns the per-query miss bound.
fn configure_udr(
    level: &mut LigeritoLevelConfig,
    positions: usize,
    first: bool,
) -> Result<f64, ConfigError> {
    // Callers first check the canonical ladder. Its largest exponent is 25.
    let n = positions as u128;
    let d = 1u128 << level.log_inv_rate;
    let distance_square = (d - 1).pow(2) * n;
    // BCHKS25 Corollary 1.4: delta >= 3*sqrt(2/n).
    // This also ensures gamma >= delta/3 at the selected upper endpoint.
    if distance_square < 18 * d * d {
        return Err(ConfigError::Invalid("UDR theorem range"));
    }
    let denominator = 2 * d * (d - 1);
    let gamma_n_numerator = distance_square - 6 * d * d;
    level.fold_grinding_bits = (0..=MAX_GRINDING_BITS)
        .find(|&bits| {
            // The fold and its quadratic cost gamma*n + 3 field errors.
            // An unground first recursive fold also shares the preceding introduction beta.
            let beta = u128::from(!first && bits == 0);
            gamma_n_numerator + (3 + beta) * denominator
                <= denominator * (1u128 << (128 - level.target_security_bits + bits))
        })
        .ok_or(ConfigError::Invalid("UDR fold grinding exceeds cap"))?;
    let miss_numerator = denominator * n - gamma_n_numerator;
    // Outward conversion and division keep this probability an upper bound.
    Ok(((miss_numerator as f64).next_up() / ((denominator * n) as f64).next_down()).next_up())
}

fn combined_query_error(
    miss_upper: f64,
    queries: usize,
    include_beta: bool,
    list_upper: f64,
) -> f64 {
    let alpha = queries.next_power_of_two().ilog2();
    let field_terms = alpha + u32::from(include_beta);
    add_up(
        miss_upper,
        mul_up(f64::from(field_terms), list_upper) * 2f64.powi(-128),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_union_can_exhaust_the_remaining_fold_budget() {
        let probability = JohnsonProbability {
            fold_coefficient_upper: 2f64.powi(28) - 100.0,
            miss_upper: 0.5,
            list_upper: 40.0,
        };
        // Sumcheck costs 80 field errors. Introduction beta raises that cost to 120.
        assert!(probability.folds_meet_target(1, 0, true));
        assert!(!probability.folds_meet_target(1, 0, false));
    }

    #[test]
    fn query_batching_covers_every_candidate() {
        // Eight alpha coordinates and one beta each range over 64 candidates.
        let error = combined_query_error(0.0, 256, true, 64.0);
        assert!(error >= 576.0 * 2f64.powi(-128));
        assert!(error < 577.0 * 2f64.powi(-128));
    }
}
