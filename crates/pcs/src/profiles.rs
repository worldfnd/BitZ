//! Derives both PCS security profiles from the padded witness size.

use flock_core::pcs::LOG_PACKING;
use flock_core::pcs::ligerito::{
    FinalBlockConfig, GrindingStep, LigeritoLevelConfig, LigeritoSecurityConfig, SoundnessRegime,
};

use crate::{ConfigError, SecurityLevel};

mod bounds;

const INITIAL_K: usize = 4;
const RECURSIVE_K: usize = 3;
const FINAL_LOG_N: usize = 5;
const JOHNSON_ETA: f64 = 0.02;

/// Builds Johnson100 with OOD, or UDR128 without OOD.
/// Returns the Flock parameters and any initial OOD grinding requirement.
pub(crate) fn security_config(
    m: usize,
    security_level: SecurityLevel,
) -> Result<(LigeritoSecurityConfig, Option<u32>), ConfigError> {
    if !(20..=35).contains(&m) {
        return Err(ConfigError::Invalid("unsupported PCS size"));
    }
    let target = security_level.bits() as usize;
    let johnson = security_level == SecurityLevel::Bits100;
    let log_n = m - LOG_PACKING;
    let mut remaining = log_n;
    let mut levels = Vec::new();
    let mut initial_ood = None;
    while remaining > FINAL_LOG_N {
        let first = levels.is_empty();
        let k = if first {
            INITIAL_K
        } else {
            RECURSIVE_K.min(remaining - FINAL_LOG_N)
        };
        remaining -= k;
        let mut level = LigeritoLevelConfig {
            log_inv_rate: levels.len() + 1,
            log_msg_cols: remaining,
            log_num_interleaved: k,
            k_recursive: k,
            regime: if johnson {
                SoundnessRegime::JohnsonOod
            } else {
                SoundnessRegime::Udr
            },
            eta: johnson.then_some(JOHNSON_ETA),
            proximity_loss: (!johnson).then_some(0.0),
            queries: 0,
            grinding_bits: 0,
            fold_grinding_bits: 0,
            ood_samples: usize::from(johnson && !first),
            target_security_bits: target,
            expected_eps_pg_bits: 0.0,
            expected_eps_query_bits: 0.0,
            expected_eps_ood_bits: None,
        };
        let ood = bounds::configure_level(&mut level, first, remaining == FINAL_LOG_N)?;
        if first {
            initial_ood = ood;
        }
        levels.push(level);
    }
    let security = LigeritoSecurityConfig {
        m,
        log_n,
        initial_k: INITIAL_K,
        target_security_bits: target,
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
    };
    Ok((security, initial_ood))
}

#[cfg(test)]
mod tests;
