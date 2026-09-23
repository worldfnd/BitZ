//! Security profiles for the commitment and its recursive opening.

use flock_core::hash::HashKind;
use flock_core::pcs::LOG_PACKING;
use flock_core::pcs::ligerito::{
    FinalBlockConfig, GrindingStep, LigeritoLevelConfig, LigeritoProfile, LigeritoSecurityConfig,
    SoundnessRegime, embedded_security_config,
};

use crate::ConfigError;

const FAST_INITIAL_K: usize = 4;
const FAST_RECURSIVE_K: usize = 3;
const FAST_MAX_FINAL_LOG_N: usize = 5;
const FAST_SECURITY_BITS: usize = 100;
const FAST_QUERY_GRINDING_BITS: usize = 16;
const JOHNSON_ETA: f64 = 0.02;

/// Uses 16-lane rows for Fast and keeps Flock's embedded Slim and Secure profiles.
/// Every level uses the commitment's Merkle hash.
/// `CheckedLigerito` validates the result when it constructs the prover and verifier configurations.
pub(crate) fn security_config(
    m: usize,
    profile: LigeritoProfile,
    merkle_hash: HashKind,
) -> Result<LigeritoSecurityConfig, ConfigError> {
    let mut security = match profile {
        LigeritoProfile::Fast => fast_security_config(m)?,
        LigeritoProfile::Slim | LigeritoProfile::Secure => {
            let toml = embedded_security_config(m, profile).ok_or(ConfigError::Invalid(
                "no security config for this size and profile",
            ))?;
            LigeritoSecurityConfig::from_toml_str(toml)
                .map_err(|_| ConfigError::Invalid("security config"))?
        }
    };
    security.hash = match merkle_hash {
        HashKind::Sha256 => "sha256",
        HashKind::Blake3 => "blake3",
    }
    .into();
    Ok(security)
}

/// Derives initial OOD grinding from the level-zero collision bound.
///
/// With `rho = 2^-log_inv_rate`, let `list_size = 1 / (2 * eta * sqrt(rho))`,
/// `pairs = max(list_size * (list_size - 1) / 2, 1)`, and
/// `degree = max(2^packed_vars - 1, 1)`. The unground bound is
/// `128 - log2(pairs) - log2(degree)` bits; grinding covers its rounded-up
/// deficit against the target. Unique-decoding profiles return `None`.
pub(crate) fn ood_grinding_bits(
    security: &LigeritoSecurityConfig,
    packed_vars: usize,
) -> Option<u32> {
    let level = security.levels.first()?;
    let eta = match level.regime {
        SoundnessRegime::JohnsonOod => level.eta?,
        SoundnessRegime::Udr => return None,
    };
    let rho = (-(level.log_inv_rate as f64)).exp2();
    let list_size = 1.0 / (2.0 * eta * rho.sqrt());
    let pairs = (list_size * (list_size - 1.0) / 2.0).max(1.0);
    let degree = ((packed_vars as f64).exp2() - 1.0).max(1.0);
    let collision_bits = 128.0 - pairs.log2() - degree.log2();
    let deficit = security.target_security_bits as f64 - collision_bits;
    let grinding_bits = deficit.ceil().max(0.0) as u32;
    Some(grinding_bits)
}

/// Reproduces the reference prover's k = 4 profile with 16-bit query grinding.
/// Flock's `derive_profile` fixes k = 6 and zero query grinding for Fast.
fn fast_security_config(m: usize) -> Result<LigeritoSecurityConfig, ConfigError> {
    if !(22..=35).contains(&m) {
        return Err(ConfigError::Invalid(
            "no security config for this size and profile",
        ));
    }

    let log_n = m - LOG_PACKING;
    let mut log_msg_cols = log_n - FAST_INITIAL_K;
    let mut levels = Vec::new();
    loop {
        let first = levels.is_empty();
        let k = if first {
            FAST_INITIAL_K
        } else {
            FAST_RECURSIVE_K
        };
        let mut level = LigeritoLevelConfig {
            log_inv_rate: LigeritoProfile::Fast.log_inv_rate() + levels.len(),
            log_msg_cols,
            log_num_interleaved: k,
            k_recursive: k,
            regime: SoundnessRegime::JohnsonOod,
            eta: Some(JOHNSON_ETA),
            proximity_loss: None,
            // One query lets Flock calculate the security contribution per query.
            queries: 1,
            grinding_bits: FAST_QUERY_GRINDING_BITS,
            fold_grinding_bits: 0,
            ood_samples: usize::from(!first),
            target_security_bits: FAST_SECURITY_BITS,
            expected_eps_pg_bits: 0.0,
            expected_eps_query_bits: 0.0,
            expected_eps_ood_bits: None,
        };
        let (proximity_bits, per_query_bits) = level.paper_predicted_bits();
        level.queries = ((FAST_SECURITY_BITS - FAST_QUERY_GRINDING_BITS) as f64 / per_query_bits)
            .ceil() as usize;
        level.fold_grinding_bits =
            (FAST_SECURITY_BITS as f64 - proximity_bits).ceil().max(0.0) as usize;
        level.expected_eps_pg_bits = round_bits(proximity_bits);
        level.expected_eps_query_bits = round_bits(level.queries as f64 * per_query_bits);
        level.expected_eps_ood_bits = level.paper_predicted_ood_bits().map(round_bits);
        levels.push(level);

        if log_msg_cols <= FAST_MAX_FINAL_LOG_N {
            break;
        }
        log_msg_cols -= FAST_RECURSIVE_K;
    }

    Ok(LigeritoSecurityConfig {
        m,
        log_n,
        initial_k: FAST_INITIAL_K,
        target_security_bits: FAST_SECURITY_BITS,
        analysis_version: "johnson_ood_row_union_over_bchks25_thm_4_6".into(),
        field: "f128".into(),
        hash: "sha256".into(),
        grinding_step: GrindingStep::PostCommitPreQueries,
        levels,
        final_block: FinalBlockConfig {
            yr_log_n: log_msg_cols,
        },
    })
}

fn round_bits(bits: f64) -> f64 {
    (10.0 * bits).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_profiles_match_original_configs() {
        // BLAKE3 fingerprints of Flock's canonical serialization of PR #66's original m22..m35 TOMLs.
        let fingerprints = [
            "ea2a8a8e069a402f46cac3b621a0d5ec4fa422e410a59cd402ce0875e6335ae6",
            "b7f47c3046888625127fb03f7995693915b40c6381718aa42049bea727311170",
            "f452a419d857364970ce991c5abcc5eedb34f826a52ca8cd79f6209133f39a70",
            "63a6a55488bae28b0af93fc904f655c96f131ccb08b9798ed261fc08654864fe",
            "36537633d668dfe7617eea1d27ad9cbf412b52d7dc0f3ef826371c683bba2b38",
            "aa7f5f62d683154cbde220298b541edd6938471eed5fed5b0fd9eb5d901674bc",
            "94d33bc4dd03edea9dcb3543921d32ef9c5ac613299de6672e95b23cb3d231be",
            "640972ec7c9c7a8243f98cae8812a297229adcd431386300f29bf99391fbc1d5",
            "53cf053efb6bb3f4c17623991c26a2b313244ee953ad61a81760814c3361b72b",
            "916129a4c90b1732d581783279059eb549bf272ecd05494072d0240af7cea75e",
            "001a4d0bef2bcad7be395b39e55e60d4ec25bdf5268a7c3bf5db2267d86060a8",
            "97c7ff6015cc19d373d6417ac8107ac20c83ce62858cab16939330d79249fadf",
            "8ab7b908ad3d93fc34abd63ba5700ef47565f9720087a16b54354253529b6be9",
            "40a19c7d4d41f3f47c92dda8c2961f8ebdbd1b555a13e8a04a6e481162354697",
        ];
        for (m, expected) in (22..=35).zip(fingerprints) {
            for (hash, hash_name) in [(HashKind::Sha256, "sha256"), (HashKind::Blake3, "blake3")] {
                let mut security = security_config(m, LigeritoProfile::Fast, hash).unwrap();
                assert_eq!(security.hash, hash_name);
                security.to_prover_verifier_configs().unwrap();
                security.hash = "sha256".into();
                let canonical = security.to_toml_string().unwrap();
                assert_eq!(
                    blake3::hash(canonical.as_bytes()).to_hex().as_str(),
                    expected,
                    "m={m}"
                );
            }
        }
    }

    #[test]
    fn other_profiles_keep_embedded_configs() {
        for m in 22..=35 {
            for profile in [LigeritoProfile::Slim, LigeritoProfile::Secure] {
                let toml = embedded_security_config(m, profile).unwrap();
                let mut expected = LigeritoSecurityConfig::from_toml_str(toml).unwrap();
                for (hash, hash_name) in
                    [(HashKind::Sha256, "sha256"), (HashKind::Blake3, "blake3")]
                {
                    let security = security_config(m, profile, hash).unwrap();
                    security.to_prover_verifier_configs().unwrap();
                    expected.hash = hash_name.into();
                    assert_eq!(
                        security.to_toml_string().unwrap(),
                        expected.to_toml_string().unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn fast_profile_rejects_unsupported_sizes() {
        for m in [0, 21, 36, usize::MAX] {
            assert!(security_config(m, LigeritoProfile::Fast, HashKind::Blake3).is_err());
        }
    }

    #[test]
    fn ood_round_parameters_match_the_level_zero_bound() {
        let security = security_config(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        assert_eq!(ood_grinding_bits(&security, 15), Some(0));

        let mut unique = security;
        unique.levels[0].regime = SoundnessRegime::Udr;
        unique.levels[0].eta = None;
        assert_eq!(ood_grinding_bits(&unique, 15), None);
    }
}
