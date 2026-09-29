use super::*;

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
            let base = (2.0 * half.powi(5) + 3.0 * half * gamma * rho) / (3.0 * rho.powf(1.5)) * n
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
