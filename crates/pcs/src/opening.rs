//! Ring-switched recursive Ligerito openings.
//!
//! Flock calls the challenger while it creates or verifies the proof. The
//! adapter writes every challenge-bearing value into `narg_string`. The hint
//! carries a bounded backend proof object because Flock verification needs
//! its Merkle rows before replay starts. During replay, the adapter compares
//! every duplicated transcript value with `narg_string`. Merkle rows and
//! authentication hashes remain hint-only and Flock verifies them.

use bincode::Options;
use field::F128 as LocalF128;
use flock_core::field::{F128 as BackendF128, PHI_8_TABLE};
use flock_core::pcs::BatchOpeningProofLigerito;
use flock_core::pcs::ligerito::VerifierConfig;
use flock_core::zerocheck::PaddingSpec;
use transcript::{ProverState, VerifierState};

use crate::bridge::as_backend_f128s;
use crate::challenger::{ProverChallenger, VerifierChallenger};
use crate::{CommitError, CommitScheme, Commitment, Pcs, ProverData};

const PROOF_HINT_LIMIT: usize = 64 * 1024 * 1024;
const STATEMENT_LABEL: &[u8] = b"f2z/pcs/linear-opening/v1";

struct RingSwitchPoint {
    z_skip: BackendF128,
    x_outer: Vec<BackendF128>,
}

impl CommitScheme for Pcs {
    type Commitment = Commitment;
    type ProverData = ProverData;

    fn commit(&self, bits: &[bool]) -> Result<(Commitment, ProverData), CommitError> {
        Pcs::commit(self, bits)
    }

    fn prove_lin(
        &self,
        data: &ProverData,
        coeffs: &[LocalF128],
        target: LocalF128,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError> {
        check_len(self.bit_len(), coeffs.len())?;
        if data.bit_len() != self.bit_len() || !params_match(self, data) {
            return Err(CommitError::InvalidConfiguration);
        }
        let point = recover_ring_switch_point(as_backend_f128s(coeffs), self.config().m)?;
        bind_statement_prover(self, &data.commitment().root, &point, target, transcript);

        let packed_witness = data.take_packed_witness()?;
        let lig_config = self
            .params()
            .ligerito_prover_config()
            .map_err(|_| CommitError::InvalidConfiguration)?;
        let x_outers = [point.x_outer.as_slice()];
        let proof = {
            let mut challenger = ProverChallenger::new(transcript);
            flock_core::pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
                packed_witness,
                data.backend(),
                data.commitment(),
                &x_outers,
                &[],
                &[],
                &PaddingSpec::dense(self.config().m),
                &lig_config,
                &mut challenger,
            )
        };
        let bytes = proof_options()
            .serialize(&proof)
            .map_err(|_| CommitError::Backend)?;
        if bytes.len() > PROOF_HINT_LIMIT {
            return Err(CommitError::Backend);
        }
        transcript.hint_bytes(&bytes);
        Ok(())
    }

    fn verify_lin(
        &self,
        commitment: &Commitment,
        coeffs: &[LocalF128],
        target: LocalF128,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), CommitError> {
        check_len(self.bit_len(), coeffs.len())?;
        let point = recover_ring_switch_point(as_backend_f128s(coeffs), self.config().m)?;
        bind_statement_verifier(self, commitment.root(), &point, target, transcript);

        let bytes = transcript
            .hint_bytes(PROOF_HINT_LIMIT)
            .map_err(|_| CommitError::MalformedProof)?;
        let proof: BatchOpeningProofLigerito = proof_options()
            .deserialize(&bytes)
            .map_err(|_| CommitError::MalformedProof)?;
        let lig_config = self
            .params()
            .ligerito_verifier_config()
            .map_err(|_| CommitError::InvalidConfiguration)?;
        validate_proof_shape(&proof, &lig_config, self.config().m, commitment.root())?;

        let backend_commitment = commitment.backend(self.params());
        let x_outers = [point.x_outer.as_slice()];
        let claims = [backend(target)];
        let z_skips = [point.z_skip];
        let (result, bridge_failed) = {
            let mut challenger = VerifierChallenger::new(transcript);
            let result = flock_core::pcs::verify_opening_batch_ligerito_mixed(
                &backend_commitment,
                &claims,
                &z_skips,
                &x_outers,
                &[],
                &proof,
                &lig_config,
                &mut challenger,
            );
            (result, challenger.failed())
        };
        if bridge_failed {
            return Err(CommitError::MalformedProof);
        }
        result.map_err(|_| CommitError::VerificationFailed)
    }
}

fn proof_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(PROOF_HINT_LIMIT as u64)
        .reject_trailing_bytes()
}

fn check_len(expected: usize, actual: usize) -> Result<(), CommitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CommitError::CoefficientLengthMismatch { expected, actual })
    }
}

fn params_match(pcs: &Pcs, data: &ProverData) -> bool {
    let expected = pcs.params();
    let actual = &data.commitment().params;
    expected.m == actual.m
        && expected.log_inv_rate == actual.log_inv_rate
        && expected.log_batch_size == actual.log_batch_size
        && expected.profile == actual.profile
        && expected.merkle_hash == actual.merkle_hash
}

/// Recovers the Flock quirky point and rejects every other linear functional.
fn recover_ring_switch_point(
    coeffs: &[BackendF128],
    m: usize,
) -> Result<RingSwitchPoint, CommitError> {
    if m < 7 || coeffs.len() != 1usize << m {
        return Err(CommitError::UnsupportedCoefficients);
    }

    let mut lambda = [BackendF128::ZERO; 64];
    let mut x_outer = vec![BackendF128::ZERO; m - 6];
    let mut total = BackendF128::ZERO;
    for (index, &coefficient) in coeffs.iter().enumerate() {
        total += coefficient;
        lambda[index & 63] += coefficient;
        for bit in 6..m {
            if (index >> bit) & 1 == 1 {
                x_outer[bit - 6] += coefficient;
            }
        }
    }
    if total != BackendF128::ONE {
        return Err(CommitError::UnsupportedCoefficients);
    }

    let z_skip = lambda
        .iter()
        .zip(PHI_8_TABLE.iter())
        .fold(BackendF128::ZERO, |sum, (&weight, &node)| {
            sum + weight * node
        });
    let prefix = flock_core::pcs::ring_switch::build_claim_weights(z_skip, x_outer[0]);
    let suffix = flock_core::zerocheck::univariate_skip::build_eq(&x_outer[1..]);
    for (index, &coefficient) in coeffs.iter().enumerate() {
        if coefficient != prefix[index & 127] * suffix[index >> 7] {
            return Err(CommitError::UnsupportedCoefficients);
        }
    }
    Ok(RingSwitchPoint { z_skip, x_outer })
}

fn bind_statement_prover(
    pcs: &Pcs,
    root: &[u8; 32],
    point: &RingSwitchPoint,
    target: LocalF128,
    transcript: &mut ProverState,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);
    for tag in pcs.statement_tags() {
        transcript.public_message(&tag);
    }
    transcript.public_message(&local(point.z_skip));
    transcript.public_message(&(point.x_outer.len() as u64));
    for &coordinate in &point.x_outer {
        transcript.public_message(&local(coordinate));
    }
    transcript.public_message(&target);
}

fn bind_statement_verifier(
    pcs: &Pcs,
    root: &[u8; 32],
    point: &RingSwitchPoint,
    target: LocalF128,
    transcript: &mut VerifierState<'_>,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);
    for tag in pcs.statement_tags() {
        transcript.public_message(&tag);
    }
    transcript.public_message(&local(point.z_skip));
    transcript.public_message(&(point.x_outer.len() as u64));
    for &coordinate in &point.x_outer {
        transcript.public_message(&local(coordinate));
    }
    transcript.public_message(&target);
}

fn validate_proof_shape(
    proof: &BatchOpeningProofLigerito,
    config: &VerifierConfig,
    m: usize,
    expected_root: &[u8; 32],
) -> Result<(), CommitError> {
    if proof.ring_switches.len() != 1
        || proof.ring_switches[0].s_hat_v.len() != 128
        || &proof.ligerito.initial_root != expected_root
    {
        return Err(CommitError::MalformedProof);
    }
    let lig = &proof.ligerito;
    let r = config.recursive_steps;
    if r == 0
        || lig.recursive_roots.len() != r
        || lig.recursive_proofs.len() != r - 1
        || lig.grinding_nonces.len() != r + 1
    {
        return Err(CommitError::MalformedProof);
    }

    let expected_ood: usize = config.ood_samples.iter().skip(1).sum();
    let expected_fold_nonces = positive_fold_nonce_count(config);
    let expected_sumchecks = 2
        + config.initial_k
        + config.recursive_ks.iter().sum::<usize>()
        + expected_ood
        + r.saturating_sub(1);
    if lig.ood_values.len() != expected_ood
        || lig.fold_grinding_nonces.len() != expected_fold_nonces
        || lig.sumcheck_transcript.len() != expected_sumchecks
    {
        return Err(CommitError::MalformedProof);
    }

    if lig.initial_proof.opened_rows.len() != config.queries[0]
        || lig
            .initial_proof
            .opened_rows
            .iter()
            .any(|row| row.len() != 1usize << config.initial_k)
    {
        return Err(CommitError::MalformedProof);
    }
    for (level, recursive) in lig.recursive_proofs.iter().enumerate() {
        if recursive.opened_rows.len() != config.queries[level + 1]
            || recursive
                .opened_rows
                .iter()
                .any(|row| row.len() != 1usize << config.recursive_ks[level])
        {
            return Err(CommitError::MalformedProof);
        }
    }
    let last_k = *config
        .recursive_ks
        .last()
        .ok_or(CommitError::MalformedProof)?;
    if lig.final_proof.opened_rows.len() != config.queries[r]
        || lig
            .final_proof
            .opened_rows
            .iter()
            .any(|row| row.len() != 1usize << last_k)
    {
        return Err(CommitError::MalformedProof);
    }
    let yr_log = (m - 7)
        .checked_sub(config.initial_k)
        .and_then(|n| n.checked_sub(config.recursive_ks.iter().sum()))
        .ok_or(CommitError::MalformedProof)?;
    if lig.final_proof.yr.len() != 1usize << yr_log {
        return Err(CommitError::MalformedProof);
    }
    Ok(())
}

fn positive_fold_nonce_count(config: &VerifierConfig) -> usize {
    let count_level =
        |k: usize, bits: usize| (0..k).filter(|&j| bits.saturating_sub(j) > 0).count();
    let initial = count_level(config.initial_k, config.fold_grinding_bits[0]);
    initial
        + config
            .recursive_ks
            .iter()
            .enumerate()
            .map(|(i, &k)| count_level(k, config.fold_grinding_bits[i + 1]))
            .sum::<usize>()
}

#[inline]
fn local(value: BackendF128) -> LocalF128 {
    LocalF128::new(value.lo, value.hi)
}

#[inline]
fn backend(value: LocalF128) -> BackendF128 {
    BackendF128::new(value.lo, value.hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MerkleHash, SecurityProfile};
    use transcript::{build_prover, build_verifier};

    fn coefficients(m: usize, z: BackendF128, x: &[BackendF128]) -> Vec<BackendF128> {
        let prefix = flock_core::pcs::ring_switch::build_claim_weights(z, x[0]);
        let suffix = flock_core::zerocheck::univariate_skip::build_eq(&x[1..]);
        (0..1usize << m)
            .map(|i| prefix[i & 127] * suffix[i >> 7])
            .collect()
    }

    #[test]
    fn recovers_and_checks_ring_switch_coefficients() {
        let m = 10;
        let z = BackendF128::new(7, 11);
        let x = [
            BackendF128::new(13, 17),
            BackendF128::new(19, 23),
            BackendF128::new(29, 31),
            BackendF128::new(37, 41),
        ];
        let mut coeffs = coefficients(m, z, &x);
        let recovered = recover_ring_switch_point(&coeffs, m).unwrap();
        assert_eq!(recovered.z_skip, z);
        assert_eq!(recovered.x_outer, x);

        coeffs[17] += BackendF128::ONE;
        assert!(matches!(
            recover_ring_switch_point(&coeffs, m),
            Err(CommitError::UnsupportedCoefficients)
        ));
    }

    #[test]
    fn rejects_a_generic_linear_functional() {
        let mut coeffs = vec![BackendF128::ZERO; 1 << 7];
        coeffs[0] = BackendF128::ONE;
        coeffs[1] = BackendF128::ONE;
        assert!(recover_ring_switch_point(&coeffs, 7).is_err());
    }

    #[test]
    fn commit_prove_verify_round_trip_and_mutations() {
        let pcs = Pcs::new(22, SecurityProfile::Fast, MerkleHash::Blake3).unwrap();
        let mut bits = vec![false; pcs.bit_len()];
        for index in [0, 1, 127, 128, 65_537, bits.len() - 1] {
            bits[index] = true;
        }
        let z = BackendF128::new(7, 11);
        let x: Vec<BackendF128> = (0..16)
            .map(|i| BackendF128::new(13 + i as u64, 29 + 3 * i as u64))
            .collect();
        let backend_coeffs = coefficients(22, z, &x);
        let coeffs: Vec<LocalF128> = backend_coeffs.iter().copied().map(local).collect();
        let target = bits
            .iter()
            .zip(coeffs.iter())
            .filter_map(|(&bit, &coefficient)| bit.then_some(coefficient))
            .fold(LocalF128::from(0u64), |sum, value| sum + value);

        let (commitment, data) = pcs.commit(&bits).unwrap();
        let mut prover = build_prover(b"pcs-opening-test", b"m22");
        pcs.prove_lin(&data, &coeffs, target, &mut prover).unwrap();
        let proof = prover.finish();

        let mut verifier = build_verifier(b"pcs-opening-test", b"m22", &proof);
        pcs.verify_lin(&commitment, &coeffs, target, &mut verifier)
            .unwrap();
        verifier.check_eof().unwrap();

        let mut changed_root = *commitment.root();
        changed_root[0] ^= 1;
        let changed_commitment = Commitment::from_root(changed_root);
        let mut verifier = build_verifier(b"pcs-opening-test", b"m22", &proof);
        assert!(
            pcs.verify_lin(&changed_commitment, &coeffs, target, &mut verifier)
                .is_err()
        );

        let mut changed_proof = proof;
        changed_proof.narg_string[0] ^= 1;
        let mut verifier = build_verifier(b"pcs-opening-test", b"m22", &changed_proof);
        assert!(
            pcs.verify_lin(&commitment, &coeffs, target, &mut verifier)
                .is_err()
        );
    }
}
