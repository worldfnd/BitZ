//! Checked commit-phase wrapper for flock's binary-field PCS.
//!
//! Commitment steps:
//! 1. Require the configured packed-witness length.
//! 2. Commit the caller-owned packed witness with Flock.
//! 3. Expose the Merkle root as the public commitment.
//! 4. Retain Flock prover data for later openings.

use crate::VerifyError;
use crate::bridge::as_flock_f128s;
use crate::ligerito::CheckedLigerito;
use crate::ood::{self, OodClaim};
use crate::profiles::security_config;
use common::{Root, SecurityLevel, Shape};
use field::F128;
use flock_core::hash::HashKind;
use flock_core::pcs::Commitment as FlockCommitment;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{LOG_PACKING, PcsParams, ProverData as FlockProverData, commit};
use transcript::{Encoding, ProverState, PublicTranscript, VerifierState};

// Increment this version when parameter derivation or transcript rules change.
// This includes protocol changes in Flock or the selected hash.
const PROTOCOL_VERSION: &[u8] = b"bitz/pcs/security/v1";

/// Errors from PCS configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The configuration violates the named invariant.
    Invalid(&'static str),
}

/// Errors from commitment creation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommitError {
    /// The packed witness length does not match the configured polynomial.
    PackedWitnessLengthMismatch,
}

#[derive(Clone, Debug)]
pub struct Pcs {
    params: PcsParams,
    checked_ligerito: CheckedLigerito,
    bit_len: usize,
    packed_len: usize,
    security_level: SecurityLevel,
}

/// Commitment and private Flock data retained for proving openings.
pub struct ProverData {
    pub(crate) commitment: Commitment,
    flock_prover_data: FlockProverData,
}

/// Root, parameters, and optional out-of-domain claim retained after commitment.
/// Both prover and verifier use this state for openings on the commitment transcript.
/// The verifier authenticates the claim when [`CommitScheme::verify_lin`](crate::CommitScheme::verify_lin) succeeds.
#[derive(Debug)]
pub struct Commitment {
    flock: FlockCommitment,
    security_level: SecurityLevel,
    pub(crate) ood: Option<OodClaim>,
}

impl Commitment {
    /// Returns the public commitment root.
    pub fn root(&self) -> Root {
        Root(self.flock.root)
    }

    pub(crate) fn matches(&self, pcs: &Pcs) -> bool {
        let expected = pcs.params();
        let actual = &self.flock.params;
        expected.m == actual.m
            && expected.log_inv_rate == actual.log_inv_rate
            && expected.log_batch_size == actual.log_batch_size
            && expected.profile == actual.profile
            && expected.merkle_hash == actual.merkle_hash
            && self.security_level == pcs.security_level()
            && self.ood.is_some() == (pcs.security_level() == SecurityLevel::Bits100)
    }
}

impl Pcs {
    /// Derives the selected round budget from the padded witness size.
    /// The 100-bit profile uses Johnson decoding; the 128-bit profile uses unique decoding.
    pub fn new(shape: &Shape, security_level: SecurityLevel) -> Result<Self, ConfigError> {
        let m = shape.log_bits();
        let security = security_config(m, security_level)?;
        let bit_len = 1usize
            .checked_shl(m as u32)
            .ok_or(ConfigError::Invalid("bit length overflow"))?;
        // The ladder fixes the L0 interleaving: the commit must use the same
        // `log_batch_size` as the opening's `initial_k`, or the L0 tree is not
        // reusable as Ligerito's first oracle.
        let params = PcsParams {
            m,
            log_inv_rate: security.levels[0].log_inv_rate,
            log_batch_size: security.initial_k,
            // Flock retains this tag; explicit geometry controls our encoding.
            profile: LigeritoProfile::Fast,
            merkle_hash: HashKind::Blake3,
        };
        let checked_ligerito = CheckedLigerito::new(&params, &security)?;
        let packed_len = bit_len >> LOG_PACKING;
        Ok(Self {
            params,
            checked_ligerito,
            bit_len,
            packed_len,
            security_level,
        })
    }

    /// Commits and samples the initial OOD claim when the security profile requires it.
    /// Call before witness-dependent challenges and continue with the same transcript.
    #[tracing::instrument(name = "Commit witness", skip_all)]
    pub fn commit(
        &self,
        packed_witness: &[F128],
        transcript: &mut ProverState,
    ) -> Result<(Root, ProverData), CommitError> {
        // 1. Input Validation
        if packed_witness.len() != self.packed_len() {
            return Err(CommitError::PackedWitnessLengthMismatch);
        }

        // 2. Commit Packed Witness
        let (flock_commitment, flock_prover_data) =
            commit(as_flock_f128s(packed_witness), &self.params);

        // 3. Build Public Commitment
        let root = Root(flock_commitment.root);
        self.bind_commitment(root, transcript);
        let ood = ood::prove(self, packed_witness, transcript);

        // 4. Retain Opening Data
        Ok((
            root,
            ProverData {
                commitment: Commitment {
                    flock: flock_commitment,
                    security_level: self.security_level,
                    ood,
                },
                flock_prover_data,
            },
        ))
    }

    /// Receives commitment state before subsequent protocol challenges.
    /// Verification of an opening authenticates the retained initial claim.
    pub fn receive_commitment(
        &self,
        root: Root,
        transcript: &mut VerifierState<'_>,
    ) -> Result<Commitment, VerifyError> {
        self.bind_commitment(root, transcript);
        let ood = ood::verify(self, transcript)?;
        Ok(Commitment {
            flock: FlockCommitment {
                root: root.0,
                params: self.params.clone(),
            },
            security_level: self.security_level,
            ood,
        })
    }

    fn bind_commitment(&self, root: Root, transcript: &mut impl PublicTranscript) {
        transcript.public_message(b"bitz/pcs/commit/v1" as &[u8]);
        transcript.public_message(&root.0);
        transcript.public_message(self);
    }

    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Returns the required number of packed `F128` elements.
    pub fn packed_len(&self) -> usize {
        self.packed_len
    }

    pub(crate) fn params(&self) -> &PcsParams {
        &self.params
    }

    /// Returns the selected classical PCS round budget.
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level
    }

    /// Maps each native PoW call to its checked effective difficulty.
    pub(crate) fn pow_schedule(&self) -> &[(u32, u32)] {
        self.checked_ligerito.pow_schedule()
    }

    pub(crate) fn prover_config(&self) -> &flock_core::pcs::ligerito::ProverConfig {
        self.checked_ligerito.prover_config()
    }

    pub(crate) fn verifier_config(&self) -> &flock_core::pcs::ligerito::VerifierConfig {
        self.checked_ligerito.verifier_config()
    }

    pub(crate) fn final_log_n(&self) -> usize {
        self.checked_ligerito.final_log_n()
    }
}

impl Encoding<[u8]> for Pcs {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut encoded = PROTOCOL_VERSION.to_vec();
        encoded.extend_from_slice(&(self.bit_len as u64).to_le_bytes());
        encoded.extend_from_slice(&self.security_level.bits().to_le_bytes());
        encoded
    }
}

impl ProverData {
    pub fn codeword_len(&self) -> usize {
        self.flock_prover_data.codeword.len()
    }

    /// The commitment this data opens against.
    pub fn root(&self) -> Root {
        self.commitment.root()
    }

    pub(crate) fn flock_data(&self) -> &FlockProverData {
        &self.flock_prover_data
    }

    /// Returns the shared commitment without the private proving data.
    pub fn commitment(&self) -> &Commitment {
        &self.commitment
    }
}

#[cfg(test)]
mod tests {
    use flock_core::pcs::pack_witness;
    use num_traits::ConstZero;
    use proptest::prelude::*;
    use transcript::{build_prover, build_verifier};

    use super::*;

    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    #[test]
    fn every_dynamic_size_builds_a_complete_native_pow_schedule() {
        for m in 20..=35 {
            let shape = Shape::new(7, m - 7).unwrap();
            for level in [SecurityLevel::Bits100, SecurityLevel::Bits128] {
                let pcs = Pcs::new(&shape, level).unwrap();
                let config = security_config(m, level).unwrap();
                let expected: usize = config
                    .levels
                    .iter()
                    .map(|level| 1 + level.fold_grinding_bits.min(level.k_recursive))
                    .sum();
                assert_eq!(pcs.pow_schedule().len(), expected);
            }
        }
    }

    #[test]
    fn commitment_profiles_select_ood_and_preserve_transcript_agreement() {
        for security in [SecurityLevel::Bits100, SecurityLevel::Bits128] {
            let pcs = Pcs::new(&shape(), security).unwrap();
            let witness = vec![F128::ZERO; pcs.packed_len()];
            let mut prover = build_prover(b"commit-test", b"profile");
            let (root, data) = pcs.commit(&witness, &mut prover).unwrap();
            let expected_ood = security == SecurityLevel::Bits100;
            assert_eq!(data.commitment().ood.is_some(), expected_ood);
            let next_challenge = prover.verifier_message::<F128>();
            let proof = prover.finish();
            assert_eq!(proof.narg_string.len(), if expected_ood { 16 } else { 0 });
            let mut verifier = build_verifier(b"commit-test", b"profile", &proof);
            let received = pcs.receive_commitment(root, &mut verifier).unwrap();
            assert_eq!(received.root(), data.commitment().root());
            assert_eq!(received.ood.is_some(), expected_ood);
            if let Some(claim) = received.ood.as_ref() {
                assert_eq!(claim.point, data.commitment().ood.as_ref().unwrap().point);
                assert_eq!(claim.point.len(), pcs.packed_len().ilog2() as usize);
            }
            assert!(received.matches(&pcs));
            assert_eq!(verifier.verifier_message::<F128>(), next_challenge);
            verifier.check_eof().unwrap();
        }
    }

    #[test]
    fn commitment_is_deterministic_for_packed_boundary_bits() {
        let scheme = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
        let mut packed_witness = vec![F128::ZERO; scheme.packed_len()];
        packed_witness[0] = F128::new(1 | (1 << 1) | (1 << 63), 1 | (1 << 63));
        packed_witness[1] = F128::new(1, 0);
        packed_witness.last_mut().unwrap().hi = 1 << 63;

        let (commitment, data) = scheme
            .commit(
                &packed_witness,
                &mut build_prover(b"commit-test", b"witness"),
            )
            .unwrap();
        let (second_commitment, _) = scheme
            .commit(
                &packed_witness,
                &mut build_prover(b"commit-test", b"witness"),
            )
            .unwrap();
        let mut changed_witness = packed_witness.clone();
        changed_witness[0].lo |= 1 << 2;
        let (changed_commitment, _) = scheme
            .commit(
                &changed_witness,
                &mut build_prover(b"commit-test", b"witness"),
            )
            .unwrap();

        assert_eq!(commitment, second_commitment);
        assert_ne!(commitment, changed_commitment);
        assert!(data.codeword_len() > 0);
    }

    #[test]
    fn caller_packing_layout_matches_flock() {
        let scheme = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
        let mut bits = vec![false; scheme.bit_len()];
        for index in [0, 63, 64, 127, 128, bits.len() - 1] {
            bits[index] = true;
        }

        let packed = pack_witness(&bits, scheme.params.m);

        assert_eq!((packed[0].lo, packed[0].hi), (1 | (1 << 63), 1 | (1 << 63)));
        assert_eq!((packed[1].lo, packed[1].hi), (1, 0));
        assert_eq!(
            (packed.last().unwrap().lo, packed.last().unwrap().hi),
            (0, 1 << 63)
        );
    }

    #[test]
    fn explicit_profiles_bind_the_target_and_witness_size() {
        let shape = Shape::new(7, 13).unwrap();
        let low = Pcs::new(&shape, SecurityLevel::Bits100).unwrap();
        let high = Pcs::new(&shape, SecurityLevel::Bits128).unwrap();
        // Fixed encoding: version tag, padded bit count (u64 LE), target (u32 LE).
        assert_eq!(
            low.encode().as_ref(),
            b"bitz/pcs/security/v1\x00\x00\x10\x00\x00\x00\x00\x00\x64\x00\x00\x00"
        );
        assert_ne!(low.encode().as_ref(), high.encode().as_ref());
        let (_, data) = low
            .commit(
                &vec![F128::ZERO; low.packed_len()],
                &mut build_prover(b"commit-test", b"witness"),
            )
            .unwrap();
        assert_eq!(crate::ligerito::validate_prover_data(&low, &data), Ok(()));
        assert_eq!(
            crate::ligerito::validate_prover_data(&high, &data),
            Err(crate::ProveError::ProverDataMismatch)
        );
        let larger_shape = Shape::new(7, 14).unwrap();
        let changed = Pcs::new(&larger_shape, SecurityLevel::Bits100).unwrap();
        assert_ne!(low.encode().as_ref(), changed.encode().as_ref());
        assert!(crate::ligerito::validate_prover_data(&changed, &data).is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn rejects_arbitrary_short_packed_witnesses(len in 0usize..4096) {
            let pcs = Pcs::new(&shape(), SecurityLevel::Bits100).unwrap();
            let packed_witness = vec![F128::ZERO; len];
            let mut transcript = build_prover(b"commit-test", b"short");

            prop_assert!(matches!(
                pcs.commit(&packed_witness, &mut transcript),
                Err(CommitError::PackedWitnessLengthMismatch)
            ));
            let proof = transcript.finish();
            prop_assert!(proof.narg_string.is_empty() && proof.hints.is_empty());
        }
    }
}
