//! Checked commit-phase wrapper for flock's binary-field PCS.
//!
//! Commitment steps:
//! 1. Require the configured packed-witness length.
//! 2. Commit the caller-owned packed witness with Flock.
//! 3. Expose the Merkle root as the public commitment.
//! 4. Retain Flock prover data for later openings.

use core::mem::size_of;

use crate::VerifyError;
use crate::bridge::as_flock_f128s;
use crate::ligerito::CheckedLigerito;
use crate::ood::{OodClaim, prove, verify};
use crate::profiles::{ood_grinding_bits, security_config};
use common::{Root, Shape};
use field::F128;
pub use flock_core::hash::HashKind;
use flock_core::pcs::Commitment as FlockCommitment;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{PcsParams, ProverData as FlockProverData, commit};
use transcript::{Encoding, ProverState, VerifierState};

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
    ood_grinding_bits: Option<u32>,
    bit_len: usize,
    packed_len: usize,
}

/// Commitment and private Flock data retained for proving openings.
pub struct ProverData {
    commitment: Commitment,
    flock_prover_data: FlockProverData,
}

/// Root, parameters, and optional out-of-domain claim retained after commitment.
/// Both prover and verifier use this state for openings on the commitment transcript.
/// The verifier authenticates the claim when [`CommitScheme::verify_lin`](crate::CommitScheme::verify_lin) succeeds.
#[derive(Debug)]
pub struct Commitment {
    flock: FlockCommitment,
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
            && self.ood.is_some() == pcs.ood_grinding_bits().is_some()
    }
}

impl Pcs {
    pub fn new(
        shape: &Shape,
        security_profile: LigeritoProfile,
        merkle_hash: HashKind,
    ) -> Result<Self, ConfigError> {
        let m = shape.log_bits();
        let bit_len = 1usize
            .checked_shl(m as u32)
            .ok_or(ConfigError::Invalid("bit length overflow"))?;
        // The ladder fixes the L0 interleaving: the commit must use the same
        // `log_batch_size` as the opening's `initial_k`, or the L0 tree is not
        // reusable as Ligerito's first oracle.
        let security = security_config(m, security_profile, merkle_hash)?;
        let params = PcsParams {
            m,
            log_inv_rate: security_profile.log_inv_rate(),
            log_batch_size: security.initial_k,
            profile: security_profile,
            merkle_hash,
        };
        let checked_ligerito = CheckedLigerito::new(&params, &security)?;
        let ood_grinding_bits = ood_grinding_bits(&security, checked_ligerito.log_n_u32() as usize);
        let packed_len = 1usize
            .checked_shl(checked_ligerito.log_n_u32())
            .ok_or(ConfigError::Invalid("packed length overflow"))?;

        Ok(Self {
            params,
            checked_ligerito,
            ood_grinding_bits,
            bit_len,
            packed_len,
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
        let ood = prove(self, &root.0, packed_witness, transcript);

        // 4. Retain Opening Data
        Ok((
            root,
            ProverData {
                commitment: Commitment {
                    flock: flock_commitment,
                    ood,
                },
                flock_prover_data,
            },
        ))
    }

    /// Receives the OOD claim for the public root before subsequent protocol challenges.
    ///
    /// Mirrors [`Self::commit`]. Invalid grinding or a truncated evaluation
    /// returns [`VerifyError::MalformedProof`]; authentication of the evaluation is
    /// deferred to [`CommitScheme::verify_lin`](crate::CommitScheme::verify_lin).
    pub fn receive_commitment(
        &self,
        root: Root,
        transcript: &mut VerifierState<'_>,
    ) -> Result<Commitment, VerifyError> {
        let ood = verify(self, &root.0, transcript)?;
        Ok(Commitment {
            flock: FlockCommitment {
                root: root.0,
                params: self.params.clone(),
            },
            ood,
        })
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

    pub(crate) fn ood_grinding_bits(&self) -> Option<u32> {
        self.ood_grinding_bits
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
        let profile_tag = match self.params.profile {
            LigeritoProfile::Fast => 0,
            LigeritoProfile::Slim => 1,
            LigeritoProfile::Secure => 2,
        };
        let hash_tag = match self.params.merkle_hash {
            HashKind::Sha256 => 0,
            HashKind::Blake3 => 1,
        };

        let tags = [
            self.params.m as u64,
            self.params.log_inv_rate as u64,
            self.params.log_batch_size as u64,
            profile_tag,
            hash_tag,
        ];
        let mut encoded = [0u8; 5 * size_of::<u64>()];
        for (chunk, tag) in encoded.chunks_exact_mut(size_of::<u64>()).zip(tags) {
            chunk.copy_from_slice(&tag.to_le_bytes());
        }
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
    fn commitment_profiles_select_ood_and_preserve_transcript_agreement() {
        for profile in [
            LigeritoProfile::Fast,
            LigeritoProfile::Slim,
            LigeritoProfile::Secure,
        ] {
            let pcs = Pcs::new(&shape(), profile, HashKind::Blake3).unwrap();
            let witness = vec![F128::ZERO; pcs.packed_len()];
            let mut prover = build_prover(b"commit-test", b"profile");
            let (root, data) = pcs.commit(&witness, &mut prover).unwrap();
            let expected_ood = profile != LigeritoProfile::Secure;
            assert_eq!(data.commitment().ood.is_some(), expected_ood);
            let next_challenge = prover.verifier_message::<F128>();
            let proof = prover.finish();
            assert_eq!(proof.narg_string.is_empty(), !expected_ood);
            let mut verifier = build_verifier(b"commit-test", b"profile", &proof);
            let received = pcs.receive_commitment(root, &mut verifier).unwrap();
            assert_eq!(received.root(), data.commitment().root());
            assert_eq!(received.ood.is_some(), expected_ood);
            assert!(received.matches(&pcs));
            assert_eq!(verifier.verifier_message::<F128>(), next_challenge);
            verifier.check_eof().unwrap();
        }
    }

    #[test]
    fn commitment_is_deterministic_for_packed_boundary_bits() {
        let scheme = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
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
        let scheme = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
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
    fn encoding_covers_every_profile_and_hash() {
        for (profile, profile_tag) in [
            (LigeritoProfile::Fast, 0),
            (LigeritoProfile::Slim, 1),
            (LigeritoProfile::Secure, 2),
        ] {
            for (hash, hash_tag) in [(HashKind::Sha256, 0), (HashKind::Blake3, 1)] {
                let pcs = Pcs::new(&shape(), profile, hash).unwrap();
                // `Fast` runs the k = 4 ladder; the other profiles keep flock's
                // embedded k = 6 generation.
                let initial_k = match profile {
                    LigeritoProfile::Fast => 4,
                    LigeritoProfile::Slim | LigeritoProfile::Secure => 6,
                };
                let expected_tags = [
                    22,
                    profile.log_inv_rate() as u64,
                    initial_k,
                    profile_tag,
                    hash_tag,
                ];
                let encoded = pcs.encode();
                let encoded = encoded.as_ref();
                assert_eq!(encoded.len(), 5 * size_of::<u64>());
                for (chunk, tag) in encoded.chunks_exact(size_of::<u64>()).zip(expected_tags) {
                    assert_eq!(chunk, tag.to_le_bytes());
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn rejects_arbitrary_short_packed_witnesses(len in 0usize..4096) {
            let pcs = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
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
