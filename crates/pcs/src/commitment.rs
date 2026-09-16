//! Checked commit-phase wrapper for flock's binary-field PCS.
//!
//! Commitment steps:
//! 1. Require the configured packed-witness length.
//! 2. Commit the caller-owned packed witness with Flock.
//! 3. Expose the Merkle root as the public commitment.
//! 4. Retain Flock prover data for later openings.

use core::mem::size_of;

use crate::bridge::as_flock_f128s;
use crate::ligerito::CheckedLigerito;
use common::{Root, Shape};
use field::F128;
pub use flock_core::hash::HashKind;
use flock_core::pcs::Commitment as FlockCommitment;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{PcsParams, ProverData as FlockProverData};
use transcript::Encoding;


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
}

/// Flock state retained between commitment and openings.
pub struct ProverData {
    commitment: FlockCommitment,
    flock_prover_data: FlockProverData,
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
        let security = crate::ligerito::security_config(m, security_profile, merkle_hash)?;
        let params = PcsParams {
            m,
            log_inv_rate: security_profile.log_inv_rate(),
            log_batch_size: security.initial_k,
            profile: security_profile,
            merkle_hash,
        };
        let checked_ligerito = CheckedLigerito::new(&params, &security)?;
        let packed_len = 1usize
            .checked_shl(checked_ligerito.log_n_u32())
            .ok_or(ConfigError::Invalid("packed length overflow"))?;

        Ok(Self {
            params,
            checked_ligerito,
            bit_len,
            packed_len,
        })
    }

    /// Commits to the exact configured number of packed field elements.
    pub fn commit(&self, packed_witness: &[F128]) -> Result<(Root, ProverData), CommitError> {
        // 1. Input Validation
        if packed_witness.len() != self.packed_len() {
            return Err(CommitError::PackedWitnessLengthMismatch);
        }

        // 2. Commit Packed Witness
        let (flock_commitment, flock_prover_data) =
            flock_core::pcs::commit(as_flock_f128s(packed_witness), &self.params);

        // 3. Build Public Commitment
        let commitment = Root(flock_commitment.root);

        // 4. Retain Opening Data
        Ok((
            commitment,
            ProverData {
                commitment: flock_commitment,
                flock_prover_data,
            },
        ))
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
        Root(self.commitment.root)
    }

    pub(crate) fn flock_data(&self) -> &FlockProverData {
        &self.flock_prover_data
    }

    pub(crate) fn commitment(&self) -> &FlockCommitment {
        &self.commitment
    }
}

#[cfg(test)]
mod tests {
    use flock_core::pcs::pack_witness;
    use num_traits::ConstZero;
    use proptest::prelude::*;

    use super::*;

    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    #[test]
    fn commitment_is_deterministic_for_packed_boundary_bits() {
        let scheme = Pcs::new(&shape(), LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let mut packed_witness = vec![F128::ZERO; scheme.packed_len()];
        packed_witness[0] = F128::new(1 | (1 << 1) | (1 << 63), 1 | (1 << 63));
        packed_witness[1] = F128::new(1, 0);
        packed_witness.last_mut().unwrap().hi = 1 << 63;

        let (commitment, data) = scheme.commit(&packed_witness).unwrap();
        let (second_commitment, _) = scheme.commit(&packed_witness).unwrap();
        let mut changed_witness = packed_witness.clone();
        changed_witness[0].lo |= 1 << 2;
        let (changed_commitment, _) = scheme.commit(&changed_witness).unwrap();

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

            prop_assert!(matches!(
                pcs.commit(&packed_witness),
                Err(CommitError::PackedWitnessLengthMismatch)
            ));
        }
    }
}
