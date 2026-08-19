//! Checked commit-phase wrapper for flock's binary-field PCS.
//!
//! Commitment steps:
//! 1. Require the configured packed-witness length.
//! 2. Commit the caller-owned packed witness with Flock.
//! 3. Expose the Merkle root as the public commitment.
//! 4. Retain Flock prover data for one opening.

use core::mem::size_of;

use crate::CommitError;
use crate::bridge::as_flock_f128s;
use field::F128;
pub use flock_core::hash::HashKind;
use flock_core::pcs::Commitment as FlockCommitment;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{LOG_PACKING, PcsParams, ProverData as FlockProverData};
use transcript::Encoding;

/// Initial Ligerito fold size required by Flock's registered security profiles.
/// The value `6` selects 64 lanes
/// See: https://github.com/succinctlabs/flock/blob/879072249e52b8b9054bf0c6a034cec20f8f6fc7/crates/flock-core/src/pcs/ligerito.rs#L1245
const LIGERITO_INITIAL_K: usize = 6;

#[derive(Clone, Debug)]
pub struct Pcs {
    params: PcsParams,
    // number of bits we commit to
    bit_len: usize,
}

/// The public commitment. Trusted parameters remain in [`Pcs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment {
    root: [u8; 32],
}

/// Flock state retained between commitment and one opening.
pub struct ProverData {
    commitment: FlockCommitment,
    flock_prover_data: FlockProverData,
}

impl Pcs {
    pub fn new(m: usize, security_profile: LigeritoProfile, merkle_hash: HashKind) -> Self {
        Self {
            params: PcsParams {
                m,
                log_inv_rate: security_profile.log_inv_rate(),
                log_batch_size: LIGERITO_INITIAL_K,
                profile: security_profile,
                merkle_hash,
            },
            bit_len: 1 << m,
        }
    }

    /// Commits to the exact configured number of packed field elements.
    pub fn commit(&self, packed_witness: &[F128]) -> Result<(Commitment, ProverData), CommitError> {
        // 1. Input Validation
        if packed_witness.len() != self.packed_len() {
            return Err(CommitError::InvalidBitLength);
        }

        // 2. Commit Packed Witness
        let (flock_commitment, flock_prover_data) =
            flock_core::pcs::commit(as_flock_f128s(packed_witness), &self.params);

        // 3. Build Public Commitment
        let commitment = Commitment {
            root: flock_commitment.root,
        };

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
        self.bit_len >> LOG_PACKING
    }

    pub(crate) fn params(&self) -> &PcsParams {
        &self.params
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

    pub(crate) fn into_flock_data(self) -> FlockProverData {
        self.flock_prover_data
    }

    pub(crate) fn commitment(&self) -> &FlockCommitment {
        &self.commitment
    }
}

impl Commitment {
    /// Reconstructs a public commitment from its canonical root.
    pub const fn from_root(root: [u8; 32]) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &[u8; 32] {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use flock_core::pcs::pack_witness;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn commitment_is_deterministic_for_packed_boundary_bits() {
        let scheme = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
        let mut packed_witness = vec![F128::default(); scheme.packed_len()];
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
        let scheme = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
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
                let pcs = Pcs::new(22, profile, hash);
                let expected_tags = [
                    22,
                    profile.log_inv_rate() as u64,
                    LIGERITO_INITIAL_K as u64,
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
            let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
            let packed_witness = vec![F128::default(); len];

            prop_assert!(matches!(
                pcs.commit(&packed_witness),
                Err(CommitError::InvalidBitLength)
            ));
        }

        #[test]
        fn commitment_root_reconstruction_round_trips(root in any::<[u8; 32]>()) {
            prop_assert_eq!(*Commitment::from_root(root).root(), root);
        }
    }
}
