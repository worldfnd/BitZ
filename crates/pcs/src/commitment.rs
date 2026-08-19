//! Checked commit-phase wrapper for flock's binary-field PCS.
//!
//! Commitment steps:
//! 1. Require the configured number of witness bits.
//! 2. Pack each 128-bit witness block into one binary-field element.
//! 3. Commit the packed witness with Flock.
//! 4. Expose the Merkle root as the public commitment.
//! 5. Retain the packed witness and Flock prover data for one opening.

use crate::CommitError;
use flock_core::field::F128 as FlockF128;
pub use flock_core::hash::HashKind;
use flock_core::pcs::Commitment as FlockCommitment;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{PcsParams, ProverData as FlockProverData, pack_witness};

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

/// Private state retained between commitment and one opening.
pub struct ProverData {
    commitment: FlockCommitment,
    packed_witness: Vec<FlockF128>,
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

    /// Commits to the exact configured number of bits.
    pub fn commit(&self, bits: &[bool]) -> Result<(Commitment, ProverData), CommitError> {
        // 1. Input Validation
        if bits.len() != self.bit_len {
            return Err(CommitError::InvalidBitLength);
        }

        // 2. Pack Witness
        let packed_witness = pack_witness(bits, self.params.m);

        // 3. Commit Packed Witness
        let (flock_commitment, flock_prover_data) =
            flock_core::pcs::commit(&packed_witness, &self.params);

        // 4. Build Public Commitment
        let commitment = Commitment {
            root: flock_commitment.root,
        };

        // 5. Retain Opening Data
        Ok((
            commitment,
            ProverData {
                commitment: flock_commitment,
                packed_witness,
                flock_prover_data,
            },
        ))
    }

    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    pub(crate) fn params(&self) -> &PcsParams {
        &self.params
    }

    pub(crate) fn statement_tags(&self) -> [u64; 5] {
        let profile_tag = match self.params.profile {
            LigeritoProfile::Fast => 0,
            LigeritoProfile::Slim => 1,
            LigeritoProfile::Secure => 2,
        };
        let hash_tag = match self.params.merkle_hash {
            HashKind::Sha256 => 0,
            HashKind::Blake3 => 1,
        };

        [
            self.params.m as u64,
            self.params.log_inv_rate as u64,
            self.params.log_batch_size as u64,
            profile_tag,
            hash_tag,
        ]
    }
}

impl ProverData {
    pub fn packed_len(&self) -> usize {
        self.packed_witness.len()
    }

    pub fn codeword_len(&self) -> usize {
        self.flock_prover_data.codeword.len()
    }

    pub(crate) fn into_opening_parts(self) -> (Vec<FlockF128>, FlockProverData) {
        (self.packed_witness, self.flock_prover_data)
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
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn commitment_is_deterministic_and_packing_preserves_boundary_bits() {
        let scheme = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
        let mut bits = vec![false; scheme.bit_len()];
        for index in [0, 1, 63, 64, 127, 128, bits.len() - 1] {
            bits[index] = true;
        }

        let (commitment, data) = scheme.commit(&bits).unwrap();
        let (second_commitment, _) = scheme.commit(&bits).unwrap();
        let mut changed_bits = bits.clone();
        changed_bits[2] = true;
        let (changed_commitment, _) = scheme.commit(&changed_bits).unwrap();

        assert_eq!(commitment, second_commitment);
        assert_ne!(commitment, changed_commitment);
        assert_eq!(data.packed_len(), bits.len() / 128);
        assert!(data.codeword_len() > 0);
        assert_eq!(data.packed_witness[0].lo, 1 | (1 << 1) | (1 << 63));
        assert_eq!(data.packed_witness[0].hi, 1 | (1 << 63));
        assert_eq!(data.packed_witness[1].lo, 1);
        assert_eq!(data.packed_witness[1].hi, 0);
        assert_eq!(data.packed_witness.last().unwrap().lo, 0);
        assert_eq!(data.packed_witness.last().unwrap().hi, 1 << 63);
    }

    #[test]
    fn statement_tags_cover_every_profile_and_hash() {
        for (profile, profile_tag) in [
            (LigeritoProfile::Fast, 0),
            (LigeritoProfile::Slim, 1),
            (LigeritoProfile::Secure, 2),
        ] {
            for (hash, hash_tag) in [(HashKind::Sha256, 0), (HashKind::Blake3, 1)] {
                let pcs = Pcs::new(22, profile, hash);
                assert_eq!(
                    pcs.statement_tags(),
                    [
                        22,
                        profile.log_inv_rate() as u64,
                        LIGERITO_INITIAL_K as u64,
                        profile_tag,
                        hash_tag,
                    ]
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn rejects_arbitrary_short_bit_lengths(len in 0usize..4096) {
            let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
            let bits = vec![false; len];

            prop_assert!(matches!(
                pcs.commit(&bits),
                Err(CommitError::InvalidBitLength)
            ));
        }

        #[test]
        fn commitment_root_reconstruction_round_trips(root in any::<[u8; 32]>()) {
            prop_assert_eq!(*Commitment::from_root(root).root(), root);
        }
    }
}
