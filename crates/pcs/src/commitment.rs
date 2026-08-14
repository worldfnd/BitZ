//! Checked commit-phase wrapper for flock's binary-field PCS.
//!
//! Commitment steps:
//! 1. Require the configured number of witness bits.
//! 2. Pack each 128-bit witness block into one binary-field element.
//! 3. Commit the packed witness with FLoCK.
//! 4. Expose the Merkle root as the public commitment.
//! 5. Retain the packed witness and FLoCK prover data for one opening.

use crate::CommitError;
use flock_core::field::F128 as FlockF128;
pub use flock_core::hash::HashKind;
use flock_core::pcs::Commitment as FlockCommitment;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{PcsParams, ProverData as FlockProverData, pack_witness};

/// Initial Ligerito fold size required by FLoCK's registered security profiles.
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
    // length of witness before packing
    bit_len: usize,
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
                bit_len: self.bit_len,
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
    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

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
    use super::*;

    #[test]
    fn rejects_the_wrong_bit_length_before_flock() {
        let scheme = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
        assert!(matches!(
            scheme.commit(&[false; 128]),
            Err(CommitError::InvalidBitLength)
        ));
    }

    #[test]
    fn commitment_is_deterministic_and_retains_the_packed_witness() {
        let scheme = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
        let mut bits = vec![false; scheme.bit_len()];
        for index in [0, 1, 63, 64, 127, 128, bits.len() - 1] {
            bits[index] = true;
        }
        let data = scheme.commit(&bits).unwrap();
        let data2 = scheme.commit(&bits).unwrap();
        assert_eq!(data.0.root, data2.0.root);
        assert_eq!(data.1.bit_len(), bits.len());
        assert_eq!(data.1.packed_len(), bits.len() / 128);
        assert!(data.1.codeword_len() > 0);
    }

    #[test]
    fn prover_data_moves_into_opening_parts() {
        let scheme = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3);
        let data = scheme.commit(&vec![false; scheme.bit_len()]).unwrap();
        let (packed_witness, flock_data) = data.1.into_opening_parts();
        assert_eq!(packed_witness.len(), scheme.bit_len() / 128);
        assert!(!flock_data.codeword.is_empty());
    }
}
