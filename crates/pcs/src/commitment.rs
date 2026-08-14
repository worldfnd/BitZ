//! Checked commit-phase wrapper for flock's binary-field PCS.

use crate::CommitError;
use flock_core::field::F128 as FlockF128;
pub use flock_core::hash::HashKind;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{LOG_PACKING, PcsParams, ProverData as FlockProverData, pack_witness};

/// A checked PCS instance.
#[derive(Clone, Debug)]
pub struct Pcs {
    params: PcsParams,
    // number of bits we commit to
    bit_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment {
    root: [u8; 32],
}

/// Private state retained between commitment and one opening.
pub struct ProverData {
    // length of witness before packing
    bit_len: usize,
    commitment: Commitment,
    packed_witness: Vec<FlockF128>,
    flock_prover_data: FlockProverData,
}

impl Pcs {
    pub fn new(m: usize, security_profile: LigeritoProfile, merkle_hash: HashKind) -> Self {
        Self {
            params: PcsParams {
                m: m,
                log_inv_rate: security_profile.log_inv_rate(),
                log_batch_size: LOG_PACKING,
                profile: security_profile,
                merkle_hash: merkle_hash,
            },
            bit_len: 1 << m,
        }
    }

    /// Commits to the exact configured number of bits.
    pub fn commit(&self, bits: &[bool]) -> Result<(Commitment, ProverData), CommitError> {
        if bits.len() != self.bit_len {
            return Err(CommitError::InvalidBitLength { len: bits.len() });
        }
        let packed_witness = pack_witness(bits, self.params.m);
        let (commitment, flock_prover_data) =
            flock_core::pcs::commit(&packed_witness, &self.params);
        Ok((
            Commitment {
                root: commitment.root,
            },
            ProverData {
                bit_len: self.bit_len,
                commitment: Commitment {
                    root: commitment.root,
                },
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

    pub fn commitment(&self) -> &Commitment {
        &self.commitment
    }
}

impl Commitment {
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
            Err(CommitError::InvalidBitLength { len: 128 })
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
        let (packed_witness, backend) = data.1.into_opening_parts();
        assert_eq!(packed_witness.len(), scheme.bit_len() / 128);
        assert!(!backend.codeword.is_empty());
    }
}
