//! Checked commit-phase wrapper for flock's binary-field PCS.

use std::sync::Mutex;

use flock_core::field::F128 as BackendF128;
use flock_core::hash::HashKind;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{
    Commitment as BackendCommitment, PcsParams, ProverData as BackendProverData, pack_witness,
};

use crate::CommitError;

#[cfg(not(target_endian = "little"))]
compile_error!("the pinned flock commitment requires a little-endian target");

const LOG_BATCH_SIZE: usize = 6;
const MIN_SUPPORTED_M: usize = 22;
const MAX_SUPPORTED_M: usize = 35;

/// The audited flock security profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityProfile {
    Fast,
    Slim,
    Secure,
}

impl SecurityProfile {
    pub(crate) fn backend(self) -> LigeritoProfile {
        match self {
            Self::Fast => LigeritoProfile::Fast,
            Self::Slim => LigeritoProfile::Slim,
            Self::Secure => LigeritoProfile::Secure,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Fast => 0,
            Self::Slim => 1,
            Self::Secure => 2,
        }
    }
}

/// The Merkle hash used by the commitment and all recursive openings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MerkleHash {
    Sha256,
    Blake3,
}

impl MerkleHash {
    pub(crate) fn backend(self) -> HashKind {
        match self {
            Self::Sha256 => HashKind::Sha256,
            Self::Blake3 => HashKind::Blake3,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Sha256 => 0,
            Self::Blake3 => 1,
        }
    }
}

/// Trusted public parameters for one PCS instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcsConfig {
    pub m: usize,
    pub log_inv_rate: usize,
    pub log_batch_size: usize,
    pub security_profile: SecurityProfile,
    pub merkle_hash: MerkleHash,
}

/// A checked PCS instance.
#[derive(Clone, Debug)]
pub struct Pcs {
    config: PcsConfig,
    params: PcsParams,
    bit_len: usize,
}

/// The public commitment. Parameters stay in the trusted [`Pcs`] instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment {
    root: [u8; 32],
}

/// Private state retained between commitment and one opening.
pub struct ProverData {
    bit_len: usize,
    packed_witness: Mutex<Option<Vec<BackendF128>>>,
    commitment: BackendCommitment,
    backend: BackendProverData,
}

impl Pcs {
    /// Creates an instance from an audited embedded flock profile.
    pub fn new(
        m: usize,
        security_profile: SecurityProfile,
        merkle_hash: MerkleHash,
    ) -> Result<Self, CommitError> {
        let profile = security_profile.backend();
        Self::from_config(PcsConfig {
            m,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: LOG_BATCH_SIZE,
            security_profile,
            merkle_hash,
        })
    }

    /// Creates an instance from explicit trusted parameters.
    pub fn from_config(config: PcsConfig) -> Result<Self, CommitError> {
        if !(MIN_SUPPORTED_M..=MAX_SUPPORTED_M).contains(&config.m)
            || config.log_batch_size != LOG_BATCH_SIZE
            || config.log_inv_rate != config.security_profile.backend().log_inv_rate()
            || config.m < 7 + config.log_batch_size
        {
            return Err(CommitError::InvalidConfiguration);
        }
        let bit_len = 1usize
            .checked_shl(u32::try_from(config.m).map_err(|_| CommitError::InvalidConfiguration)?)
            .ok_or(CommitError::InvalidConfiguration)?;
        let params = PcsParams {
            m: config.m,
            log_inv_rate: config.log_inv_rate,
            log_batch_size: config.log_batch_size,
            profile: config.security_profile.backend(),
            merkle_hash: config.merkle_hash.backend(),
        };
        params
            .ligerito_prover_config()
            .map_err(|_| CommitError::InvalidConfiguration)?;
        params
            .ligerito_verifier_config()
            .map_err(|_| CommitError::InvalidConfiguration)?;
        Ok(Self {
            config,
            params,
            bit_len,
        })
    }

    /// Commits to the exact configured number of bits.
    pub fn commit(&self, bits: &[bool]) -> Result<(Commitment, ProverData), CommitError> {
        if bits.len() != self.bit_len {
            return Err(CommitError::InvalidBitLength { len: bits.len() });
        }
        let packed_witness = pack_witness(bits, self.params.m);
        let (commitment, backend) = flock_core::pcs::commit(&packed_witness, &self.params);
        Ok((
            Commitment {
                root: commitment.root,
            },
            ProverData {
                bit_len: self.bit_len,
                packed_witness: Mutex::new(Some(packed_witness)),
                commitment,
                backend,
            },
        ))
    }

    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    pub fn config(&self) -> PcsConfig {
        self.config
    }

    pub(crate) fn params(&self) -> &PcsParams {
        &self.params
    }

    pub(crate) fn statement_tags(&self) -> [u64; 5] {
        [
            self.config.m as u64,
            self.config.log_inv_rate as u64,
            self.config.log_batch_size as u64,
            self.config.security_profile.tag() as u64,
            self.config.merkle_hash.tag() as u64,
        ]
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

    pub(crate) fn backend(&self, params: &PcsParams) -> BackendCommitment {
        BackendCommitment {
            root: self.root,
            params: params.clone(),
        }
    }
}

impl ProverData {
    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    pub fn packed_len(&self) -> usize {
        self.packed_witness
            .lock()
            .expect("packed-witness mutex poisoned")
            .as_ref()
            .map_or(0, Vec::len)
    }

    pub fn codeword_len(&self) -> usize {
        self.backend.codeword.len()
    }

    pub(crate) fn take_packed_witness(&self) -> Result<Vec<BackendF128>, CommitError> {
        self.packed_witness
            .lock()
            .map_err(|_| CommitError::Backend)?
            .take()
            .ok_or(CommitError::ProverDataConsumed)
    }

    pub(crate) fn commitment(&self) -> &BackendCommitment {
        &self.commitment
    }

    pub(crate) fn backend(&self) -> &BackendProverData {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unregistered_shapes() {
        for m in [0, 21, 36, usize::MAX] {
            assert!(Pcs::new(m, SecurityProfile::Fast, MerkleHash::Blake3).is_err());
        }
    }

    #[test]
    fn rejects_the_wrong_bit_length_before_flock() {
        let scheme = Pcs::new(22, SecurityProfile::Fast, MerkleHash::Blake3).unwrap();
        assert!(matches!(
            scheme.commit(&[false; 128]),
            Err(CommitError::InvalidBitLength { len: 128 })
        ));
    }

    #[test]
    fn commitment_is_deterministic_and_retains_the_packed_witness() {
        let scheme = Pcs::new(22, SecurityProfile::Fast, MerkleHash::Blake3).unwrap();
        let mut bits = vec![false; scheme.bit_len()];
        for index in [0, 1, 63, 64, 127, 128, bits.len() - 1] {
            bits[index] = true;
        }
        let (first, data) = scheme.commit(&bits).unwrap();
        let (second, _) = scheme.commit(&bits).unwrap();
        assert_eq!(first, second);
        assert_eq!(data.bit_len(), bits.len());
        assert_eq!(data.packed_len(), bits.len() / 128);
        assert!(data.codeword_len() > 0);
    }

    #[test]
    fn packed_witness_moves_once_without_a_clone() {
        let scheme = Pcs::new(22, SecurityProfile::Fast, MerkleHash::Blake3).unwrap();
        let (_, data) = scheme.commit(&vec![false; scheme.bit_len()]).unwrap();
        assert_eq!(
            data.take_packed_witness().unwrap().len(),
            scheme.bit_len() / 128
        );
        assert_eq!(
            data.take_packed_witness(),
            Err(CommitError::ProverDataConsumed)
        );
    }
}
