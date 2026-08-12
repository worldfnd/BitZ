//! Commit-phase wrapper for flock's binary-field PCS.

use std::sync::Mutex;

use flock_core::field::F128 as FlockF128;
use flock_core::hash::HashKind;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::pcs::{Commitment, PcsParams, ProverData, pack_witness};

use crate::CommitError;

#[cfg(not(target_endian = "little"))]
compile_error!("the pinned flock commitment requires a little-endian target");

/// The row-batch size required by every embedded flock profile at the pin.
const LOG_BATCH_SIZE: usize = 6;
const MIN_SUPPORTED_M: usize = 22;
const MAX_SUPPORTED_M: usize = 35;

/// A checked commit-phase configuration for the pinned flock backend.
///
/// `prove_lin` and `verify_lin` will complete the [`crate::CommitScheme`]
/// implementation in the opening phase.
#[derive(Clone, Debug)]
pub struct FlockScheme {
    params: PcsParams,
    bit_len: usize,
}

/// The public flock Merkle commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlockCommitment {
    root: [u8; 32],
}

/// Private state retained between commitment and opening.
///
/// Flock does not retain the packed witness. This wrapper retains it without
/// another representation conversion or allocation.
pub struct FlockProverData {
    bit_len: usize,
    packed_witness: Mutex<Option<Vec<FlockF128>>>,
    commitment: Commitment,
    backend: ProverData,
}

impl FlockScheme {
    /// Creates a scheme from trusted public parameters.
    ///
    /// Flock provides audited opening profiles for `m` values from 22 to 35.
    pub fn new(
        m: usize,
        profile: LigeritoProfile,
        merkle_hash: HashKind,
    ) -> Result<Self, CommitError> {
        if !(MIN_SUPPORTED_M..=MAX_SUPPORTED_M).contains(&m) {
            return Err(CommitError::InvalidConfiguration);
        }
        let bit_len = 1usize
            .checked_shl(u32::try_from(m).map_err(|_| CommitError::InvalidConfiguration)?)
            .ok_or(CommitError::InvalidConfiguration)?;
        let params = PcsParams {
            m,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: LOG_BATCH_SIZE,
            profile,
            merkle_hash,
        };

        // These calls validate the embedded profile and the recursion shape.
        params
            .ligerito_prover_config()
            .map_err(|_| CommitError::InvalidConfiguration)?;
        params
            .ligerito_verifier_config()
            .map_err(|_| CommitError::InvalidConfiguration)?;

        Ok(Self { params, bit_len })
    }

    /// Commits to `bits` through flock's optimized `Pack_128` implementation.
    pub fn commit(&self, bits: &[bool]) -> Result<(FlockCommitment, FlockProverData), CommitError> {
        if bits.len() != self.bit_len {
            return Err(CommitError::InvalidBitLength { len: bits.len() });
        }

        // Flock reads the bool slice as bytes, packs in parallel, and returns
        // its native F128 vector. No local F128 vector or representation copy
        // exists on this path.
        let packed_witness = pack_witness(bits, self.params.m);
        let (commitment, backend) = flock_core::pcs::commit(&packed_witness, &self.params);

        Ok((
            FlockCommitment {
                root: commitment.root,
            },
            FlockProverData {
                bit_len: self.bit_len,
                packed_witness: Mutex::new(Some(packed_witness)),
                commitment,
                backend,
            },
        ))
    }

    /// Returns the committed bit length.
    pub fn bit_len(&self) -> usize {
        self.bit_len
    }
}

impl FlockCommitment {
    /// Returns the Merkle root.
    pub fn root(&self) -> &[u8; 32] {
        &self.root
    }
}

impl FlockProverData {
    /// Returns the committed bit length.
    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Returns the number of packed field elements.
    pub fn packed_len(&self) -> usize {
        self.packed_witness
            .lock()
            .expect("packed-witness mutex poisoned")
            .as_ref()
            .map_or(0, Vec::len)
    }

    /// Returns the retained encoded codeword length.
    pub fn codeword_len(&self) -> usize {
        self.backend.codeword.len()
    }

    /// Moves the packed witness into flock's one-shot opening prover.
    #[inline]
    #[allow(dead_code, reason = "used by the linear-opening implementation")]
    pub(crate) fn take_packed_witness(&self) -> Result<Vec<FlockF128>, CommitError> {
        self.packed_witness
            .lock()
            .map_err(|_| CommitError::Backend)?
            .take()
            .ok_or(CommitError::ProverDataConsumed)
    }

    #[allow(dead_code, reason = "used by the linear-opening implementation")]
    pub(crate) fn commitment(&self) -> &Commitment {
        &self.commitment
    }

    #[allow(dead_code, reason = "used by the linear-opening implementation")]
    pub(crate) fn backend(&self) -> &ProverData {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unregistered_shapes() {
        assert!(FlockScheme::new(0, LigeritoProfile::Fast, HashKind::Blake3).is_err());
        assert!(FlockScheme::new(21, LigeritoProfile::Fast, HashKind::Blake3).is_err());
        assert!(FlockScheme::new(36, LigeritoProfile::Fast, HashKind::Blake3).is_err());
        assert!(FlockScheme::new(usize::MAX, LigeritoProfile::Fast, HashKind::Blake3).is_err());
    }

    #[test]
    fn rejects_the_wrong_bit_length_before_flock() {
        let scheme = FlockScheme::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        assert!(matches!(
            scheme.commit(&[false; 128]),
            Err(CommitError::InvalidBitLength { len: 128 })
        ));
    }

    #[test]
    fn commitment_is_deterministic_and_retains_the_packed_witness() {
        let scheme = FlockScheme::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
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
        let packed = data.packed_witness.lock().unwrap();
        let packed = packed.as_ref().unwrap();
        assert_eq!(
            packed[0],
            FlockF128::new((1 << 0) | (1 << 1) | (1 << 63), (1 << 0) | (1 << 63))
        );
        assert_eq!(packed[1], FlockF128::new(1, 0));
        assert_eq!(packed[packed.len() - 1], FlockF128::new(0, 1 << 63));
    }

    #[test]
    fn packed_witness_moves_once_without_a_clone() {
        let scheme = FlockScheme::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let (_, data) = scheme.commit(&vec![false; scheme.bit_len()]).unwrap();
        let packed = data.take_packed_witness().unwrap();

        assert_eq!(packed.len(), scheme.bit_len() / 128);
        assert_eq!(
            data.take_packed_witness(),
            Err(CommitError::ProverDataConsumed)
        );
    }
}
