//! Shared shape and storage for query-specific ring-switch claims.

use flock_core::field::F128 as FlockF128;
use flock_core::pcs::LOG_PACKING;

use crate::CommitError;

pub(crate) const CLAIM_COUNT: usize = 1 << LOG_PACKING;
const _: () = assert!(CLAIM_COUNT == 128);

/// The fixed 128-value claim container shared by both ring switches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Claims([FlockF128; CLAIM_COUNT]);

impl Claims {
    pub(crate) fn from_array(values: [FlockF128; CLAIM_COUNT]) -> Self {
        Self(values)
    }

    /// Copies proof values into the fixed protocol shape.
    pub(crate) fn from_proof(values: &[FlockF128]) -> Result<Self, CommitError> {
        let values: &[FlockF128; CLAIM_COUNT] = values
            .try_into()
            .map_err(|_| CommitError::VerificationFailed)?;
        Ok(Self(*values))
    }

    pub(crate) fn as_array(&self) -> &[FlockF128; CLAIM_COUNT] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_claim_shape_rejects_other_lengths() {
        assert_eq!(
            Claims::from_proof(&vec![FlockF128::ZERO; CLAIM_COUNT - 1]),
            Err(CommitError::VerificationFailed),
        );
    }
}
