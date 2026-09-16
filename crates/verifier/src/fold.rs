//! The fold round: read the column folds, check them, then take the
//! challenge.

use common::{Fold, FoldError, LinearClaim, column_images, reconstruct, row_images};

use crate::BitZVerifier;
use transcript::VerifierState;

/// A fold the verifier rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiveError {
    /// A record is missing or does not decode.
    MalformedProof,
    /// A fold is above `k_1 (q - 1)`, where its image stops determining it.
    FoldOutOfRange,
    /// The folds do not reconstruct the claimed target.
    TargetMismatch,
    /// The round's parts disagree about the shape.
    Fold(FoldError),
}

impl<const Q: u128> BitZVerifier<Q> {
    /// Reads the fold round and checks it.
    ///
    /// The proof carries only the folds; their images are derived here rather than
    /// transmitted, so there is no image to disagree with a fold and no check that
    /// they match. What replaces that check is the range bound: `u -> g^u` is
    /// injective only over `[0, k_1(q-1)]`, so a fold outside it would not
    /// determine the image the grand product then proves.
    ///
    /// The obligations fire in the order the protocol fixes — every fold is
    /// range-checked, then the reconstruction ties them back to `mu`, and only
    /// then is the challenge drawn. Reading a record absorbs it, so the folds are
    /// in the sponge before either check; what the ordering protects is the
    /// challenge, which must not be reachable until both have passed.
    #[tracing::instrument(name = "Verify column folds", skip_all)]
    pub fn receive_fold(
        &self,
        claim: &LinearClaim<field::Fq<Q>>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<Fold, ReceiveError> {
        let shape = self.params().shape();

        let folds = (0..shape.columns())
            .map(|_| {
                transcript
                    .prover_message::<[u8; 16]>()
                    .map(u128::from_le_bytes)
                    .map_err(|_| ReceiveError::MalformedProof)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if folds.iter().any(|&fold| fold > self.fold_bound()) {
            return Err(ReceiveError::FoldOutOfRange);
        }
        if reconstruct(claim, &folds).map_err(ReceiveError::Fold)? != claim.target() {
            return Err(ReceiveError::TargetMismatch);
        }

        let images = column_images(self.comb(), &folds);
        let row_images = row_images(self.comb(), &claim.row_exponents());
        let zeta = (0..shape.log_columns())
            .map(|_| transcript.verifier_message())
            .collect();

        Fold::new(shape, folds, images, row_images, zeta).map_err(ReceiveError::Fold)
    }
}
