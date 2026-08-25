//! The fold round: read the column folds, check them, then take the
//! challenge.

use common::{CoreStatement, Fold, FoldError, reconstruct, row_images};
use field::{F128, FixedBasePow};
use transcript::VerifierState;

/// A fold the verifier rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiveError {
    /// A record is missing or does not decode.
    MalformedProof,
    /// The comb table was built on a different base than the statement's `g`.
    GeneratorMismatch,
    /// A fold is above `k_1 (q - 1)`, where its image stops determining it.
    FoldOutOfRange,
    /// The folds do not reconstruct the claimed target.
    TargetMismatch,
    /// The round's parts disagree about the shape.
    Fold(FoldError),
}

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
pub fn receive_fold<const Q: u128>(
    statement: &CoreStatement<Q>,
    generator: &FixedBasePow,
    transcript: &mut VerifierState<'_>,
) -> Result<Fold, ReceiveError> {
    // The statement's generator is checked for full order, but nothing else
    // ties it to the comb table the caller passes in. Without this a comb
    // built on any other base accepts a self-consistent proof of the wrong
    // claim. `pow(1)` reads the base straight out of the table.
    if generator.pow(1) != statement.generator() {
        return Err(ReceiveError::GeneratorMismatch);
    }
    let shape = statement.shape();

    let folds = (0..shape.columns())
        .map(|_| {
            transcript
                .prover_message::<[u8; 16]>()
                .map(u128::from_le_bytes)
                .map_err(|_| ReceiveError::MalformedProof)
        })
        .collect::<Result<Vec<_>, _>>()?;

    if folds.iter().any(|&fold| fold > statement.fold_bound()) {
        return Err(ReceiveError::FoldOutOfRange);
    }
    if reconstruct(statement, &folds).map_err(ReceiveError::Fold)? != statement.target() {
        return Err(ReceiveError::TargetMismatch);
    }

    let images: Vec<F128> = folds.iter().map(|&fold| generator.pow(fold)).collect();
    let row_images = row_images(statement, generator);
    let zeta = (0..shape.s())
        .map(|_| transcript.verifier_message())
        .collect();

    Fold::new(shape, folds, images, row_images, zeta).map_err(ReceiveError::Fold)
}
