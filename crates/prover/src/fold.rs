//! The fold round: send the column folds, then take the challenge.

use common::{BitTable, F2ZConfig, Fold, FoldError, LinearClaim, fold_column, row_images};
use field::F128;
use transcript::ProverState;

/// A fold the prover cannot produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError {
    /// The witness is not shaped the way the configuration says it is.
    ShapeMismatch,
    /// The round's parts disagree about the shape.
    Fold(FoldError),
}

/// Runs the fold round.
///
/// Only the folds `eta_j` are sent. Their images `g^{eta_j}` are what the
/// grand product consumes, but the verifier derives them for itself, so
/// transmitting both would be sending the same information twice: at
/// `k_2 = 2^21` that second copy is 32 MiB.
///
/// The folds go out on the narg channel, which writes and absorbs the same
/// bytes, so the challenge below depends on them. That binds the images just
/// as tightly, because `u -> g^u` is injective over the range the verifier
/// admits — which is exactly what the range check and the full order of `g`
/// establish.
///
/// Each fold is one record. The count is not itself absorbed, which is safe
/// only because it derives from the configured shape; the wire profile is
/// where a length-delimited vector record would belong.
pub fn send_fold<const Q: u128>(
    config: &F2ZConfig<Q>,
    claim: &LinearClaim<Q>,
    table: &BitTable<'_>,
    transcript: &mut ProverState,
) -> Result<Fold, SendError> {
    // The weights are sized by the configured row count while the bits are
    // read at the table's. Disagreement is a panic, a silently wrong fold, or
    // a desynchronised transcript depending on which way it goes.
    if table.shape() != config.shape() {
        return Err(SendError::ShapeMismatch);
    }
    let shape = config.shape();

    // Lifted once: the fold reads it per set bit across every column.
    let exponents = claim.row_exponents();
    let folds: Vec<u128> = (0..shape.columns())
        .map(|column| fold_column(table, &exponents, column))
        .collect();

    for fold in &folds {
        transcript.prover_message(&fold.to_le_bytes());
    }

    let images: Vec<F128> = folds.iter().map(|&fold| config.comb().pow(fold)).collect();
    let row_images = row_images(config, claim);
    let zeta = (0..shape.s())
        .map(|_| transcript.verifier_message())
        .collect();

    Fold::new(shape, folds, images, row_images, zeta).map_err(SendError::Fold)
}
