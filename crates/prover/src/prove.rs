//! `ProveF2Z`.

use common::{BitTable, CoreStatement, F2ZConfig, OpeningClaim, ReductionInput, Root};
use transcript::ProverState;

use crate::{SendError, send_fold};

/// A proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError<E> {
    /// The fold round failed.
    Fold(SendError),
    /// The reduction failed.
    Reduction(E),
}

/// Steps 5.2 and 5.2a: the grand product, and the sumcheck that turns its
/// affine leaf into a claim on the committed bits.
///
/// TODO: #8 implements this. Until then the round trip stubs it, so the
/// transcript order below is exercised and the reduction's argument is not.
pub trait Reduction<const Q: u128> {
    type Error;

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
    ) -> Result<OpeningClaim, Self::Error>;
}

/// Proves the caller's linear claim about the committed bits.
///
/// The caller commits first — it packs the witness with
/// [`common::BitTable::pack`] and passes the root in as `com`. The transcript
/// arrives carrying the caller's events; this appends and hands it back.
pub fn prove<const Q: u128, R: Reduction<Q>>(
    config: &F2ZConfig<Q>,
    statement: &CoreStatement<Q>,
    com: Root,
    table: &BitTable<'_>,
    reduction: &R,
    transcript: &mut ProverState,
) -> Result<OpeningClaim, ProveError<R::Error>> {
    // Step 1: bind. Absorbing the root here is not redundant with the opening
    // scheme, whose batched opening binds it only in its own statement mode --
    // and that fires at Step 5.3, long after the fold has squeezed.
    //
    // The claim itself is not bound: neither `v^(1)`, `v^(2)` nor `mu` reaches
    // the sponge here, only the parameters. They enter through the caller's
    // own events.
    transcript.public_message(&com.0);
    transcript.public_message(config);

    // TODO: Step 5.0, reduce the modulus, is absent. It runs when q is too
    // large for the shape, and a const modulus parameter cannot express its
    // `q <- q'`. Callers must supply an admissible q; CoreStatement::new
    // rejects anything else.

    // Step 5.1: fold each column into an integer exponent.
    let fold = send_fold(config, statement, table, transcript).map_err(ProveError::Fold)?;

    // Steps 5.2 and 5.2a: the grand product over the folds, then the sumcheck
    // that turns its affine leaf into a claim on the committed bits.
    //
    // TODO(#8): both live behind `Reduction`, which nothing implements yet.
    let input = ReductionInput {
        config,
        statement,
        commitment: com,
        fold: &fold,
    };
    let claim = reduction
        .reduce(&input, table, transcript)
        .map_err(ProveError::Reduction)?;

    // Step 5.2b, batching the per-column claims, is conditional and a
    // merged-forest grand product does not need it: it draws its challenge
    // once across all columns, so the claims never separate.

    // TODO(#15): Step 5.3, the ring switch and the opening, is absent, so this
    // hands back the claim the opening would consume instead of consuming it
    // -- a caller that stops here has proved nothing. The tail becomes
    // `prove_lin_batch(data, packed, &[claim], AlreadyBound, transcript)`.
    Ok(claim)
}
