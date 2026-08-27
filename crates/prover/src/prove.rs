//! `ProveF2Z`.

use common::{BitTable, LinearClaim, OpeningClaim, ReductionInput, Root};
use transcript::ProverState;

use crate::{F2ZProver, SendError};

/// A proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError<E> {
    /// The fold round failed.
    Fold(SendError),
    /// The reduction failed.
    Reduction(E),
}

/// Step 4: the grand product, and the sumcheck that turns its affine leaf
/// into a claim on the committed bits.
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

impl<const Q: u128> F2ZProver<Q> {
    /// Proves the caller's linear claim about the committed bits.
    ///
    /// The caller commits first, and passes the root in as `com`. The
    /// transcript arrives carrying the caller's events; this appends and hands
    /// it back.
    pub fn prove<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<Q>,
        com: Root,
        table: &BitTable<'_>,
        reduction: &R,
        transcript: &mut ProverState,
    ) -> Result<OpeningClaim, ProveError<R::Error>> {
        // Step 1: bind. Absorbing the root here is not redundant with the opening
        // scheme, whose batched opening binds it only in its own statement mode --
        // and that fires at step 6, long after the fold has squeezed.
        //
        // The claim itself is not bound: neither `v^(1)`, `v^(2)` nor `mu` reaches
        // the sponge here, only the parameters. They enter through the caller's
        // own events.
        transcript.public_message(&com.0);
        transcript.public_message(self.params());

        // TODO: step 2, reducing the modulus, is absent. It runs when q is too
        // large for the shape, and a const modulus parameter cannot express its
        // `q <- q'`. Callers must supply an admissible q; LinearClaim::new
        // rejects anything else.

        // Step 3: fold each column into an integer exponent.
        let fold = self
            .send_fold(claim, table, transcript)
            .map_err(ProveError::Fold)?;

        // Step 4: the grand product over the folds, then the sumcheck that
        // turns its affine leaf into a claim on the committed bits.
        //
        // TODO(#8): both live behind `Reduction`, which nothing implements yet.
        let input = ReductionInput {
            params: self.params(),
            claim,
            commitment: com,
            fold: &fold,
        };
        let claim = reduction
            .reduce(&input, table, transcript)
            .map_err(ProveError::Reduction)?;

        // Step 5, batching the per-column claims, is conditional and a
        // merged-forest grand product does not need it: it draws its challenge
        // once across all columns, so the claims never separate.

        // TODO(#15): step 6, the ring switch and the opening, is absent, so this
        // hands back the claim the opening would consume instead of consuming it
        // -- a caller that stops here has proved nothing. The tail becomes
        // `prove_lin_batch(data, packed, &[claim], AlreadyBound, transcript)`.
        Ok(claim)
    }
}
