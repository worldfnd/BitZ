//! `ProveF2Z`.

use common::{BitTable, LinearClaim, OpeningQuery, ReductionInput, TableError};
use field::F128;
use pcs::{CommitScheme, Pcs, ProveError as OpeningProveError, ProverData, StatementBinding};
use transcript::ProverState;

use crate::{F2ZProver, SendError};

/// A proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError<E> {
    /// The witness is not the length the shape calls for.
    Witness(TableError),
    /// The fold round failed.
    Fold(SendError),
    /// The reduction failed.
    Reduction(E),
    /// The opening failed, so the reduction's claim was never discharged.
    Opening(OpeningProveError),
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
    ) -> Result<OpeningQuery, Self::Error>;
}

impl<const Q: u128> F2ZProver<Q> {
    /// Proves the caller's linear claim about the committed bits.
    ///
    /// The caller commits first and passes what that produced: the `data` the
    /// opening reads and the packed witness itself. The root is read back off
    /// `data` rather than passed alongside it, so the two cannot disagree.
    /// `pcs` must be the scheme that committed, or the opening will not verify.
    ///
    /// The witness arrives owned because the opening consumes it. The transcript
    /// arrives carrying the caller's events; this appends and hands it back.
    pub fn prove<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<Q>,
        pcs: &Pcs,
        data: &ProverData,
        packed: Vec<F128>,
        reduction: &R,
        transcript: &mut ProverState,
    ) -> Result<(), ProveError<R::Error>> {
        let com = data.root();
        let table = self.params().table(&packed).map_err(ProveError::Witness)?;

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
            .send_fold(claim, &table, transcript)
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
        let query = reduction
            .reduce(&input, &table, transcript)
            .map_err(ProveError::Reduction)?;

        // Step 5, batching the per-column claims, is conditional and a
        // merged-forest grand product does not need it: it draws its challenge
        // once across all columns, so the claims never separate.

        // Step 6: the ring switch and the opening, which discharge the claim
        // step 4 handed over. `Bind` rather than `AlreadyBound`: the opening's
        // own parameters are not in the frame step 1 absorbed, and binding them
        // here is what puts them in the sponge before the opener's first
        // squeeze.
        pcs.prove_lin(data, packed, &query, StatementBinding::Bind, transcript)
            .map_err(ProveError::Opening)
    }
}
