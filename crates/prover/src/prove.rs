//! `ProveBitZ`.

use common::{BitTable, ClaimError, LinearClaim, OpeningQuery, TableError};
use field::{F128, Fq};
use pcs::{CommitScheme, Pcs, ProveError as OpeningProveError, ProverData, StatementBinding};
use transcript::ProverState;

use crate::{BitZProver, SendError, reduce::gkr_reduce};

/// A proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The witness is not the length the shape calls for.
    Witness(TableError),
    /// The fold round failed.
    Fold(SendError),
    /// The derived GKR weight counts do not match the table shape.
    Reduction(ClaimError),
    /// The opening failed, so the reduction's claim was never discharged.
    Opening(OpeningProveError),
}

impl<const Q: u128> BitZProver<Q> {
    /// Proves the caller's linear claim about the committed bits.
    ///
    /// The caller commits first and passes what that produced: the `data` the
    /// opening reads and the packed witness itself. The root is read back off
    /// `data` rather than passed alongside it, so the two cannot disagree.
    /// `pcs` must be the scheme that committed, or the opening will not verify.
    ///
    /// The witness arrives owned because the opening consumes it. The transcript
    /// arrives carrying the caller's events; this appends and hands it back.
    pub fn prove(
        &self,
        claim: &LinearClaim<Fq<Q>>,
        pcs: &Pcs,
        data: &ProverData,
        packed: Vec<F128>,
        transcript: &mut ProverState,
    ) -> Result<(), ProveError> {
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

        // Steps 3 and 4: integer column folds, then GKR to a factored bit claim.
        let query = self.fold_and_reduce(claim, &table, transcript)?;

        // Step 6: inner-product sumcheck, ring switching, and commitment opening.
        // Bind the derived query and PCS parameters before the opening challenges.
        pcs.prove_lin(data, packed, &query, StatementBinding::Bind, transcript)
            .map_err(ProveError::Opening)
    }

    /// Folds the columns and reduces their products to a claim about `table`.
    /// The caller binds the statement before this call and opens the returned claim.
    pub(crate) fn fold_and_reduce(
        &self,
        claim: &LinearClaim<field::Fq<Q>>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
    ) -> Result<OpeningQuery, ProveError> {
        // Step 2 is absent: Q is fixed, and BitZParams::new checks its fold bound.

        // Step 3: fold each column into an integer exponent.
        let fold = self
            .send_fold(claim, table, transcript)
            .map_err(ProveError::Fold)?;

        // Step 4: GKR reduces the batched column products to a factored bit claim.
        // Step 5 needs no separate batching: fold.zeta already batches the columns.
        gkr_reduce(transcript, &fold, table).map_err(ProveError::Reduction)
    }
}
