//! `ProveBitZ`.

use common::{
    BitTable, LinearClaim, OpeningQuery, TableError, VirtualMap, VirtualMapError, VirtualStatement,
};
use field::{F128, Fq};
use pcs::{CommitScheme, Pcs, ProveError as OpeningProveError, ProverData, StatementBinding};
use transcript::ProverState;

use crate::{BitZProver, SendError, reduce::ReduceError, reduce::gkr_reduce};

/// A proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The setup or PCS bit count differs from the virtual parameters.
    ParameterMismatch,
    /// The reduced claim cannot be transposed onto the committed bits.
    VirtualMap(VirtualMapError),
    /// The witness is not the length the shape calls for.
    Witness(TableError),
    /// The fold round failed.
    Fold(SendError),
    /// The GKR left no claim.
    Reduction(ReduceError),
    /// The opening failed, so the reduction's claim was never discharged.
    Opening(OpeningProveError),
}

/// Packed committed bits `f` and virtual bits `h = M (1 || f)`.
///
/// The caller zero-pads each vector to its shape in [`common::VirtualParams`]. Bit `i` is bit
/// `i % 128` of element `i / 128`, with the low 64 bits in `lo` and the rest in `hi`.
#[derive(Debug)]
pub struct VirtualWitness<'a> {
    /// The commitment opening consumes the committed witness.
    pub committed_bits: Vec<F128>,
    /// Folding and reduction borrow the virtual witness.
    pub virtual_bits: &'a [F128],
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
    #[tracing::instrument(name = "Prove BitZ", skip_all)]
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

    /// Proves a claim on `h = M (1 || f)` against the commitment to `f`.
    ///
    /// Build the setup from `statement.params().claim()`. Commit `witness.committed_bits`
    /// with `pcs` under the committed shape and pass its returned `data`. GKR reduces
    /// the input claim to an inner product on padded virtual bits. This method
    /// transposes its coefficients before PCS opens the committed bits.
    ///
    /// Initialize the transcript with the session, instance, and enclosing public inputs.
    /// This method binds the inputs listed in [`VirtualStatement`].
    #[tracing::instrument(name = "Prove virtual BitZ", skip_all)]
    pub fn prove_virtual(
        &self,
        statement: &VirtualStatement<'_, Q, impl VirtualMap>,
        pcs: &Pcs,
        data: &ProverData,
        witness: VirtualWitness<'_>,
        transcript: &mut ProverState,
    ) -> Result<(), ProveError> {
        let params = statement.params();
        let claim = statement.claim();
        if self.params() != params.claim()
            || pcs.bit_len() != 1 << params.committed_shape().log_bits()
        {
            return Err(ProveError::ParameterMismatch);
        }
        // Validate the committed witness length before binding the transcript.
        params
            .table(&witness.committed_bits)
            .map_err(ProveError::Witness)?;
        let table = self
            .params()
            .table(witness.virtual_bits)
            .map_err(ProveError::Witness)?;
        let com = data.root();

        // Step 1: bind the virtual statement before drawing fold challenges.
        // The domain separates this proof from a direct proof on committed bits.
        transcript.public_message(b"bitz/virtual-statement/v1");
        transcript.public_message(&com.0);
        transcript.public_message(params);
        transcript.public_message(&statement.map().digest());
        transcript.public_message(claim);

        // Steps 3 and 4: fold the virtual columns and reduce them through GKR.
        // Modulus reduction is absent; the parameters must already be admissible.
        let query = self.fold_and_reduce(claim, &table, transcript)?;

        // Transpose the reduced claim from h to f, since the PCS commits to f.
        let query = statement
            .transpose_query(query)
            .map_err(ProveError::VirtualMap)?;

        // Step 6: the post-GKR sumcheck, ring switching, and opening on committed bits.
        // Bind the PCS parameters and transposed query before its challenges.
        pcs.prove_lin(
            data,
            witness.committed_bits,
            &query,
            StatementBinding::Bind,
            transcript,
        )
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
