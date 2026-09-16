//! `VerifyBitZ`.

use common::{LinearClaim, OpeningQuery, Root, VirtualMap, VirtualMapError, VirtualStatement};
use field::Fq;
use pcs::{CommitScheme, Pcs, StatementBinding, VerifyError as OpeningVerifyError};
use transcript::VerifierState;

use crate::{BitZVerifier, ReceiveError, ReduceError, reduce::gkr_reduce};

/// A proof the verifier rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The setup or PCS bit count differs from the virtual parameters.
    ParameterMismatch,
    /// The reduced claim cannot be transposed onto the committed bits.
    VirtualMap(VirtualMapError),
    /// The fold round failed its own checks.
    Fold(ReceiveError),
    /// The reduction failed.
    Reduction(ReduceError),
    /// The opening did not discharge the reduction's claim.
    Opening(OpeningVerifyError),
    /// A stream held bytes the protocol never read.
    TrailingData,
}

impl<const Q: u128> BitZVerifier<Q> {
    /// Verifies a claim on `h = M (1 || f)` against the commitment to `f`.
    ///
    /// Build the setup from `statement.params().claim()` and match the commitment's
    /// PCS parameters. GKR checks the input claim and returns an inner product on
    /// padded virtual bits. Supply the public circuit's map; its digest must cover
    /// its shape and entries.
    ///
    /// Start the transcript with the prover's session, instance, and public-input events.
    /// This method binds the inputs in [`VirtualStatement`], transposes the reduced
    /// claim, verifies the PCS opening, and rejects trailing proof or hint bytes.
    pub fn verify_virtual(
        &self,
        statement: &VirtualStatement<'_, Q, impl VirtualMap>,
        pcs: &Pcs,
        com: Root,
        mut transcript: VerifierState<'_>,
    ) -> Result<(), VerifyError> {
        let params = statement.params();
        let claim = statement.claim();
        if self.params() != params.claim()
            || pcs.bit_len() != 1 << params.committed_shape().log_bits()
        {
            return Err(VerifyError::ParameterMismatch);
        }
        transcript.public_message(b"bitz/virtual-statement/v1");
        transcript.public_message(&com.0);
        transcript.public_message(params);
        transcript.public_message(&statement.map().digest());
        transcript.public_message(claim);
        let query = self.fold_and_reduce(claim, &mut transcript)?;
        let query = statement
            .transpose_query(query)
            .map_err(VerifyError::VirtualMap)?;
        pcs.verify_lin(&com, &query, StatementBinding::Bind, &mut transcript)
            .map_err(VerifyError::Opening)?;
        transcript
            .check_eof()
            .map_err(|_| VerifyError::TrailingData)
    }

    /// Replays the proof of the caller's linear claim about the committed bits.
    ///
    /// `pcs` must be the scheme the commitment was made under. The transcript
    /// arrives carrying the caller's events; this appends and consumes it.
    pub fn verify(
        &self,
        claim: &LinearClaim<Fq<Q>>,
        pcs: &Pcs,
        com: Root,
        mut transcript: VerifierState<'_>,
    ) -> Result<(), VerifyError> {
        // Step 1: the admissibility and precondition checks have already run --
        // the shape gates in Shape::new, the modulus in Fq's own const assertions,
        // the generator's order in BitZParams::new and the weight counts in
        // LinearClaim::new. What is left is binding, before any challenge.
        transcript.public_message(&com.0);
        transcript.public_message(self.params());

        // Steps 3 and 4: check integer folds and replay GKR to obtain a bit claim.
        let query = self.fold_and_reduce(claim, &mut transcript)?;

        // Step 6: verify the inner-product sumcheck, ring switch, and opening.
        // Acceptance requires authenticating GKR's terminal claim against com.
        pcs.verify_lin(&com, &query, StatementBinding::Bind, &mut transcript)
            .map_err(VerifyError::Opening)?;

        // Both streams must be spent. Taking the transcript by value is what
        // makes that assertable here rather than by the caller.
        transcript
            .check_eof()
            .map_err(|_| VerifyError::TrailingData)?;

        Ok(())
    }

    /// Checks the column folds and derives GKR's factored bit claim.
    /// The caller binds the statement before this call and verifies the opening afterward.
    pub(crate) fn fold_and_reduce(
        &self,
        claim: &LinearClaim<field::Fq<Q>>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningQuery, VerifyError> {
        // Step 2 is absent: Q is fixed, and BitZParams::new checks its fold bound.

        // Step 3: read the folds, range-check them, reconstruct against mu.
        let fold = self
            .receive_fold(claim, transcript)
            .map_err(VerifyError::Fold)?;

        // Step 4: replay GKR from fold.e0 at fold.zeta to obtain the bit claim.
        // Step 5 needs no separate batching: fold.zeta already batches the columns.
        gkr_reduce(transcript, &fold, self.params().shape()).map_err(VerifyError::Reduction)
    }
}
