//! `VerifyBitZ`.

use std::collections::VecDeque;

use common::{LinearClaim, OpeningQuery, ReductionInput, Root};
use field::Fq;
use pcs::{CommitScheme, Pcs, StatementBinding, VerifyError as OpeningVerifyError};
use transcript::VerifierState;

use crate::{BitZVerifier, ReceiveError};

/// A proof the verifier rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError<E> {
    /// The fold round failed its own checks.
    Fold(ReceiveError),
    /// The reduction failed.
    Reduction(E),
    /// The opening did not discharge the reduction's claim.
    Opening(OpeningVerifyError),
    /// A stream held bytes the protocol never read.
    TrailingData,
}

/// Step 4, replayed.
///
/// TODO: #8 implements this.
pub trait Reduction<const Q: u128> {
    type Error;

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningQuery, Self::Error>;
}

impl<const Q: u128> BitZVerifier<Q> {
    /// Replays the proof of the caller's linear claim about the committed bits.
    ///
    /// `pcs` must be the scheme the commitment was made under. The transcript
    /// arrives carrying the caller's events; this appends and consumes it.
    pub fn verify<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<Fq<Q>>,
        pcs: &Pcs,
        com: Root,
        reduction: &R,
        mut transcript: VerifierState<'_>,
    ) -> Result<(), VerifyError<R::Error>> {
        // Step 1: the admissibility and precondition checks have already run --
        // the shape gates in Shape::new, the modulus in Fq's own const assertions,
        // the generator's order in BitZParams::new and the weight counts in
        // LinearClaim::new. What is left is binding, before any challenge.
        transcript.public_message(&com.0);
        transcript.public_message(self.params());

        // Steps 2 to 5, replayed.
        let query = self.fold_and_reduce(claim, com, reduction, &mut transcript)?;
        // TODO: step 2, reducing the modulus, is absent, as on the prover.

        // Step 3: read the folds, range-check them, reconstruct against mu.
        let fold = self
            .receive_fold(claim, &mut transcript)
            .map_err(VerifyError::Fold)?;

        // Step 4, replayed.
        // build input based on row_image and the bit table.
        let circuit = _;
        let mut point = fold.zeta.clone();
        point.reverse();
        let point = VecDeque::from(point);

        gkr::gpgkr_verify(&mut transcript, fold.e0, point, pcs);
        // TODO(#8): both live behind `Reduction`, which nothing implements yet.
        let input = ReductionInput {
            params: self.params(),
            claim,
            commitment: com,
            fold: &fold,
        };
        let query = reduction
            .reduce(&input, &mut transcript)
            .map_err(VerifyError::Reduction)?;

        // Step 5 is conditional and a merged forest does not need it.

        // Step 6, replayed. This is what makes the return an acceptance rather
        // than a claim handed back undischarged, so it belongs above the
        // exhaustion check and not after it.
        pcs.verify_lin(&com, &query, StatementBinding::Bind, &mut transcript)
            .map_err(VerifyError::Opening)?;

        // Both streams must be spent. Taking the transcript by value is what
        // makes that assertable here rather than by the caller.
        transcript
            .check_eof()
            .map_err(|_| VerifyError::TrailingData)?;

        Ok(())
    }

    /// Steps 2 to 5, which every entry point replays identically.
    ///
    /// Step 1 and step 6 stay with the caller: it absorbs its own statement
    /// frame and decides what to do with the query, which is a claim about the
    /// vector the fold ran over, not always the one the oracle commits to.
    pub(crate) fn fold_and_reduce<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<field::Fq<Q>>,
        com: Root,
        reduction: &R,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningQuery, VerifyError<R::Error>> {
        // TODO: step 2, reducing the modulus, is absent, as on the prover.

        // Step 3: read the folds, range-check them, reconstruct against mu.
        let fold = self
            .receive_fold(claim, transcript)
            .map_err(VerifyError::Fold)?;

        // Step 4, replayed.
        //
        // TODO(#8): both live behind `Reduction`, which nothing implements yet.
        let input = ReductionInput {
            params: self.params(),
            claim,
            commitment: com,
            fold: &fold,
        };

        // Step 5 is conditional and a merged forest does not need it.
        reduction
            .reduce(&input, transcript)
            .map_err(VerifyError::Reduction)
    }
}
