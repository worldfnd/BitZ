//! `VerifyF2Z`.

use common::{LinearClaim, OpeningClaim, ReductionInput, Root};
use transcript::VerifierState;

use crate::{F2ZVerifier, ReceiveError};

/// A proof the verifier rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError<E> {
    /// The fold round failed its own checks.
    Fold(ReceiveError),
    /// The reduction failed.
    Reduction(E),
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
    ) -> Result<OpeningClaim, Self::Error>;
}

impl<const Q: u128> F2ZVerifier<Q> {
    /// Replays the proof of the caller's linear claim about the committed bits.
    ///
    /// The transcript arrives carrying the caller's events; this appends and
    /// hands it back.
    pub fn verify<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<Q>,
        com: Root,
        reduction: &R,
        mut transcript: VerifierState<'_>,
    ) -> Result<OpeningClaim, VerifyError<R::Error>> {
        // Step 1: the admissibility and precondition checks have already run --
        // the shape gates in Shape::new, the modulus in Fq's own const assertions,
        // the generator's order in F2ZConfig::new and the weight counts in
        // LinearClaim::new. What is left is binding, before any challenge.
        transcript.public_message(&com.0);
        transcript.public_message(self.params());

        // TODO: step 2, reducing the modulus, is absent, as on the prover.

        // Step 3: read the folds, range-check them, reconstruct against mu.
        let fold = self
            .receive_fold(claim, &mut transcript)
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
        let claim = reduction
            .reduce(&input, &mut transcript)
            .map_err(VerifyError::Reduction)?;

        // Step 5 is conditional and a merged forest does not need it.

        // Both streams must be spent. Taking the transcript by value is what
        // makes that assertable here rather than by the caller, and it is why
        // step 6 belongs above this line rather than after the return.
        //
        // TODO(#15): step 6 is absent, so returning Ok is not accepting a
        // proof: this hands back the claim the opening would discharge rather
        // than discharging it. `verify_lin_batch(commitment, &[claim],
        // AlreadyBound, &mut transcript)` goes here, and only then is the
        // return an acceptance.
        transcript
            .check_eof()
            .map_err(|_| VerifyError::TrailingData)?;

        Ok(claim)
    }
}
