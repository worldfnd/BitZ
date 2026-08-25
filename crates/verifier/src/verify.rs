//! `VerifyF2Z`.

use common::{CoreStatement, OpeningClaim, ReductionInput, Root};
use field::FixedBasePow;
use transcript::VerifierState;

use crate::{ReceiveError, receive_fold};

/// A proof the verifier rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError<E> {
    /// The fold round failed its own checks.
    Fold(ReceiveError),
    /// The reduction failed.
    Reduction(E),
}

/// Steps 5.2 and 5.2a, replayed.
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

/// Replays the proof of the caller's linear claim about the committed bits.
///
/// The transcript arrives carrying the caller's events; this appends and hands
/// it back.
pub fn verify<const Q: u128, R: Reduction<Q>>(
    statement: &CoreStatement<Q>,
    com: Root,
    generator: &FixedBasePow,
    reduction: &R,
    transcript: &mut VerifierState<'_>,
) -> Result<OpeningClaim, VerifyError<R::Error>> {
    // Step 1: the admissibility and precondition checks have already run --
    // the shape gates in Shape::new, the modulus in Fq's own const assertions,
    // the generator's order and the weight counts in CoreStatement::new. What
    // is left is binding, before any challenge.
    transcript.public_message(&com.0);
    transcript.public_message(statement);

    // TODO: Step 5.0, reduce the modulus, is absent, as on the prover.

    // Step 5.1: read the folds, range-check them, reconstruct against mu.
    let fold = receive_fold(statement, generator, transcript).map_err(VerifyError::Fold)?;

    // Steps 5.2 and 5.2a, replayed.
    //
    // TODO(#8): both live behind `Reduction`, which nothing implements yet.
    let input = ReductionInput {
        statement,
        commitment: com,
        fold: &fold,
    };
    let claim = reduction
        .reduce(&input, transcript)
        .map_err(VerifyError::Reduction)?;

    // Step 5.2b is conditional and a merged forest does not need it.

    // TODO(#15): Step 5.3 is absent, so returning Ok is not accepting a proof:
    // this hands back the claim the opening would discharge rather than
    // discharging it. The tail becomes `verify_lin_batch(commitment, &[claim],
    // AlreadyBound, transcript)` then `check_eof`, and only then is the return
    // an acceptance.
    Ok(claim)
}
