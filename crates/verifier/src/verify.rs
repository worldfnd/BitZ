//! `VerifyF2Z`.

use std::collections::VecDeque;

use common::{Fold, LinearClaim, OpeningQuery, ReductionInput, Root};
use field::F128;
use num_traits::ConstOne;
use pcs::{CommitError, CommitScheme, Pcs, StatementBinding};
use transcript::VerifierState;

use crate::{F2ZVerifier, ReceiveError};

/// A proof the verifier rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError<E> {
    /// The fold round failed its own checks.
    Fold(ReceiveError),
    /// The reduction failed.
    Reduction(E),
    /// The opening did not discharge the reduction's claim.
    Opening(CommitError),
    /// A stream held bytes the protocol never read.
    TrailingData,
}

/// Step 4, replayed.
///
/// TODO: #8 implements this.
/// Xander: This trait only for mock testing?
pub trait Reduction<const Q: u128> {
    type Error;

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningQuery, Self::Error>;
}

/// Stands in for #8: runs the sumcheck verifier over the grand product, but
/// does not yet turn the resulting claim into a discharged [`OpeningQuery`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Reduce;

pub enum ReduceError {
    GKR,
}

impl<const Q: u128> Reduction<Q> for Reduce {
    type Error = ReduceError;

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningQuery, Self::Error> {
        let fold = input.fold;

        let (_inner_product_claim, _u1, _u2) =
            gkr_reduce(transcript, fold).ok_or(ReduceError::GKR)?;

        // TODO(#8): turn `u1`/`u2`/the inner-product claim into a discharged `OpeningQuery`.
        // Sina / Alex?
        todo!("#8")
    }
}

fn gkr_reduce(transcript: &mut VerifierState, fold: &Fold) -> Option<(F128, Vec<F128>, Vec<F128>)> {
    // throughout the function point is less than r1+r2 elements
    let mut point = fold.zeta.clone();
    point.reverse();
    let point = VecDeque::from(point);

    let r1 = fold.row_images.len().max(1).ilog2();

    let (mut point, mle_leaf_claim) = gkr::gpgkr_verify(transcript, fold.e0, point, r1)?;

    let alfa_c = Vec::from(point.split_off(r1 as usize));
    let alfa_b = Vec::from(point);

    let inner_product_claim = mle_leaf_claim - F128::ONE;
    // Allocates 2*l1+l2 space if the compiler doesn't fuse.
    let u1: Vec<_> = fold
        .row_images
        .iter()
        .zip(poly::eq_table(&alfa_b))
        .map(|(a, b)| *a * b) // Does the later step benefit from wide mul?
        .collect();
    let u2 = poly::eq_table(&alfa_c);
    Some((inner_product_claim, u1, u2))
}

impl<const Q: u128> F2ZVerifier<Q> {
    /// Replays the proof of the caller's linear claim about the committed bits.
    ///
    /// `pcs` must be the scheme the commitment was made under. The transcript
    /// arrives carrying the caller's events; this appends and consumes it.
    pub fn verify<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<Q>,
        pcs: &Pcs,
        com: Root,
        reduction: &R,
        mut transcript: VerifierState<'_>,
    ) -> Result<(), VerifyError<R::Error>> {
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
}
