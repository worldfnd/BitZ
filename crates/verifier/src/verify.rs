//! `VerifyF2Z`.

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

        let (_inner_product_claim, _u1) = gkr_reduce(transcript, fold).ok_or(ReduceError::GKR)?;

        // TODO(#8): turn `u1`/`u2`/the inner-product claim into a discharged `OpeningQuery`.
        // Sina / Alex?
        todo!("#8")
    }
}

fn gkr_reduce(transcript: &mut VerifierState, fold: &Fold) -> Option<(F128, Vec<F128>)> {
    // throughout the function point is less than r1+r2 elements
    let r1 = fold.row_images.len().max(1).ilog2();

    let (point, mle_leaf_claim) = gkr::gpgkr_verify(transcript, fold.e0, &fold.zeta, r1)?;

    let r2 = point.len() - r1 as usize;
    let alfa_b = &point[r2..];

    let inner_product_claim = mle_leaf_claim - F128::ONE;
    // Allocates 2*l1 space if the compiler doesn't fuse.
    let u1: Vec<_> = fold
        .row_images
        .iter()
        .zip(poly::eq_table(alfa_b))
        .map(|(a, b)| (*a - F128::ONE) * b) // Does the later step benefit from wide mul?
        .collect();

    Some((inner_product_claim, u1))
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

#[cfg(test)]
mod round_trip_ai_test {
    //! Drives a real proof through `gkr::GrandProductCircuit` +
    //! `gkr::gpgkr_prove` (mirroring what `prover::prove::gkr_reduce` does --
    //! that function is crate-private, so it can't be called directly from
    //! here) and checks this crate's own `gkr_reduce` recovers the same
    //! `alfa_b`/`u1`/`inner_product_claim` a prover computes from the same
    //! challenges. This is the check `order_check` (in the `prover` crate)
    //! can't provide on its own: that this crate's independent reverse/split
    //! logic agrees with the prover's, not just that each is internally
    //! self-consistent.
    use common::{F2ZParams, Fold, Shape};
    use field::gf128::smallest_generator;
    use gkr::{GrandProductCircuit, gpgkr_prove};
    use num_traits::{ConstOne, ConstZero, identities::Zero};

    use super::*;

    const Q: u128 = (1 << 114) - 11;

    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    fn params() -> F2ZParams<Q> {
        F2ZParams::new(shape(), smallest_generator()).unwrap()
    }

    #[test]
    fn verifier_agrees_with_a_real_prover_transcript() {
        let shape = shape();

        let mut packed = vec![F128::ZERO; (1 << shape.log_bits()) / 128];
        for column in 0..shape.columns() {
            for row in 0..shape.rows() {
                let h = (row as u64).wrapping_mul(2654435761)
                    ^ (column as u64).wrapping_mul(0x9E3779B97F4A7C15);
                if (h >> 5) & 1 == 1 {
                    let index = (column << shape.log_rows()) | row;
                    let element = &mut packed[index >> 7];
                    let offset = index % 128;
                    if offset < 64 {
                        element.lo |= 1u64 << offset;
                    } else {
                        element.hi |= 1u64 << (offset - 64);
                    }
                }
            }
        }
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        // Real prover-side leaf construction and GKR proof, matching
        // `prover::prove::gkr_reduce` exactly, so the transcript this
        // produces is one this crate's `gkr_reduce` must accept.
        let mut leafs = vec![F128::zero(); shape.columns() * shape.rows()];
        for b in 0..shape.rows() {
            for c in 0..shape.columns() {
                leafs[b * shape.columns() + c] = if table.bit(c, b) {
                    row_images[b]
                } else {
                    F128::ONE
                };
            }
        }
        let circuit = GrandProductCircuit::new(leafs);
        let (top_layer, witnesses) = circuit.batched_eval(shape.columns());

        // `gpgkr_verify`'s starting claim is `fold.e0`, so it has to be the
        // real circuit's own top (output) layer evaluated at `zeta` -- not
        // an arbitrary placeholder -- or the verifier correctly rejects.
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, top_layer, row_images.clone(), zeta.clone()).unwrap();

        let mut prover = transcript::build_prover("verifier-round-trip", &F128::ZERO);
        let (mut point, claim) = gpgkr_prove(&mut prover, &zeta, witnesses);
        let proof = prover.finish();

        // Expected values, derived independently from the prover's own
        // returned point/claim using the same r1/r2 split `gkr_reduce` (in
        // both crates) relies on.
        let r1 = row_images.len().max(1).ilog2();
        let r2 = point.len() - r1 as usize;
        let expected_alfa_b = point.split_off(r2);
        let expected_u1: Vec<F128> = row_images
            .iter()
            .zip(poly::eq_table(&expected_alfa_b))
            .map(|(a, b)| (*a - F128::ONE) * b)
            .collect();
        let expected_inner_product_claim = claim - F128::ONE;

        let mut verifier = transcript::build_verifier("verifier-round-trip", &F128::ZERO, &proof);
        let (inner_product_claim, u1) = gkr_reduce(&mut verifier, &fold).unwrap();
        verifier.check_eof().unwrap();

        assert_eq!(
            inner_product_claim, expected_inner_product_claim,
            "verifier's inner_product_claim disagrees with the prover's"
        );
        assert_eq!(
            u1, expected_u1,
            "verifier's u1 disagrees with the prover's -- alfa_b is split \
             or ordered differently between the two crates"
        );
    }
}
