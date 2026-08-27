//! The seam between the fold and the opening.
//!
//! Steps 5.2 and 5.2a — the grand product and the sumcheck that normalizes its
//! affine leaf — are not implemented here. What is here is the contract: what
//! that reduction is handed, and what it must hand back.

use field::F128;

use crate::{F2ZConfig, Fold, LinearClaim, Root};

/// A multilinear evaluation claim on the committed bits: `f~(point) = target`.
///
/// TODO: this is `pcs::OpeningQuery` under another name. Once #15 lands it
/// should be that type, so the reduction's output is the opening's input with
/// nothing to convert between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningClaim {
    /// The evaluation point, low-index-bit-first.
    pub point: Vec<F128>,
    /// The claimed evaluation at `point`.
    pub target: F128,
}

/// Everything the reduction reads, which is the bundle Step 5.2 names.
///
/// The images `g^{eta_j}` and the row images `y_i` are already in [`Fold`],
/// along with the challenge and the batched output claim, so this carries a
/// reference rather than restating them.
#[derive(Clone, Copy)]
pub struct ReductionInput<'a, const Q: u128> {
    pub config: &'a F2ZConfig<Q>,
    pub claim: &'a LinearClaim<Q>,
    pub commitment: Root,
    pub fold: &'a Fold,
}
