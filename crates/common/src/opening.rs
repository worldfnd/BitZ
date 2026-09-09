//! The claim the opening scheme discharges.
//!
//! The reduction that produces this and the commitment scheme that consumes it
//! sit on opposite sides of the protocol, so the type belongs to neither.

use field::F128;

use crate::LinearClaim;

/// A linear opening claim over the original committed bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpeningQuery {
    /// A multilinear evaluation claim.
    Mle {
        /// The evaluation point, in low-index-bit-first order.
        point: Vec<F128>,
        /// The claimed multilinear evaluation at `point`.
        target: F128,
    },
    /// An inner-product claim with factored weights over `F128`.
    InnerProduct {
        /// The row weights, column weights, and claimed target.
        claim: LinearClaim<F128>,
    },
}
