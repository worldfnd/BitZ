//! The claim the opening scheme discharges.
//!
//! The reduction that produces this and the commitment scheme that consumes it
//! sit on opposite sides of the protocol, so the type belongs to neither.

use field::F128;

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
    /// An arbitrary `F128` inner-product claim.
    InnerProduct {
        /// One weight for each original bit, in commitment index order.
        weights: Vec<F128>,
        /// The claimed inner product.
        target: F128,
    },
}

impl OpeningQuery {
    /// Returns the transcript domain label for this query's ring-switch claims.
    pub fn label(&self) -> &'static [u8] {
        match self {
            Self::Mle { .. } => b"f2z/pcs/mle-claims/v1",
            Self::InnerProduct { .. } => b"f2z/pcs/bit-inner-product-claims/v1",
        }
    }
}
