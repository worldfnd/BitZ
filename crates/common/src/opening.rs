//! The claim the opening scheme discharges.
//!
//! The reduction that produces this and the commitment scheme that consumes it
//! sit on opposite sides of the protocol, so the type belongs to neither.

use field::F128;

/// A multilinear evaluation claim on the committed bits: `f~(point) = target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningQuery {
    /// The evaluation point, low-index-bit-first.
    pub point: Vec<F128>,
    /// The claimed evaluation at `point`.
    pub target: F128,
}
