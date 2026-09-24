//! What the prover and the verifier must arrive at identically.
//!
//! Nothing here is secret or transmitted: every value is a function of the
//! public shape and the caller's claim, so both sides derive it rather than
//! one side sending it.

pub mod claim;
pub mod fold;
pub mod opening;
pub mod params;
pub mod shape;
pub mod table;
pub mod virtual_map;

pub use claim::{ClaimError, LinearClaim, Root};
pub use fold::{
    Fold, FoldError, column_images, fold_column, fold_columns, reconstruct, row_images,
};
pub use opening::OpeningQuery;
pub use params::{BitZParams, ParamsError, VirtualParams, VirtualParamsError};
pub use shape::{Shape, ShapeError};
pub use table::{BitTable, TableError, TransposeError, TransposedBitTable};
pub use virtual_map::{
    TransposedWeights, VirtualMap, VirtualMapError, VirtualStatement, VirtualStatementError,
};

use crypto_primitives::Semiring;
use std::ops::Neg;

pub trait BitzSemiring: Semiring + From<u64> {}

impl<T> BitzSemiring for T where T: Semiring + From<u64> {}

// Since BigInt does not support CheckedNeg and CheckedRem, we can't use Ring here
pub trait BitzRing: BitzSemiring + Neg<Output = Self> {}

impl<T> BitzRing for T where T: BitzSemiring + Neg<Output = Self> {}
