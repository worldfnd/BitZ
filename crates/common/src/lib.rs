//! What the prover and the verifier must arrive at identically.
//!
//! Nothing here is secret or transmitted: every value is a function of the
//! public shape and the caller's claim, so both sides derive it rather than
//! one side sending it.

pub mod claim;
pub mod fold;
pub mod opening;
pub mod params;
pub mod reduction;
pub mod shape;
pub mod table;
pub mod virtual_map;

pub use claim::{ClaimError, LinearClaim, Root};
pub use fold::{
    Fold, FoldError, column_images, fold_column, fold_columns, reconstruct, row_images,
};
pub use opening::OpeningQuery;
pub use params::{F2ZParams, ParamsError, VirtualParams, VirtualParamsError};
pub use reduction::ReductionInput;
pub use shape::{Shape, ShapeError};
pub use table::{BitTable, TableError};
pub use virtual_map::{TransposedWeights, VirtualMap, VirtualMapError, transpose_query};
