//! What the prover and the verifier must arrive at identically.
//!
//! Nothing here is secret or transmitted: every value is a function of the
//! public shape and the caller's claim, so both sides derive it rather than
//! one side sending it.

pub mod fold;
pub mod reduction;
pub mod shape;
pub mod statement;
pub mod table;

pub use fold::{Fold, FoldError, fold_column, reconstruct, row_images};
pub use reduction::{OpeningClaim, ReductionInput};
pub use shape::{Shape, ShapeError};
pub use statement::{CoreStatement, EXTENSION_BITS, Root, StatementError};
pub use table::{BitTable, TableError};
