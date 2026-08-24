//! What the prover and the verifier must arrive at identically.
//!
//! Nothing here is secret or transmitted: every value is a function of the
//! public shape and the caller's claim, so both sides derive it rather than
//! one side sending it.

pub mod shape;
pub mod table;

pub use shape::{Shape, ShapeError};
pub use table::{BitTable, TableError};
