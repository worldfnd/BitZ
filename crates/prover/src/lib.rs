//! The BitZ prover.

pub mod fold;
pub mod prove;
pub mod setup;

pub use fold::SendError;
pub use prove::{ProveError, Reduce, Reduction};
pub use setup::BitZProver;
