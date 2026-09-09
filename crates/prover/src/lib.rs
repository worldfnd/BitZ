//! The BitZ prover.

pub mod fold;
pub mod prove;
pub mod setup;

pub use fold::SendError;
pub use prove::{GkrReduction, ProveError, Reduction};
pub use setup::BitZProver;
