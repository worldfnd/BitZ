//! The BitZ prover.

pub mod fold;
pub mod prove;
mod reduce;
pub mod setup;

pub use fold::SendError;
pub use prove::ProveError;
pub use setup::BitZProver;
