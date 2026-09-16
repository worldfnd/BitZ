//! The BitZ verifier, written from the spec rather than from the prover.

pub mod fold;
mod reduce;
pub mod setup;
pub mod verify;

pub use fold::ReceiveError;
pub use reduce::ReduceError;
pub use setup::BitZVerifier;
pub use verify::VerifyError;
