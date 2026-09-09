//! The F2Z verifier, written from the spec rather than from the prover.

pub mod fold;
pub mod setup;
pub mod verify;

pub use fold::ReceiveError;
pub use setup::F2ZVerifier;
pub use verify::{GkrReduction, Reduction, VerifyError};
