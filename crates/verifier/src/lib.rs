//! The F2Z verifier.
//!
//! Written from the specification rather than against the prover: it shares
//! only `common`, so a check that happens to agree with how the prover
//! computed something is agreeing with the protocol, not with an
//! implementation detail.

pub mod fold;
pub mod verify;

pub use fold::{ReceiveError, receive_fold};
pub use verify::{Reduction, VerifyError, verify};
