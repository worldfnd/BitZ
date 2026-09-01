//! The Fiat-Shamir transcript: spongefish plus a hint channel.
//!
//! Every prover and verifier message passes through this layer. The prover
//! writes the narg string, hints ride in a second byte vector the sponge
//! never sees, and the proof is the pair of both.
//!
//! # Framing rules
//!
//! - **F1** The protocol id, session and instance are absorbed as domain
//!   tags at construction and never appear in the narg string. The instance
//!   is mandatory; spongefish's typestate makes omitting it a compile error.
//! - **F2** Every prover message is absorbed through its canonical
//!   [`Encoding`]. The verifier deserializes from the narg string and
//!   re-absorbs the canonical re-encoding, so both sponges see identical
//!   bytes exactly when the wire bytes are canonical.
//! - **F3** Challenges leave the sponge only through `verifier_message`
//!   squeezes.
//! - **F4** Hints bypass the sponge entirely. A value may ride as a hint
//!   only if already-absorbed data determines it and it is verified before
//!   the next challenge is sampled — this layer cannot enforce that, the
//!   protocol above it must. Verification consumes both streams to the end:
//!   [`VerifierState::check_eof`] fails on leftover narg or hint bytes.
//! - **F5** Field wire formats are fixed in `field::codec`, one canonical
//!   form per element.
//! - **F6** Records are not self-delimiting. The protocol's fixed order gives
//!   the next record's type, and the type gives how to read it.

mod domain;
mod proof;
mod prover;
mod verifier;

pub use domain::{PROTOCOL_LABEL, build_prover, build_verifier};
pub use proof::Proof;
pub use prover::ProverState;
pub use spongefish::{
    Decoding, Encoding, NargDeserialize, NargSerialize, VerificationError, VerificationResult,
};
pub use verifier::VerifierState;
