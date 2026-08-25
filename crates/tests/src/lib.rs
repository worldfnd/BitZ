//! Round-trip coverage for the prover and the verifier.
//!
//! This crate exists only so the two sides can be paired without either
//! depending on the other: `verifier` is written from the specification, and a
//! dev-dependency on `prover` would let a check drift into agreeing with how
//! the prover happens to compute something.
//!
//! Everything lives in `tests/`; the library is deliberately empty.
