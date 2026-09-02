//! Multilinear openings over the Flock commitment.
//!
//! [`open`] and [`verify`] define the protocol roles. [`ring_switch`] contains
//! the pure reduction from one MLE claim to one packed Ligerito claim.

mod open;
mod ring_switch;
mod verify;

pub(crate) use open::open;
pub(crate) use verify::verify;
