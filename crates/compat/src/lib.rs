//! Independent execution of the ordinary U32 F2Z wire profile.
//! Reference artifacts are compared by the executable, never read by the prover.
pub mod statement;
pub use spartan::reference::{ROWS, Witness};
pub mod forest;

pub mod messages;

pub mod capture;
pub mod parity;
