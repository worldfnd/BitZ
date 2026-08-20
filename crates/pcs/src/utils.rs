//! Shared encoding and transcript rules for multilinear openings.
//!
//! The ring-switch, batching, and opening-target events use wire-v1 numeric frames.
//! See the normative [wire-v1 event frame and tag registry].
//! The complete opening proof uses one bounded hint because Flock verifies an in-memory proof.
//! Fiat–Shamir values also use NARG and are checked against the hint during replay.
//!
//! [wire-v1 event frame and tag registry]: https://github.com/worldfnd/f2z-benchmark/blob/5014c717e88ab5e54e70e7a1099caaca5c41a926/docs/f2z-pcs-spec/part5-wire-format.tex#L235-L319

mod batch;
mod proof;
mod transcript;
mod wire;

pub(crate) use batch::{bind_statement, validate_batch};
pub(crate) use proof::{read_opening_proof, write_opening_proof};
pub(crate) use transcript::{PcsTranscript, PublicTranscript};
pub(crate) use wire::{
    bind_ring_switch_message, observe_opening_target, sample_batching_scalars,
    sample_shared_ring_switch_point,
};

pub(crate) const STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v1";
const NO_SCOPE: u32 = u32::MAX;
