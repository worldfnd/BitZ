//! Shared encoding and transcript rules for multilinear openings.
//!
//! The ring-switch, batching, and opening-target events use wire-v1 numeric frames.
//! See the normative [wire-v1 event frame and tag registry].
//! The complete opening proof uses one bounded hint because Flock verifies an in-memory proof.
//! Fiat–Shamir values also use NARG and are checked against the hint during replay.
//!
//! [wire-v1 event frame and tag registry]: https://github.com/worldfnd/f2z-benchmark/blob/5014c717e88ab5e54e70e7a1099caaca5c41a926/docs/f2z-pcs-spec/part5-wire-format.tex#L235-L319

use core::mem::size_of;

use field::F128 as LocalF128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::LOG_PACKING;
use transcript::{Encoding, NargDeserialize, ProverState, VerifierState};

use crate::bridge::{as_flock_f128, from_flock_f128};
use crate::{CommitError, Pcs, ScopedOpeningQuery};

mod proof;

pub(crate) use proof::{read_opening_proof, write_opening_proof};

pub(crate) const STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v4";

const EVENT_HEADER_LEN: usize = 24;
const NO_SCOPE: u32 = u32::MAX;
const RING_SWITCH_MESSAGE_TAG: u16 = 0x4001;
const RING_SWITCH_CHALLENGE_TAG: u16 = 0x4101;
const BATCHING_CHALLENGE_TAG: u16 = 0x4102;
const OPENING_TARGET_TAG: u16 = 0x5001;
const SQUEEZE_RESULT_BIT: u16 = 0x8000;
const FIELD_ENCODING_LEN: u32 = 16;
const RING_SWITCH_VALUE_COUNT: usize = 1 << LOG_PACKING;
const RING_SWITCH_PAYLOAD_LEN: u64 =
    size_of::<u32>() as u64 + (RING_SWITCH_VALUE_COUNT * FIELD_ENCODING_LEN as usize) as u64;

pub(crate) trait PublicTranscript {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T);

    fn verifier_message_f128(&mut self) -> LocalF128;
}

impl PublicTranscript for ProverState {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        ProverState::public_message(self, message);
    }

    fn verifier_message_f128(&mut self) -> LocalF128 {
        ProverState::verifier_message(self)
    }
}

impl PublicTranscript for VerifierState<'_> {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        VerifierState::public_message(self, message);
    }

    fn verifier_message_f128(&mut self) -> LocalF128 {
        VerifierState::verifier_message(self)
    }
}

pub(crate) trait PcsTranscript: PublicTranscript {
    /// Writes `expected`, or reads and verifies the next prover message.
    fn bind_prover_message<T>(&mut self, expected: &T) -> Result<(), CommitError>
    where
        T: Encoding<[u8]> + NargDeserialize + PartialEq;
}

impl PcsTranscript for ProverState {
    fn bind_prover_message<T>(&mut self, expected: &T) -> Result<(), CommitError>
    where
        T: Encoding<[u8]> + NargDeserialize + PartialEq,
    {
        ProverState::prover_message(self, expected);
        Ok(())
    }
}

impl PcsTranscript for VerifierState<'_> {
    fn bind_prover_message<T>(&mut self, expected: &T) -> Result<(), CommitError>
    where
        T: Encoding<[u8]> + NargDeserialize + PartialEq,
    {
        let observed =
            VerifierState::prover_message::<T>(self).map_err(|_| CommitError::MalformedProof)?;
        if observed != *expected {
            return Err(CommitError::MalformedProof);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct EventScope {
    chunk: u32,
    level: u32,
    index: u32,
}

impl EventScope {
    const NONE: Self = Self {
        chunk: NO_SCOPE,
        level: NO_SCOPE,
        index: NO_SCOPE,
    };

    const fn chunk(chunk: u32) -> Self {
        Self {
            chunk,
            level: NO_SCOPE,
            index: NO_SCOPE,
        }
    }

    const fn index(index: u32) -> Self {
        Self {
            chunk: NO_SCOPE,
            level: NO_SCOPE,
            index,
        }
    }
}

fn event_header(tag: u16, scope: EventScope, payload_len: u64) -> [u8; EVENT_HEADER_LEN] {
    let mut header = [0; EVENT_HEADER_LEN];
    header[0..2].copy_from_slice(&tag.to_le_bytes());
    header[4..8].copy_from_slice(&scope.chunk.to_le_bytes());
    header[8..12].copy_from_slice(&scope.level.to_le_bytes());
    header[12..16].copy_from_slice(&scope.index.to_le_bytes());
    header[16..24].copy_from_slice(&payload_len.to_le_bytes());
    header
}

/// Binds one canonical tag-4001 ring-switch vector through NARG.
pub(crate) fn bind_ring_switch_message(
    transcript: &mut impl PcsTranscript,
    scope: u32,
    values: &[FlockF128],
) -> Result<(), CommitError> {
    if values.len() != RING_SWITCH_VALUE_COUNT {
        return Err(CommitError::invalid_configuration(
            "ring-switch vector length mismatch",
        ));
    }

    transcript.bind_prover_message(&event_header(
        RING_SWITCH_MESSAGE_TAG,
        EventScope::chunk(scope),
        RING_SWITCH_PAYLOAD_LEN,
    ))?;
    transcript.bind_prover_message(&(RING_SWITCH_VALUE_COUNT as u32))?;
    for &value in values {
        transcript.bind_prover_message(&from_flock_f128(value))?;
    }
    Ok(())
}

/// Samples the seven tag-4101 coordinates of the shared ring-switch point.
pub(crate) fn sample_shared_ring_switch_point(
    transcript: &mut impl PublicTranscript,
) -> Vec<FlockF128> {
    (0..LOG_PACKING)
        .map(|index| {
            sample_f128_event(
                transcript,
                RING_SWITCH_CHALLENGE_TAG,
                EventScope::index(index as u32),
            )
        })
        .collect()
}

/// Samples one tag-4102 batching scalar for each active claim scope.
pub(crate) fn sample_batching_scalars(
    transcript: &mut impl PublicTranscript,
    scopes: impl IntoIterator<Item = u32>,
) -> Vec<FlockF128> {
    scopes
        .into_iter()
        .map(|scope| {
            sample_f128_event(transcript, BATCHING_CHALLENGE_TAG, EventScope::chunk(scope))
        })
        .collect()
}

/// Absorbs the derived tag-5001 opening target before recursive Ligerito.
pub(crate) fn observe_opening_target(
    transcript: &mut impl PublicTranscript,
    m_p: usize,
    beta: FlockF128,
) -> Result<(), CommitError> {
    let m_p =
        u32::try_from(m_p).map_err(|_| CommitError::invalid_configuration("m_p exceeds u32"))?;
    transcript.public_message(&event_header(
        OPENING_TARGET_TAG,
        EventScope::NONE,
        (size_of::<u32>() + FIELD_ENCODING_LEN as usize) as u64,
    ));
    transcript.public_message(&m_p);
    transcript.public_message(&from_flock_f128(beta));
    Ok(())
}

fn sample_f128_event(
    transcript: &mut impl PublicTranscript,
    tag: u16,
    scope: EventScope,
) -> FlockF128 {
    transcript.public_message(&event_header(tag, scope, size_of::<u32>() as u64));
    transcript.public_message(&FIELD_ENCODING_LEN);
    let result = transcript.verifier_message_f128();
    transcript.public_message(&event_header(
        tag | SQUEEZE_RESULT_BIT,
        scope,
        FIELD_ENCODING_LEN as u64,
    ));
    transcript.public_message(&result);
    as_flock_f128(result)
}

/// Absorbs the public statement in either transcript.
pub(crate) fn bind_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    queries: &[ScopedOpeningQuery<'_>],
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(queries.len() as u64));
    for scoped_query in queries {
        transcript.public_message(&scoped_query.scope);
        let query = scoped_query.query;
        transcript.public_message(&(query.point.len() as u64));
        for coordinate in &query.point {
            transcript.public_message(coordinate);
        }
        transcript.public_message(&query.target);
    }
}

pub(crate) fn validate_batch(
    queries: &[ScopedOpeningQuery<'_>],
    expected_m: usize,
) -> Result<(), CommitError> {
    if queries.is_empty() {
        return Err(CommitError::EmptyBatch);
    }
    if queries
        .iter()
        .any(|scoped_query| scoped_query.scope == NO_SCOPE)
    {
        return Err(CommitError::InvalidClaimScope);
    }
    if queries
        .windows(2)
        .any(|pair| pair[0].scope >= pair[1].scope)
    {
        return Err(CommitError::InvalidClaimScopeOrder);
    }
    if queries
        .iter()
        .any(|scoped_query| scoped_query.query.point.len() != expected_m)
    {
        return Err(CommitError::PointLengthMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
