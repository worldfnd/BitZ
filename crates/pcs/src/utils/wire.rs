use core::mem::size_of;

use flock_core::field::F128 as FlockF128;
use flock_core::pcs::LOG_PACKING;

use crate::CommitError;
use crate::bridge::{as_flock_f128, from_flock_f128};

use super::{NO_SCOPE, PcsTranscript, PublicTranscript};

const EVENT_HEADER_LEN: usize = 24;
const RING_SWITCH_MESSAGE_TAG: u16 = 0x4001;
const RING_SWITCH_CHALLENGE_TAG: u16 = 0x4101;
const BATCHING_CHALLENGE_TAG: u16 = 0x4102;
const OPENING_TARGET_TAG: u16 = 0x5001;
const SQUEEZE_RESULT_BIT: u16 = 0x8000;
const FIELD_ENCODING_LEN: u32 = 16;
const RING_SWITCH_VALUE_COUNT: usize = 1 << LOG_PACKING;
const RING_SWITCH_PAYLOAD_LEN: u64 =
    size_of::<u32>() as u64 + (RING_SWITCH_VALUE_COUNT * FIELD_ENCODING_LEN as usize) as u64;

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

#[cfg(test)]
mod tests {
    use core::mem::size_of;
    use std::collections::VecDeque;

    use field::F128;
    use flock_core::field::F128 as FlockF128;
    use flock_core::pcs::LOG_PACKING;
    use transcript::{Encoding, build_prover, build_verifier};

    use super::*;
    use crate::CommitError;

    fn ring_switch_challenges(last_value: FlockF128) -> (Vec<FlockF128>, Vec<FlockF128>) {
        let first = [FlockF128::new(3, 5); 1 << LOG_PACKING];
        let mut last = [FlockF128::new(7, 11); 1 << LOG_PACKING];
        last[1 << (LOG_PACKING - 1)] = last_value;
        let mut prover = build_prover(b"pcs-protocol-test", b"ring-switch-schedule");

        bind_ring_switch_message(&mut prover, 0, &first).unwrap();
        bind_ring_switch_message(&mut prover, 2, &last).unwrap();
        let r_dprime = sample_shared_ring_switch_point(&mut prover);
        let etas = sample_batching_scalars(&mut prover, [0, 2]);
        (r_dprime, etas)
    }

    #[test]
    fn last_ring_switch_message_changes_shared_point_and_later_etas() {
        let original = ring_switch_challenges(FlockF128::new(13, 17));
        let changed = ring_switch_challenges(FlockF128::new(19, 23));

        assert_ne!(original.0, changed.0);
        assert_ne!(original.1, changed.1);
    }

    #[test]
    fn ring_switch_message_matches_wire_v1() {
        let mut values = [FlockF128::ZERO; RING_SWITCH_VALUE_COUNT];
        values[0] = FlockF128::new(1, 2);
        values[RING_SWITCH_VALUE_COUNT - 1] = FlockF128::new(3, 4);
        let mut prover = build_prover(b"pcs-protocol-test", b"ring-frame");

        bind_ring_switch_message(&mut prover, 2, &values).unwrap();
        let proof = prover.finish();

        assert_eq!(proof.narg_string.len(), EVENT_HEADER_LEN + 4 + 128 * 16);
        assert_eq!(
            &proof.narg_string[..EVENT_HEADER_LEN],
            &[
                0x01, 0x40, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]
        );
        assert_eq!(&proof.narg_string[24..28], &128u32.to_le_bytes());
        assert_eq!(&proof.narg_string[28..44], &F128::new(1, 2).to_bytes());
        assert_eq!(
            &proof.narg_string[proof.narg_string.len() - 16..],
            &F128::new(3, 4).to_bytes()
        );

        let mut verifier = build_verifier(b"pcs-protocol-test", b"ring-frame", &proof);
        bind_ring_switch_message(&mut verifier, 2, &values).unwrap();
        verifier.check_eof().unwrap();
    }

    #[test]
    fn ring_switch_message_rejects_wrong_frame_fields_and_values() {
        let values = [FlockF128::ZERO; RING_SWITCH_VALUE_COUNT];
        let mut prover = build_prover(b"pcs-protocol-test", b"malformed-ring-frame");
        bind_ring_switch_message(&mut prover, 2, &values).unwrap();
        let proof = prover.finish();

        for byte_index in [
            0,
            2,
            4,
            8,
            12,
            16,
            EVENT_HEADER_LEN,
            EVENT_HEADER_LEN + size_of::<u32>(),
            EVENT_HEADER_LEN + size_of::<u32>() + 64 * FIELD_ENCODING_LEN as usize,
            EVENT_HEADER_LEN + size_of::<u32>() + 127 * FIELD_ENCODING_LEN as usize,
        ] {
            let mut malformed = proof.clone();
            malformed.narg_string[byte_index] ^= 1;
            let mut verifier =
                build_verifier(b"pcs-protocol-test", b"malformed-ring-frame", &malformed);
            assert_eq!(
                bind_ring_switch_message(&mut verifier, 2, &values),
                Err(CommitError::MalformedProof)
            );
        }

        let mut verifier = build_verifier(b"pcs-protocol-test", b"malformed-ring-frame", &proof);
        assert_eq!(
            bind_ring_switch_message(&mut verifier, 3, &values),
            Err(CommitError::MalformedProof)
        );
    }

    #[test]
    fn ring_switch_message_rejects_expected_value_mismatches() {
        let values = [FlockF128::ZERO; RING_SWITCH_VALUE_COUNT];
        let mut prover = build_prover(b"pcs-protocol-test", b"mismatched-ring-values");
        bind_ring_switch_message(&mut prover, 2, &values).unwrap();
        let proof = prover.finish();

        for index in [0, RING_SWITCH_VALUE_COUNT / 2, RING_SWITCH_VALUE_COUNT - 1] {
            let mut expected = values;
            expected[index] = FlockF128::new(1, 0);
            let mut verifier =
                build_verifier(b"pcs-protocol-test", b"mismatched-ring-values", &proof);

            assert_eq!(
                bind_ring_switch_message(&mut verifier, 2, &expected),
                Err(CommitError::MalformedProof)
            );
        }
    }

    struct RecordingTranscript {
        absorbed: Vec<u8>,
        challenges: VecDeque<F128>,
        squeeze_offsets: Vec<usize>,
    }

    impl PublicTranscript for RecordingTranscript {
        fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
            self.absorbed.extend_from_slice(message.encode().as_ref());
        }

        fn verifier_message_f128(&mut self) -> F128 {
            self.squeeze_offsets.push(self.absorbed.len());
            self.challenges.pop_front().unwrap()
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct RecordedFrame {
        tag: u16,
        scope: (u32, u32, u32),
        payload: Vec<u8>,
    }

    fn parse_recorded_frames(mut bytes: &[u8]) -> Vec<RecordedFrame> {
        let mut frames = Vec::new();
        while !bytes.is_empty() {
            assert!(bytes.len() >= EVENT_HEADER_LEN);
            let tag = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
            assert_eq!(&bytes[2..4], &[0, 0]);
            let chunk = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            let level = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            let index = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
            let payload_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
            assert!(bytes.len() >= EVENT_HEADER_LEN + payload_len);
            let payload = bytes[EVENT_HEADER_LEN..EVENT_HEADER_LEN + payload_len].to_vec();
            frames.push(RecordedFrame {
                tag,
                scope: (chunk, level, index),
                payload,
            });
            bytes = &bytes[EVENT_HEADER_LEN + payload_len..];
        }
        frames
    }

    #[test]
    fn derived_group_8_to_10_frames_match_wire_v1() {
        let challenges = (1u64..=9).map(F128::from).collect::<VecDeque<_>>();
        let mut transcript = RecordingTranscript {
            absorbed: Vec::new(),
            challenges,
            squeeze_offsets: Vec::new(),
        };

        let r_dprime = sample_shared_ring_switch_point(&mut transcript);
        let etas = sample_batching_scalars(&mut transcript, [0, 2]);
        observe_opening_target(&mut transcript, 15, FlockF128::new(10, 11)).unwrap();
        let frames = parse_recorded_frames(&transcript.absorbed);

        assert_eq!(r_dprime.len(), 7);
        assert_eq!(etas.len(), 2);
        assert_eq!(
            transcript.squeeze_offsets,
            (0..9).map(|event| 28 + event * 68).collect::<Vec<_>>()
        );
        assert_eq!(frames.len(), 19);
        for index in 0..7 {
            let request = &frames[2 * index];
            let response = &frames[2 * index + 1];
            assert_eq!(request.tag, RING_SWITCH_CHALLENGE_TAG);
            assert_eq!(request.scope, (NO_SCOPE, NO_SCOPE, index as u32));
            assert_eq!(request.payload, FIELD_ENCODING_LEN.to_le_bytes());
            assert_eq!(response.tag, RING_SWITCH_CHALLENGE_TAG | SQUEEZE_RESULT_BIT);
            assert_eq!(response.scope, request.scope);
            assert_eq!(response.payload, F128::from(index as u64 + 1).to_bytes());
        }
        for (offset, scope) in [0, 2].into_iter().enumerate() {
            let request = &frames[14 + 2 * offset];
            let response = &frames[15 + 2 * offset];
            assert_eq!(request.tag, BATCHING_CHALLENGE_TAG);
            assert_eq!(request.scope, (scope, NO_SCOPE, NO_SCOPE));
            assert_eq!(request.payload, FIELD_ENCODING_LEN.to_le_bytes());
            assert_eq!(response.tag, BATCHING_CHALLENGE_TAG | SQUEEZE_RESULT_BIT);
            assert_eq!(response.scope, request.scope);
            assert_eq!(response.payload, F128::from(offset as u64 + 8).to_bytes());
        }
        assert_eq!(
            frames[18],
            RecordedFrame {
                tag: OPENING_TARGET_TAG,
                scope: (NO_SCOPE, NO_SCOPE, NO_SCOPE),
                payload: [
                    15u32.to_le_bytes().as_slice(),
                    F128::new(10, 11).to_bytes().as_slice()
                ]
                .concat(),
            }
        );
    }
}
