//! Wire-v1 frames for single PCS openings.

use core::mem::size_of;

use flock_core::field::F128 as FlockF128;
use flock_core::pcs::LOG_PACKING;
use transcript::{ProverState, VerifierState};

use crate::CommitError;
use crate::bridge::{as_flock_f128, from_flock_f128};
use crate::ring_switch::CLAIM_COUNT;

use super::PublicTranscript;
use super::transcript::PcsTranscript;

const EVENT_HEADER_LEN: usize = 24;
const RING_SWITCH_MESSAGE_TAG: u16 = 0x4001;
const RING_SWITCH_CHALLENGE_TAG: u16 = 0x4101;
const INNER_PRODUCT_COORDINATES_TAG: u16 = 0x4201;
const INNER_PRODUCT_BATCHING_CHALLENGE_TAG: u16 = 0x4301;
const OPENING_TARGET_TAG: u16 = 0x5001;
const SQUEEZE_RESULT_BIT: u16 = 0x8000;
const FIELD_ENCODING_LEN: u32 = 16;
const NO_SCOPE: u32 = u32::MAX;
const SINGLE_OPENING_SCOPE: u32 = 0;
const RING_SWITCH_PAYLOAD_LEN: u64 =
    size_of::<u32>() as u64 + (CLAIM_COUNT * FIELD_ENCODING_LEN as usize) as u64;
const INNER_PRODUCT_COORDINATES_PAYLOAD_LEN: u64 =
    size_of::<u32>() as u64 + (CLAIM_COUNT * FIELD_ENCODING_LEN as usize) as u64;

/// Binds the single opening's tag-4001 ring-switch vector through NARG.
pub(crate) fn bind_ring_switch_message(
    transcript: &mut impl PcsTranscript,
    values: &[FlockF128],
) -> Result<(), CommitError> {
    if values.len() != CLAIM_COUNT {
        return Err(CommitError::invalid_configuration(
            "ring-switch vector length mismatch",
        ));
    }
    transcript.bind_prover_message(&event_header(
        RING_SWITCH_MESSAGE_TAG,
        SINGLE_OPENING_SCOPE,
        NO_SCOPE,
        NO_SCOPE,
        RING_SWITCH_PAYLOAD_LEN,
    ))?;
    transcript.bind_prover_message(&(CLAIM_COUNT as u32))?;
    for &value in values {
        transcript.bind_prover_message(&from_flock_f128(value))?;
    }
    Ok(())
}

/// Writes the 128 packed coordinate claims for an arbitrary bit inner product.
pub(crate) fn write_inner_product_claims(
    transcript: &mut ProverState,
    values: &[FlockF128; CLAIM_COUNT],
) {
    transcript.prover_message(&event_header(
        INNER_PRODUCT_COORDINATES_TAG,
        SINGLE_OPENING_SCOPE,
        NO_SCOPE,
        NO_SCOPE,
        INNER_PRODUCT_COORDINATES_PAYLOAD_LEN,
    ));
    transcript.prover_message(&(CLAIM_COUNT as u32));
    for &value in values {
        transcript.prover_message(&from_flock_f128(value));
    }
}

/// Reads the 128 packed coordinate claims for an arbitrary bit inner product.
pub(crate) fn read_inner_product_claims(
    transcript: &mut VerifierState<'_>,
) -> Result<[FlockF128; CLAIM_COUNT], CommitError> {
    let expected_header = event_header(
        INNER_PRODUCT_COORDINATES_TAG,
        SINGLE_OPENING_SCOPE,
        NO_SCOPE,
        NO_SCOPE,
        INNER_PRODUCT_COORDINATES_PAYLOAD_LEN,
    );
    let header = transcript
        .prover_message::<[u8; EVENT_HEADER_LEN]>()
        .map_err(|_| CommitError::MalformedProof)?;
    if header != expected_header {
        return Err(CommitError::MalformedProof);
    }
    let count = transcript
        .prover_message::<u32>()
        .map_err(|_| CommitError::MalformedProof)?;
    if count as usize != CLAIM_COUNT {
        return Err(CommitError::MalformedProof);
    }

    let mut values = [FlockF128::ZERO; CLAIM_COUNT];
    for value in &mut values {
        *value = transcript
            .prover_message::<field::F128>()
            .map(as_flock_f128)
            .map_err(|_| CommitError::MalformedProof)?;
    }
    Ok(values)
}

/// Samples the seven tag-4101 coordinates of the shared ring-switch point.
pub(crate) fn sample_ring_switch_point(
    transcript: &mut impl PublicTranscript,
) -> [FlockF128; LOG_PACKING] {
    core::array::from_fn(|index| {
        sample_f128_event(
            transcript,
            RING_SWITCH_CHALLENGE_TAG,
            NO_SCOPE,
            NO_SCOPE,
            index as u32,
        )
    })
}

/// Samples one independent batching challenge for each coordinate claim.
pub(crate) fn sample_inner_product_batching_challenges(
    transcript: &mut impl PublicTranscript,
) -> [FlockF128; CLAIM_COUNT] {
    core::array::from_fn(|index| {
        sample_f128_event(
            transcript,
            INNER_PRODUCT_BATCHING_CHALLENGE_TAG,
            NO_SCOPE,
            NO_SCOPE,
            index as u32,
        )
    })
}

/// Absorbs the derived tag-5001 opening target before recursive Ligerito.
pub(crate) fn observe_opening_target(
    transcript: &mut impl PublicTranscript,
    m_p: u32,
    beta: FlockF128,
) {
    transcript.public_message(&event_header(
        OPENING_TARGET_TAG,
        NO_SCOPE,
        NO_SCOPE,
        NO_SCOPE,
        (size_of::<u32>() + FIELD_ENCODING_LEN as usize) as u64,
    ));
    transcript.public_message(&m_p);
    transcript.public_message(&from_flock_f128(beta));
}

fn sample_f128_event(
    transcript: &mut impl PublicTranscript,
    tag: u16,
    chunk: u32,
    level: u32,
    index: u32,
) -> FlockF128 {
    transcript.public_message(&event_header(
        tag,
        chunk,
        level,
        index,
        size_of::<u32>() as u64,
    ));
    transcript.public_message(&FIELD_ENCODING_LEN);
    let result = transcript.verifier_message_f128();
    transcript.public_message(&event_header(
        tag | SQUEEZE_RESULT_BIT,
        chunk,
        level,
        index,
        FIELD_ENCODING_LEN as u64,
    ));
    transcript.public_message(&result);
    as_flock_f128(result)
}

fn event_header(
    tag: u16,
    chunk: u32,
    level: u32,
    index: u32,
    payload_len: u64,
) -> [u8; EVENT_HEADER_LEN] {
    let mut header = [0; EVENT_HEADER_LEN];
    header[0..2].copy_from_slice(&tag.to_le_bytes());
    header[4..8].copy_from_slice(&chunk.to_le_bytes());
    header[8..12].copy_from_slice(&level.to_le_bytes());
    header[12..16].copy_from_slice(&index.to_le_bytes());
    header[16..24].copy_from_slice(&payload_len.to_le_bytes());
    header
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use field::F128;
    use transcript::{Encoding, build_prover, build_verifier};

    use super::*;

    #[test]
    fn ring_switch_message_matches_wire_v1() {
        let mut values = [FlockF128::ZERO; CLAIM_COUNT];
        values[0] = FlockF128::new(1, 2);
        values[CLAIM_COUNT - 1] = FlockF128::new(3, 4);
        let mut prover = build_prover(b"pcs-protocol-test", b"ring-frame");

        bind_ring_switch_message(&mut prover, &values).unwrap();
        let proof = prover.finish();

        assert_eq!(proof.narg_string.len(), EVENT_HEADER_LEN + 4 + 128 * 16);
        assert_eq!(
            &proof.narg_string[..EVENT_HEADER_LEN],
            &[
                0x01, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
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
        bind_ring_switch_message(&mut verifier, &values).unwrap();
        verifier.check_eof().unwrap();
    }

    #[test]
    fn inner_product_coordinate_message_uses_its_own_frame() {
        let mut values = [FlockF128::ZERO; CLAIM_COUNT];
        values[0] = FlockF128::new(1, 2);
        values[CLAIM_COUNT - 1] = FlockF128::new(3, 4);
        let mut prover = build_prover(b"pcs-protocol-test", b"inner-product-frame");

        write_inner_product_claims(&mut prover, &values);
        let proof = prover.finish();

        assert_eq!(
            proof.narg_string.len(),
            EVENT_HEADER_LEN + 4 + CLAIM_COUNT * 16,
        );
        assert_eq!(
            &proof.narg_string[..2],
            &INNER_PRODUCT_COORDINATES_TAG.to_le_bytes(),
        );
        assert_eq!(
            &proof.narg_string[EVENT_HEADER_LEN..EVENT_HEADER_LEN + 4],
            &(CLAIM_COUNT as u32).to_le_bytes(),
        );

        let mut verifier = build_verifier(b"pcs-protocol-test", b"inner-product-frame", &proof);
        assert_eq!(read_inner_product_claims(&mut verifier).unwrap(), values,);
        verifier.check_eof().unwrap();
    }

    #[test]
    fn inner_product_coordinate_reader_rejects_the_wrong_count() {
        let expected_header = event_header(
            INNER_PRODUCT_COORDINATES_TAG,
            SINGLE_OPENING_SCOPE,
            NO_SCOPE,
            NO_SCOPE,
            INNER_PRODUCT_COORDINATES_PAYLOAD_LEN,
        );
        let mut prover = build_prover(b"pcs-protocol-test", b"wrong-coordinate-count");
        prover.prover_message(&expected_header);
        prover.prover_message(&((CLAIM_COUNT - 1) as u32));
        let proof = prover.finish();
        let mut verifier = build_verifier(b"pcs-protocol-test", b"wrong-coordinate-count", &proof);

        assert_eq!(
            read_inner_product_claims(&mut verifier),
            Err(CommitError::MalformedProof),
        );
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
    fn single_opening_frames_match_wire_v1() {
        let challenges = (1u64..=7).map(F128::from).collect::<VecDeque<_>>();
        let mut transcript = RecordingTranscript {
            absorbed: Vec::new(),
            challenges,
            squeeze_offsets: Vec::new(),
        };

        let r_dprime = sample_ring_switch_point(&mut transcript);
        observe_opening_target(&mut transcript, 15, FlockF128::new(10, 11));
        let frames = parse_recorded_frames(&transcript.absorbed);

        assert_eq!(r_dprime.len(), LOG_PACKING);
        assert_eq!(
            transcript.squeeze_offsets,
            (0..7).map(|event| 28 + event * 68).collect::<Vec<_>>()
        );
        assert_eq!(frames.len(), 15);
        for index in 0..LOG_PACKING {
            let request = &frames[2 * index];
            let response = &frames[2 * index + 1];
            assert_eq!(request.tag, RING_SWITCH_CHALLENGE_TAG);
            assert_eq!(request.scope, (NO_SCOPE, NO_SCOPE, index as u32));
            assert_eq!(request.payload, FIELD_ENCODING_LEN.to_le_bytes());
            assert_eq!(response.tag, RING_SWITCH_CHALLENGE_TAG | SQUEEZE_RESULT_BIT);
            assert_eq!(response.scope, request.scope);
            assert_eq!(response.payload, F128::from(index as u64 + 1).to_bytes());
        }
        assert_eq!(
            frames[14],
            RecordedFrame {
                tag: OPENING_TARGET_TAG,
                scope: (NO_SCOPE, NO_SCOPE, NO_SCOPE),
                payload: [
                    15u32.to_le_bytes().as_slice(),
                    F128::new(10, 11).to_bytes().as_slice(),
                ]
                .concat(),
            }
        );
    }

    #[test]
    fn inner_product_batching_uses_128_dedicated_challenge_frames() {
        let challenges = (1u64..=CLAIM_COUNT as u64)
            .map(F128::from)
            .collect::<VecDeque<_>>();
        let mut transcript = RecordingTranscript {
            absorbed: Vec::new(),
            challenges,
            squeeze_offsets: Vec::new(),
        };

        let sampled = sample_inner_product_batching_challenges(&mut transcript);
        let frames = parse_recorded_frames(&transcript.absorbed);

        assert_eq!(sampled.len(), CLAIM_COUNT);
        assert_eq!(frames.len(), 2 * CLAIM_COUNT);
        for index in 0..CLAIM_COUNT {
            let request = &frames[2 * index];
            let response = &frames[2 * index + 1];
            assert_eq!(request.tag, INNER_PRODUCT_BATCHING_CHALLENGE_TAG);
            assert_eq!(request.scope, (NO_SCOPE, NO_SCOPE, index as u32));
            assert_eq!(request.payload, FIELD_ENCODING_LEN.to_le_bytes());
            assert_eq!(
                response.tag,
                INNER_PRODUCT_BATCHING_CHALLENGE_TAG | SQUEEZE_RESULT_BIT,
            );
            assert_eq!(response.scope, request.scope);
            assert_eq!(response.payload, F128::from(index as u64 + 1).to_bytes(),);
        }
    }
}
