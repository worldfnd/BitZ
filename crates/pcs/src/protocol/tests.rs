use std::collections::VecDeque;

use field::F128;
use proptest::prelude::*;
use transcript::{build_prover, build_verifier};

use super::*;
use crate::{HashKind, LigeritoProfile, OpeningQuery, ScopedOpeningQuery};

proptest! {
    #[test]
    fn statement_binding_matches_between_roles(
        root in any::<[u8; 32]>(),
        point_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..32),
        target_words in (any::<u64>(), any::<u64>()),
    ) {
        let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let query = OpeningQuery {
            point: point_words
                .iter()
                .map(|&(lo, hi)| F128::new(lo, hi))
                .collect(),
            target: F128::new(target_words.0, target_words.1),
        };
        let scoped_query = ScopedOpeningQuery::new(7, &query);

        let mut prover = build_prover(b"pcs-protocol-test", b"statement-binding");
        bind_statement(
            &pcs,
            &root,
            core::slice::from_ref(&scoped_query),
            &mut prover,
        );
        let expected = prover.verifier_message::<F128>();
        let proof = prover.finish();

        let mut verifier = build_verifier(
            b"pcs-protocol-test",
            b"statement-binding",
            &proof,
        );
        bind_statement(
            &pcs,
            &root,
            core::slice::from_ref(&scoped_query),
            &mut verifier,
        );
        prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
        prop_assert!(verifier.check_eof().is_ok());
    }
}

fn statement_challenge(queries: &[ScopedOpeningQuery<'_>]) -> F128 {
    let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let mut prover = build_prover(b"pcs-protocol-test", b"batch-binding");
    bind_statement(&pcs, &[7; 32], queries, &mut prover);
    prover.verifier_message::<F128>()
}

#[test]
fn statement_binding_commits_to_batch_order_and_count() {
    let first = OpeningQuery {
        point: vec![F128::from(1u64), F128::from(2u64)],
        target: F128::from(3u64),
    };
    let second = OpeningQuery {
        point: vec![F128::from(4u64), F128::from(5u64)],
        target: F128::from(6u64),
    };

    let ordered = statement_challenge(&[
        ScopedOpeningQuery::new(0, &first),
        ScopedOpeningQuery::new(2, &second),
    ]);
    let reversed = statement_challenge(&[
        ScopedOpeningQuery::new(0, &second),
        ScopedOpeningQuery::new(2, &first),
    ]);
    let changed_scope = statement_challenge(&[
        ScopedOpeningQuery::new(0, &first),
        ScopedOpeningQuery::new(3, &second),
    ]);
    let prefix = statement_challenge(&[ScopedOpeningQuery::new(0, &first)]);

    assert_ne!(ordered, reversed);
    assert_ne!(ordered, changed_scope);
    assert_ne!(ordered, prefix);
}

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
        let mut verifier = build_verifier(b"pcs-protocol-test", b"mismatched-ring-values", &proof);

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

#[test]
fn batch_validation_requires_strictly_increasing_scopes() {
    let query = OpeningQuery {
        point: vec![F128::default(); 22],
        target: F128::default(),
    };

    assert_eq!(
        validate_batch(
            &[
                ScopedOpeningQuery::new(2, &query),
                ScopedOpeningQuery::new(2, &query),
            ],
            22,
        ),
        Err(CommitError::InvalidClaimScopeOrder)
    );
    assert_eq!(
        validate_batch(
            &[
                ScopedOpeningQuery::new(3, &query),
                ScopedOpeningQuery::new(2, &query),
            ],
            22,
        ),
        Err(CommitError::InvalidClaimScopeOrder)
    );
    assert_eq!(
        validate_batch(&[ScopedOpeningQuery::new(u32::MAX, &query)], 22),
        Err(CommitError::InvalidClaimScope)
    );
}
