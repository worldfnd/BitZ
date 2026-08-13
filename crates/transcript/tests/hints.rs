//! The hint channel: round-trips beside the narg string, never touches the
//! sponge.

use field::{F128, FqDefault};
use transcript::{Proof, build_prover, build_verifier};

const SESSION: &[u8] = b"hints-session";
const INSTANCE: &[u8] = b"hints-instance";

const MSG_1: F128 = F128::new(1, 2);
const HINT_BYTES: [u8; 5] = [0xAA; 5];
const HINT_F128: F128 = F128::new(7, 8);
const HINT_U32: u32 = 42;

fn prove() -> (Proof, F128, F128) {
    let mut prover = build_prover(SESSION, INSTANCE);
    prover.prover_message(&MSG_1);
    let c1: F128 = prover.verifier_message();
    prover.hint(&HINT_BYTES);
    prover.prover_message(&FqDefault::from(12345u128));
    prover.hint(&HINT_F128);
    let c2: F128 = prover.verifier_message();
    prover.hint(&HINT_U32);
    (prover.finish(), c1, c2)
}

#[test]
fn mixed_messages_and_hints_round_trip() {
    let (proof, c1, c2) = prove();

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    assert_eq!(verifier.prover_message::<F128>().unwrap(), MSG_1);
    assert_eq!(verifier.verifier_message::<F128>(), c1);
    assert_eq!(verifier.hint::<[u8; 5]>().unwrap(), HINT_BYTES);
    assert_eq!(
        verifier.prover_message::<FqDefault>().unwrap(),
        FqDefault::from(12345u128)
    );
    assert_eq!(verifier.hint::<F128>().unwrap(), HINT_F128);
    assert_eq!(verifier.verifier_message::<F128>(), c2);
    assert_eq!(verifier.hint::<u32>().unwrap(), HINT_U32);
    verifier.check_eof().unwrap();
}

#[test]
fn tampered_hint_leaves_challenges_unchanged() {
    let (mut proof, c1, c2) = prove();
    proof.hints[0] ^= 0xFF;

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    assert_eq!(verifier.prover_message::<F128>().unwrap(), MSG_1);
    assert_eq!(verifier.verifier_message::<F128>(), c1);
    assert_ne!(verifier.hint::<[u8; 5]>().unwrap(), HINT_BYTES);
    verifier.prover_message::<FqDefault>().unwrap();
    verifier.hint::<F128>().unwrap();
    assert_eq!(verifier.verifier_message::<F128>(), c2);
    verifier.hint::<u32>().unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn truncated_hints_fail_the_read() {
    let (mut proof, _, _) = prove();
    proof.hints.pop();

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    verifier.prover_message::<F128>().unwrap();
    verifier.verifier_message::<F128>();
    verifier.hint::<[u8; 5]>().unwrap();
    verifier.prover_message::<FqDefault>().unwrap();
    verifier.hint::<F128>().unwrap();
    verifier.verifier_message::<F128>();
    assert!(verifier.hint::<u32>().is_err());
}

#[test]
fn trailing_hint_bytes_fail_eof() {
    let (mut proof, _, _) = prove();
    proof.hints.push(0);

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    verifier.prover_message::<F128>().unwrap();
    verifier.verifier_message::<F128>();
    verifier.hint::<[u8; 5]>().unwrap();
    verifier.prover_message::<FqDefault>().unwrap();
    verifier.hint::<F128>().unwrap();
    verifier.verifier_message::<F128>();
    verifier.hint::<u32>().unwrap();
    assert!(verifier.check_eof().is_err());
}

#[test]
fn bounded_hint_bytes_round_trip() {
    let mut prover = build_prover(SESSION, INSTANCE);
    prover.hint_bytes(b"opening-proof");
    let proof = prover.finish();

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    assert_eq!(verifier.hint_bytes(13).unwrap(), b"opening-proof");
    verifier.check_eof().unwrap();

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    assert!(verifier.hint_bytes(12).is_err());
}
