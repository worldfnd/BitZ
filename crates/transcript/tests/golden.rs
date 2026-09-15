//! Pins the challenge sequence and wire bytes for one fixed transcript.
//! A mismatch means the framing changed and every existing proof is invalid.

use field::{F128, FqDefault};
use transcript::{Proof, build_prover, build_verifier};

const SESSION: &[u8] = b"golden-session";
const INSTANCE: &[u8] = b"golden-instance";

const MSG_F128: F128 = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
const MSG_FQ: u128 = 12345;
const HINT: [u8; 5] = [0xAA; 5];

const NARG: &str = "efcdab89674523011032547698badcfe39300000000000000000000000000000";
const HINTS: &str = "aaaaaaaaaa";
const C1: &str = "1d1bcda36aa1541c3752f7b23fedef81";
const C2: &str = "e80729d4aa76e8e4d28dd085143eb3d5";

fn prove() -> (Proof, F128, F128) {
    let mut prover = build_prover(SESSION, INSTANCE);
    prover.prover_message(&MSG_F128);
    let c1: F128 = prover.verifier_message();
    prover.hint(&HINT);
    prover.prover_message(&FqDefault::from(MSG_FQ));
    let c2: F128 = prover.verifier_message();
    (prover.finish(), c1, c2)
}

#[test]
fn golden_transcript_bytes_are_pinned() {
    let (proof, c1, c2) = prove();
    assert_eq!(hex(&proof.narg_string), NARG);
    assert_eq!(hex(&proof.hints), HINTS);
    assert_eq!(hex(&c1.to_bytes()), C1);
    assert_eq!(hex(&c2.to_bytes()), C2);
}

#[test]
fn verifier_replays_the_golden_transcript() {
    let (proof, c1, c2) = prove();

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    assert_eq!(verifier.prover_message::<F128>().unwrap(), MSG_F128);
    assert_eq!(verifier.verifier_message::<F128>(), c1);
    assert_eq!(verifier.hint::<[u8; 5]>().unwrap(), HINT);
    assert_eq!(
        verifier.prover_message::<FqDefault>().unwrap(),
        FqDefault::from(MSG_FQ)
    );
    assert_eq!(verifier.verifier_message::<F128>(), c2);
    verifier.check_eof().unwrap();
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
