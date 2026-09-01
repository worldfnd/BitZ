//! The boundary, end to end: prove, serialize, deserialize, verify.
//!
//! The point of these tests is that the verifier reaches acceptance with the
//! proof arriving as nothing but bytes. The claim is not among them: both
//! sides hold it already, and in full F2Z it is derived from the PIOP rather
//! than shipped, so there is no encoding of it to round-trip.

use host::wire_proof;
use tests::{
    EchoReduction, Instance, narrow_shape, prover_transcript, verifier_transcript, wide_shape,
};

/// Runs an honest prover and hands back what a caller would ship.
fn shipped(instance: &Instance) -> (common::OpeningClaim, Vec<u8>) {
    let mut transcript = prover_transcript();
    let claim = instance
        .prover
        .prove(
            &instance.claim,
            instance.com,
            &instance.table(),
            &EchoReduction,
            &mut transcript,
        )
        .expect("honest instance");

    (claim, wire_proof::encode(&transcript.finish()))
}

#[test]
fn a_proof_survives_the_round_trip_through_bytes() {
    for shape in [narrow_shape(), wide_shape()] {
        let instance = Instance::honest(shape, 41);
        let (proved, proof_bytes) = shipped(&instance);

        let proof = wire_proof::decode(&proof_bytes).expect("its own encoding");

        // The fold round sends one 16-byte fold per column and no hints, so
        // this is the whole proof size for it.
        let wire_proof = wire_proof::WireProof::new(&proof);
        assert_eq!(wire_proof.byte_len(), 32 + 16 * shape.columns());
        assert_eq!(wire_proof.byte_len(), proof_bytes.len());
        assert!(wire_proof.hints.is_empty());

        // Verification against the decoded proof, with the claim supplied
        // the way a caller supplies it on both sides.
        let transcript = verifier_transcript(&proof);
        let verified = instance
            .verifier
            .verify(&instance.claim, instance.com, &EchoReduction, transcript)
            .expect("honest proof");
        assert_eq!(verified, proved);
    }
}

#[test]
fn a_tampered_fold_is_left_for_the_verifier_to_catch() {
    // The container frames and does not authenticate, so a rewritten fold
    // decodes cleanly and it is the sponge -- replaying the absorptions the
    // prover made -- that refuses it. `container`'s own
    // `a_flip_anywhere_in_the_payload_is_carried_not_caught` pins the first
    // half of that; this pins the second.
    let instance = Instance::honest(wide_shape(), 43);
    let (_, proof_bytes) = shipped(&instance);

    let mut tampered = proof_bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x80;

    let proof = wire_proof::decode(&tampered).expect("a payload is opaque to the frame");
    assert_ne!(proof, wire_proof::decode(&proof_bytes).unwrap());

    let transcript = verifier_transcript(&proof);
    assert!(
        instance
            .verifier
            .verify(&instance.claim, instance.com, &EchoReduction, transcript,)
            .is_err(),
        "a tampered fold must not verify"
    );
}
