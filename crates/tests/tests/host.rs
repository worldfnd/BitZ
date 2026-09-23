//! The boundary, end to end: prove, serialize, deserialize, verify.
//!
//! The point of these tests is that the verifier reaches acceptance with the
//! proof arriving as nothing but bytes. The claim is not among them: both
//! sides hold it already, and in full BitZ it is derived from the PIOP rather
//! than shipped, so there is no encoding of it to round-trip.

use host::wire_proof;
use tests::{Instance, narrow_shape, prover_transcript, verifier_transcript, wide_shape};

/// Runs an honest prover and hands back what a caller would ship.
fn shipped(instance: &Instance) -> Vec<u8> {
    let mut transcript = prover_transcript();
    let (_, data) = instance
        .pcs
        .commit_with_ood(&instance.packed, &mut transcript)
        .unwrap();
    instance
        .prover
        .prove(
            &instance.claim,
            &instance.pcs,
            &data,
            instance.packed.clone(),
            &mut transcript,
        )
        .expect("honest instance");

    wire_proof::encode(&transcript.finish())
}

#[test]
fn a_proof_survives_the_round_trip_through_bytes() {
    for shape in [narrow_shape(), wide_shape()] {
        let instance = Instance::honest(shape, 41);
        let proof_bytes = shipped(&instance);

        let proof = wire_proof::decode(&proof_bytes).expect("its own encoding");

        // narg string: folds, GKR messages, the opening's records
        // hint stream: the opening proof
        let wire_proof = wire_proof::WireProof::new(&proof);
        assert!(wire_proof.narg_string.len() > 16 * shape.columns());
        assert_eq!(wire_proof.byte_len(), proof_bytes.len());

        // Verification against the decoded proof, with the claim supplied
        // the way a caller supplies it on both sides.
        instance
            .verifier
            .verify(
                &instance.claim,
                &instance.pcs,
                instance.com,
                verifier_transcript(&proof),
            )
            .expect("honest proof");
    }
}

#[test]
fn a_tampered_fold_is_left_for_the_verifier_to_catch() {
    // The container frames but does not authenticate: a tampered fold decodes
    // cleanly and the sponge refuses it on replay.
    let shape = wide_shape();
    let instance = Instance::honest(shape, 43);
    let proof_bytes = shipped(&instance);

    let mut tampered = proof_bytes.clone();
    let last_fold_byte = 32 + 16 * shape.columns() - 1;
    tampered[last_fold_byte] ^= 0x80;

    let proof = wire_proof::decode(&tampered).expect("a payload is opaque to the frame");
    assert_ne!(proof, wire_proof::decode(&proof_bytes).unwrap());

    let transcript = verifier_transcript(&proof);
    assert!(
        instance
            .verifier
            .verify(&instance.claim, &instance.pcs, instance.com, transcript)
            .is_err(),
        "a tampered fold must not verify"
    );
}
