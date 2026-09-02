//! The top-level prove and verify, against a stubbed reduction.

use common::Root;
use field::Fq;
use tests::{
    EchoReduction, Instance, large_shape, narrow_shape, prover_transcript, verifier_transcript,
    wide_shape,
};
use verifier::{ReceiveError, VerifyError};

#[test]
fn the_two_sides_agree_on_every_shape_the_profile_admits() {
    for shape in [narrow_shape(), wide_shape(), large_shape()] {
        let instance = Instance::honest(shape, 31);

        let mut transcript = prover_transcript();
        let proved = instance
            .prover
            .prove(
                &instance.claim,
                instance.com,
                &instance.table(),
                &EchoReduction,
                &mut transcript,
            )
            .expect("honest instance");
        let proof = transcript.finish();

        // `verify` consumes the transcript and asserts both streams are spent,
        // so an honest round trip failing here would mean a stray record. Not
        // an acceptance: step 6 is absent, so nothing has discharged the claim.
        let verified = instance
            .verifier
            .verify(
                &instance.claim,
                instance.com,
                &EchoReduction,
                verifier_transcript(&proof),
            )
            .expect("honest proof");

        assert_eq!(proved, verified, "t = {}", shape.log_rows());
        assert_eq!(proved.point.len(), shape.log_bits());
    }
}

#[test]
fn the_commitment_is_bound_before_the_first_challenge() {
    // Replaying under a different root must move the challenge. The root is
    // not in the proof, so this is not a decode failure -- the two sides
    // simply stop agreeing, which is what binding it buys.
    let instance = Instance::honest(narrow_shape(), 32);

    let mut transcript = prover_transcript();
    let proved = instance
        .prover
        .prove(
            &instance.claim,
            instance.com,
            &instance.table(),
            &EchoReduction,
            &mut transcript,
        )
        .unwrap();
    let proof = transcript.finish();

    let verified = instance
        .verifier
        .verify(
            &instance.claim,
            Root([0xffu8; 32]),
            &EchoReduction,
            verifier_transcript(&proof),
        )
        .expect("every record still decodes and every check still passes");

    assert_ne!(proved.point, verified.point);
}

#[test]
fn the_statement_is_bound_before_the_first_challenge() {
    let instance = Instance::honest(narrow_shape(), 33);

    let mut transcript = prover_transcript();
    instance
        .prover
        .prove(
            &instance.claim,
            instance.com,
            &instance.table(),
            &EchoReduction,
            &mut transcript,
        )
        .unwrap();
    let proof = transcript.finish();

    // Same folds, same commitment, a claim that differs only in its
    // claimed value. The fold's own reconstruction rejects it, which is the
    // check the binding backs up rather than replaces.
    let retargeted = instance.with_target(instance.claim.target() + Fq::from(1u128));
    assert_eq!(
        instance.verifier.verify(
            &retargeted,
            instance.com,
            &EchoReduction,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Fold(ReceiveError::TargetMismatch))
    );
}

#[test]
fn a_proof_with_trailing_bytes_is_refused() {
    let instance = Instance::honest(narrow_shape(), 34);

    let mut transcript = prover_transcript();
    instance
        .prover
        .prove(
            &instance.claim,
            instance.com,
            &instance.table(),
            &EchoReduction,
            &mut transcript,
        )
        .unwrap();
    let mut proof = transcript.finish();
    proof.hints.push(0);

    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            instance.com,
            &EchoReduction,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::TrailingData)
    );
}
