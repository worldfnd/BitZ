//! The top-level prove and verify, against a stubbed reduction.

mod fixtures;

use common::Root;
use field::Fq;
use fixtures::{
    EchoReduction, Instance, narrow_shape, prover_transcript, verifier_transcript, wide_shape,
};
use verifier::{ReceiveError, VerifyError};

#[test]
fn the_two_sides_agree_on_every_shape_the_profile_admits() {
    for shape in [narrow_shape(), wide_shape()] {
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

        let mut transcript = verifier_transcript(&proof);
        let verified = instance
            .verifier
            .verify(
                &instance.claim,
                instance.com,
                &EchoReduction,
                &mut transcript,
            )
            .expect("honest proof");

        assert_eq!(proved, verified, "t = {}", shape.t());
        assert_eq!(proved.point.len(), shape.m());
        // Not an acceptance: step 5.3 is absent, so the opening the claim
        // feeds has not run and there is nothing yet to exhaust the streams.
        transcript.check_eof().expect("nothing beyond the fold yet");
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

    let mut transcript = verifier_transcript(&proof);
    let verified = instance
        .verifier
        .verify(
            &instance.claim,
            Root([0xffu8; 32]),
            &EchoReduction,
            &mut transcript,
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
    let mut transcript = verifier_transcript(&proof);
    assert_eq!(
        instance
            .verifier
            .verify(&retargeted, instance.com, &EchoReduction, &mut transcript,),
        Err(VerifyError::Fold(ReceiveError::TargetMismatch))
    );
}
