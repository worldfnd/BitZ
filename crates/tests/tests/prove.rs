//! The top-level prove and verify, through the real opening.

use common::Root;
use field::Fq;
use pcs::CommitError;
use tests::{
    HonestStub, Instance, large_shape, narrow_shape, prover_transcript, verifier_transcript,
    wide_shape,
};
use transcript::Proof;
use verifier::{ReceiveError, VerifyError};

fn prove(instance: &Instance) -> Proof {
    let mut transcript = prover_transcript();
    instance
        .prover
        .prove(
            &instance.claim,
            &instance.pcs,
            &instance.data,
            instance.packed.clone(),
            &HonestStub,
            &mut transcript,
        )
        .expect("honest instance");
    transcript.finish()
}

#[test]
fn an_honest_proof_verifies_on_every_shape_the_profile_admits() {
    for shape in [narrow_shape(), wide_shape(), large_shape()] {
        let instance = Instance::honest(shape, 31);
        let proof = prove(&instance);

        instance
            .verifier
            .verify(
                &instance.claim,
                &instance.pcs,
                instance.com,
                &HonestStub,
                verifier_transcript(&proof),
            )
            .unwrap_or_else(|error| panic!("t = {}: {error:?}", shape.log_rows()));
    }
}

#[test]
fn a_proof_replayed_under_a_different_commitment_is_refused() {
    let instance = Instance::honest(narrow_shape(), 32);
    let proof = prove(&instance);

    // The root is not in the proof, so this is not a decode failure: the two
    // sponges diverge at step 1, the point the stub squeezes moves with them,
    // and the opening is left discharging a claim at the wrong point.
    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            &instance.pcs,
            Root([0xffu8; 32]),
            &HonestStub,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(CommitError::VerificationFailed))
    );
}

#[test]
fn the_statement_is_bound_before_the_first_challenge() {
    let instance = Instance::honest(narrow_shape(), 33);
    let proof = prove(&instance);

    // Same folds, same commitment, a claim that differs only in its claimed
    // value. The fold's own reconstruction rejects it, which is the check the
    // binding backs up rather than replaces.
    let retargeted = instance.with_target(instance.claim.target() + Fq::from(1u128));
    assert_eq!(
        instance.verifier.verify(
            &retargeted,
            &instance.pcs,
            instance.com,
            &HonestStub,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Fold(ReceiveError::TargetMismatch))
    );
}

#[test]
fn a_proof_with_trailing_bytes_is_refused() {
    let instance = Instance::honest(narrow_shape(), 34);
    let mut proof = prove(&instance);
    proof.hints.push(0);

    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            &instance.pcs,
            instance.com,
            &HonestStub,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::TrailingData)
    );
}
