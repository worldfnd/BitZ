//! The top-level prove and verify, through the real opening.

use common::{Root, TableError};
use field::{F128, Fq};
use pcs::{CommitError, HashKind, LigeritoProfile, Pcs};
use prover::ProveError;
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

#[test]
fn an_opening_against_another_commitment_is_refused() {
    // Everything but the opening lines up: the root the prover binds is the one
    // the verifier is given, the folds are over the witness the claim describes,
    // and the stub's evaluation is true of that witness. Only the codeword and
    // the Merkle tree the opening reads belong to a different commitment.
    let proved = Instance::honest(narrow_shape(), 35);
    let committed = Instance::honest(narrow_shape(), 36);

    let mut transcript = prover_transcript();
    proved
        .prover
        .prove(
            &proved.claim,
            &proved.pcs,
            &committed.data,
            proved.packed.clone(),
            &HonestStub,
            &mut transcript,
        )
        .expect("the prover checks the claim, not the commitment behind it");
    let proof = transcript.finish();

    assert_eq!(
        proved.verifier.verify(
            &proved.claim,
            &proved.pcs,
            committed.com,
            &HonestStub,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(CommitError::VerificationFailed))
    );
}

#[test]
fn a_tampered_opening_proof_is_refused() {
    // The opening rides the hint channel, which the sponge never sees, so
    // nothing upstream of the opening notices this. The opening itself must.
    let instance = Instance::honest(narrow_shape(), 37);
    let mut proof = prove(&instance);
    let middle = proof.hints.len() / 2;
    proof.hints[middle] ^= 0xff;

    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            &instance.pcs,
            instance.com,
            &HonestStub,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(CommitError::VerificationFailed))
    );
}

#[test]
fn a_proof_verified_under_a_different_profile_is_refused() {
    // The profile is not in the frame step 1 absorbs, so what rejects this is
    // the opening binding its own parameters: a different profile encodes
    // differently, the two sponges part, and the ring-switch check fails.
    let instance = Instance::honest(narrow_shape(), 38);
    let slim = Pcs::new(
        instance.params.shape(),
        LigeritoProfile::Slim,
        HashKind::Blake3,
    )
    .unwrap();
    let proof = prove(&instance);

    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            &slim,
            instance.com,
            &HonestStub,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(CommitError::VerificationFailed))
    );
}

#[test]
fn a_witness_of_the_wrong_length_is_refused_before_anything_is_written() {
    let instance = Instance::honest(narrow_shape(), 39);

    let mut transcript = prover_transcript();
    assert_eq!(
        instance.prover.prove(
            &instance.claim,
            &instance.pcs,
            &instance.data,
            vec![F128::default(); 10],
            &HonestStub,
            &mut transcript,
        ),
        Err(ProveError::Witness(TableError::BitCountMismatch))
    );

    // The shape is checked before the first absorb, so a rejected witness
    // leaves no half-written proof behind.
    let proof = transcript.finish();
    assert!(proof.narg_string.is_empty() && proof.hints.is_empty());
}
