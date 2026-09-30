//! The top-level prove and verify, through the real opening.

use common::{Root, TableError};
use field::F128;
use num_traits::{ConstOne, ConstZero};
use pcs::{HashKind, LigeritoProfile, Pcs, VerifyError as PcsVerifyError};
use prover::ProveError;
use tests::{Instance, narrow_shape, verifier_transcript, wide_shape};
use transcript::Proof;
use verifier::{ReceiveError, VerifyError};

type F = field::FqDefault;

fn prove(instance: &mut Instance<F>) -> Proof {
    let mut transcript = instance.transcript.take().unwrap();
    instance
        .prover
        .prove(
            &instance.claim,
            &instance.pcs,
            &instance.data,
            instance.packed.clone(),
            &mut transcript,
        )
        .expect("honest instance");
    transcript.finish()
}

#[test]
fn an_honest_proof_verifies_on_both_floor_shapes() {
    for shape in [narrow_shape(), wide_shape()] {
        let mut instance = Instance::<F>::honest(shape, 31);
        let proof = prove(&mut instance);

        instance
            .verifier
            .verify(
                &instance.claim,
                &instance.pcs,
                instance.com,
                verifier_transcript(&proof),
            )
            .unwrap_or_else(|error| panic!("t = {}: {error:?}", shape.log_rows()));
    }
}

#[test]
fn a_proof_replayed_under_a_different_commitment_is_refused() {
    let mut instance = Instance::<F>::honest(narrow_shape(), 32);
    let proof = prove(&mut instance);

    // Binding a different root changes the fold batching point, so GKR rejects.
    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            &instance.pcs,
            Root([0xffu8; 32]),
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Reduction(verifier::ReduceError::GKR))
    );
}

#[test]
fn the_statement_is_bound_before_the_first_challenge() {
    let mut instance = Instance::<F>::honest(narrow_shape(), 33);
    let proof = prove(&mut instance);

    // Same folds, same commitment, a claim that differs only in its claimed
    // value. The fold's own reconstruction rejects it, which is the check the
    // binding backs up rather than replaces.
    let retargeted = instance.with_target(instance.claim.target() + F::ONE);
    assert_eq!(
        instance.verifier.verify(
            &retargeted,
            &instance.pcs,
            instance.com,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Fold(ReceiveError::TargetMismatch))
    );
}

#[test]
fn a_proof_with_trailing_bytes_is_refused() {
    let mut instance = Instance::<F>::honest(narrow_shape(), 34);
    let mut proof = prove(&mut instance);
    proof.hints.push(0);

    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            &instance.pcs,
            instance.com,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::TrailingData)
    );
}

#[test]
fn an_opening_against_another_commitment_is_refused() {
    // Everything but the opening lines up: the root the prover binds is the one
    // the verifier is given, the folds are over the witness the claim describes,
    // and the GKR claim is true of that witness. Only the codeword and
    // the Merkle tree the opening reads belong to a different commitment.
    let proved = Instance::<F>::honest(narrow_shape(), 35);
    let mut committed = Instance::<F>::honest(narrow_shape(), 36);

    let mut transcript = committed.transcript.take().unwrap();
    proved
        .prover
        .prove(
            &proved.claim,
            &proved.pcs,
            &committed.data,
            proved.packed.clone(),
            &mut transcript,
        )
        .expect("the prover checks the claim, not the commitment behind it");
    let proof = transcript.finish();

    assert_eq!(
        proved.verifier.verify(
            &proved.claim,
            &proved.pcs,
            committed.com,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(PcsVerifyError::VerificationFailed))
    );
}

#[test]
fn a_tampered_opening_proof_is_refused() {
    // The opening rides the hint channel, which the sponge never sees, so
    // nothing upstream of the opening notices this. The opening itself must.
    let mut instance = Instance::<F>::honest(narrow_shape(), 37);
    let mut proof = prove(&mut instance);
    let middle = proof.hints.len() / 2;
    proof.hints[middle] ^= 0xff;

    assert_eq!(
        instance.verifier.verify(
            &instance.claim,
            &instance.pcs,
            instance.com,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(PcsVerifyError::VerificationFailed))
    );
}

#[test]
fn a_proof_verified_under_a_different_profile_is_refused() {
    // OOD binds PCS parameters before the first fold challenge, so a different
    // profile changes the fold transcript and GKR rejects.
    let mut instance = Instance::<F>::honest(narrow_shape(), 38);
    let slim = Pcs::new(
        instance.params.shape(),
        LigeritoProfile::Slim,
        HashKind::Blake3,
    )
    .unwrap();
    let proof = prove(&mut instance);

    assert!(matches!(
        instance.verifier.verify(
            &instance.claim,
            &slim,
            instance.com,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Reduction(_))
    ));
}

#[test]
fn a_witness_of_the_wrong_length_is_refused_without_changing_the_commitment_transcript() {
    let mut instance = Instance::<F>::honest(narrow_shape(), 39);

    let mut transcript = instance.transcript.take().unwrap();
    assert_eq!(
        instance.prover.prove(
            &instance.claim,
            &instance.pcs,
            &instance.data,
            vec![F128::ZERO; 10],
            &mut transcript,
        ),
        Err(ProveError::Witness(TableError::BitCountMismatch))
    );

    let next_challenge = transcript.verifier_message::<F128>();
    let proof = transcript.finish();
    let mut verifier = verifier_transcript(&proof);
    instance
        .pcs
        .receive_commitment(instance.com, &mut verifier)
        .unwrap();
    assert_eq!(verifier.verifier_message::<F128>(), next_challenge);
    verifier.check_eof().unwrap();
}
