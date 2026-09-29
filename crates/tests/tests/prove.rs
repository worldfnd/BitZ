//! The top-level prove and verify, through the real opening.

use common::{OpeningQuery, Root, Shape, TableError};
use field::{F128, Fq};
use num_traits::{ConstOne, ConstZero};
use pcs::{CommitScheme, Pcs, StatementBinding, VerifyError as PcsVerifyError};
use prover::ProveError;
use tests::{
    Instance, large_shape, narrow_shape, prover_transcript, verifier_transcript, wide_shape,
};
use transcript::{Proof, SecurityLevel};
use verifier::{ReceiveError, VerifyError};

fn prove(instance: &Instance) -> Proof {
    let mut transcript = prover_transcript();
    let (_, data) = instance
        .pcs
        .commit(&instance.packed, &mut transcript)
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
    transcript.finish()
}

#[test]
fn an_honest_proof_verifies_on_the_test_shapes() {
    for shape in [narrow_shape(), wide_shape(), large_shape()] {
        let instance = Instance::honest(shape, 31);
        let proof = prove(&instance);

        instance
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
    let instance = Instance::honest(narrow_shape(), 32);
    let proof = prove(&instance);

    // Binding a different root changes the fold batching point, so GKR rejects.
    assert_eq!(
        instance.verify(
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
    let instance = Instance::honest(narrow_shape(), 33);
    let proof = prove(&instance);

    // Same folds, same commitment, a claim that differs only in its claimed
    // value. The fold's own reconstruction rejects it, which is the check the
    // binding backs up rather than replaces.
    let retargeted = instance.with_target(instance.claim.target() + Fq::ONE);
    assert_eq!(
        instance.verify(
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
    let instance = Instance::honest(narrow_shape(), 34);
    let mut proof = prove(&instance);
    proof.hints.push(0);

    assert_eq!(
        instance.verify(
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
    let proved = Instance::honest(narrow_shape(), 35);
    let committed = Instance::honest(narrow_shape(), 36);

    let mut transcript = prover_transcript();
    let (_, data) = committed
        .pcs
        .commit(&committed.packed, &mut transcript)
        .unwrap();
    proved
        .prover
        .prove(
            &proved.claim,
            &proved.pcs,
            &data,
            proved.packed.clone(),
            &mut transcript,
        )
        .expect("the prover checks the claim, not the commitment behind it");
    let proof = transcript.finish();

    assert_eq!(
        proved.verify(
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
    let instance = Instance::honest(narrow_shape(), 37);
    let mut proof = prove(&instance);
    let middle = proof.hints.len() / 2;
    proof.hints[middle] ^= 0xff;

    assert!(matches!(
        instance.verify(
            &instance.claim,
            &instance.pcs,
            instance.com,
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(
            PcsVerifyError::MalformedProof | PcsVerifyError::VerificationFailed
        ))
    ));
}

#[test]
fn a_proof_verified_under_a_different_security_target_is_refused() {
    // PCS parameters enter the transcript before the first fold challenge.
    let instance = Instance::honest(narrow_shape(), 38);
    let other_pcs = Pcs::new(instance.params.shape(), SecurityLevel::Bits128).unwrap();
    let proof = prove(&instance);

    assert!(
        instance
            .verify(
                &instance.claim,
                &other_pcs,
                instance.com,
                verifier_transcript(&proof),
            )
            .is_err()
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
            vec![F128::ZERO; 10],
            &mut transcript,
        ),
        Err(ProveError::Witness(TableError::BitCountMismatch))
    );

    // The shape is checked before the first absorb, so a rejected witness
    // leaves no half-written proof behind.
    let proof = transcript.finish();
    assert!(proof.narg_string.is_empty() && proof.hints.is_empty());
}

#[test]
fn explicit_security_targets_verify_and_reject_replay_or_tampering() {
    let mut instance = Instance::honest(narrow_shape(), 40);
    for level in [SecurityLevel::Bits100, SecurityLevel::Bits128] {
        instance.pcs = Pcs::new(instance.params.shape(), level).unwrap();
        (instance.com, instance.data) = instance
            .pcs
            .commit(&instance.packed, &mut prover_transcript())
            .unwrap();
        let proof = prove(&instance);
        instance
            .verify(
                &instance.claim,
                &instance.pcs,
                instance.com,
                verifier_transcript(&proof),
            )
            .unwrap();

        let other_level = match level {
            SecurityLevel::Bits100 => SecurityLevel::Bits128,
            SecurityLevel::Bits128 => SecurityLevel::Bits100,
        };
        let other_pcs = Pcs::new(instance.params.shape(), other_level).unwrap();
        assert!(
            instance
                .verify(
                    &instance.claim,
                    &other_pcs,
                    instance.com,
                    verifier_transcript(&proof),
                )
                .is_err()
        );

        let wrong_target = instance.with_target(instance.claim.target() + Fq::ONE);
        assert_eq!(
            instance.verify(
                &wrong_target,
                &instance.pcs,
                instance.com,
                verifier_transcript(&proof),
            ),
            Err(VerifyError::Fold(ReceiveError::TargetMismatch))
        );

        if level == SecurityLevel::Bits128 {
            // The first nonce follows the column folds.
            let mut changed = proof.clone();
            changed.narg_string[16 * instance.params.shape().columns()] ^= 0xff;
            assert!(
                instance
                    .verify(
                        &instance.claim,
                        &instance.pcs,
                        instance.com,
                        verifier_transcript(&changed),
                    )
                    .is_err()
            );
        }

        // Both policies use the same commitment geometry, but different opening configurations.
        // Retained data must match the security target as well as the commitment shape.
        let query = OpeningQuery::Mle {
            point: vec![F128::ZERO; instance.params.shape().log_bits()],
            target: F128::from(instance.packed[0].lo & 1),
        };
        let mut transcript = prover_transcript();
        assert_eq!(
            other_pcs.prove_lin(
                &instance.data,
                instance.packed.clone(),
                &query,
                StatementBinding::Bind,
                &mut transcript,
            ),
            Err(pcs::ProveError::ProverDataMismatch)
        );
        assert_eq!(transcript.finish(), Proof::default());
    }
}

#[test]
fn a_mismatched_direct_commitment_size_is_rejected_before_transcript_mutation() {
    let instance = Instance::honest(narrow_shape(), 41);
    let pcs = Pcs::new(&Shape::new(7, 16).unwrap(), SecurityLevel::Bits128).unwrap();
    let mut transcript = prover_transcript();
    assert_eq!(
        instance.prover.prove(
            &instance.claim,
            &pcs,
            &instance.data,
            instance.packed.clone(),
            &mut transcript,
        ),
        Err(ProveError::ParameterMismatch)
    );
    assert_eq!(transcript.finish(), Proof::default());
    assert_eq!(
        instance.verify(
            &instance.claim,
            &pcs,
            instance.com,
            verifier_transcript(&Proof::default()),
        ),
        Err(VerifyError::ParameterMismatch)
    );
}
