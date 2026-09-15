//! The fold round, prover against verifier.

use common::{BitZParams, FoldError, LinearClaim};
use field::{F128, Fq, gf128::smallest_generator};
use num_traits::{ConstOne, ConstZero};
use prover::{BitZProver, SendError};
use tests::{
    Instance, Q, WINDOW, narrow_shape, prover_transcript, verifier_transcript, wide_shape,
};
use transcript::Proof;
use verifier::{BitZVerifier, ReceiveError};

/// Runs an honest prover and returns the round it produced with its proof.
fn prove(instance: &Instance) -> (common::Fold, Proof) {
    let mut transcript = prover_transcript();
    let round = instance
        .prover
        .send_fold(&instance.claim, &instance.table(), &mut transcript)
        .unwrap();
    (round, transcript.finish())
}

/// A proof carrying `folds` and nothing else, which is the whole wire format
/// of this round.
fn forge(folds: &[u128]) -> Proof {
    let mut transcript = prover_transcript();
    for &fold in folds {
        transcript.prover_message(&fold.to_le_bytes());
    }
    transcript.finish()
}

#[test]
fn the_two_sides_agree_on_every_shape_the_profile_admits() {
    for shape in [narrow_shape(), wide_shape()] {
        let instance = Instance::honest(shape, 7);
        let (sent, proof) = prove(&instance);

        let mut transcript = verifier_transcript(&proof);
        let received = instance
            .verifier
            .receive_fold(&instance.claim, &mut transcript)
            .expect("honest proof");

        assert_eq!(sent, received, "t = {}", shape.log_rows());
        assert_eq!(received.row_images.len(), shape.rows());
        assert_eq!(received.zeta.len(), shape.log_columns());
        transcript.check_eof().expect("both streams exhausted");
    }
}

#[test]
fn the_proof_carries_only_the_folds() {
    // Sending the images too would double this round's proof size: at
    // k_2 = 2^21 the second copy is 32 MiB.
    let instance = Instance::honest(narrow_shape(), 2);
    let (_, proof) = prove(&instance);

    assert_eq!(
        proof.narg_string.len(),
        16 * instance.params.shape().columns()
    );
    assert!(proof.hints.is_empty());
}

#[test]
fn the_challenge_depends_on_the_folds() {
    // The folds are written and absorbed by the same call, so the challenge
    // must move when they do. That is what binds the images the grand product
    // consumes, since `u -> g^u` is injective over the admitted range.
    let instance = Instance::honest(narrow_shape(), 3);
    let (round, _) = prove(&instance);

    let echoed = forge(&round.folds);
    let mut transcript = verifier_transcript(&echoed);
    let replayed: Vec<u128> = (0..instance.params.shape().columns())
        .map(|_| {
            transcript
                .prover_message::<[u8; 16]>()
                .map(u128::from_le_bytes)
                .unwrap()
        })
        .collect();
    assert_eq!(replayed, round.folds);

    let zeta: Vec<F128> = (0..instance.params.shape().log_columns())
        .map(|_| transcript.verifier_message())
        .collect();
    assert_eq!(zeta, round.zeta);
}

#[test]
fn a_fold_at_the_bound_is_accepted_and_one_past_it_is_not() {
    // Every weight at `q - 1` and every bit set puts the fold exactly on
    // `k_1 (q - 1)`, the largest value the verifier may accept.
    let shape = narrow_shape();
    let packed = vec![F128::new(u64::MAX, u64::MAX); (1 << shape.log_bits()) / 128];

    let fold = (shape.rows() as u128) * (Q - 1);
    let params = BitZParams::<Q>::new(shape, smallest_generator()).unwrap();
    let prover = BitZProver::new(params, WINDOW);
    let verifier = BitZVerifier::new(params, WINDOW);
    let table = params.table(&packed).unwrap();
    let claim = LinearClaim::new(
        &params,
        vec![Fq::from(Q - 1); shape.rows()],
        vec![Fq::from(1u128); shape.columns()],
        Fq::from(fold) * Fq::from(shape.columns() as u128),
    )
    .unwrap();

    let mut transcript = prover_transcript();
    let round = prover.send_fold(&claim, &table, &mut transcript).unwrap();
    assert!(round.folds.iter().all(|&value| value == fold));

    let proof = transcript.finish();
    let mut transcript = verifier_transcript(&proof);
    assert!(verifier.receive_fold(&claim, &mut transcript).is_ok());

    let over = forge(&vec![fold + 1; shape.columns()]);
    let mut transcript = verifier_transcript(&over);
    assert_eq!(
        verifier.receive_fold(&claim, &mut transcript),
        Err(ReceiveError::FoldOutOfRange)
    );
}

#[test]
fn the_range_check_fires_before_the_reconstruction() {
    // A proof violating both. The order is fixed, so the earlier obligation is
    // the one that must be reported.
    let instance = Instance::honest(narrow_shape(), 6);
    let shape = instance.params.shape();
    let over = vec![instance.params.fold_bound() + 1; shape.columns()];

    let forged = forge(&over);
    let mut transcript = verifier_transcript(&forged);
    assert_eq!(
        instance
            .verifier
            .receive_fold(&instance.claim, &mut transcript),
        Err(ReceiveError::FoldOutOfRange)
    );
    // The same folds also fail the reconstruction, so the assertion above is
    // about ordering and not about only one check being violated.
    assert_ne!(
        common::reconstruct(&instance.claim, &over),
        Ok(instance.claim.target())
    );
}

#[test]
fn folds_that_do_not_reconstruct_the_target_are_rejected() {
    let instance = Instance::honest(narrow_shape(), 5);
    let (_, proof) = prove(&instance);

    // The proof is honest; the claim it is replayed against is not.
    let retargeted = instance.with_target(instance.claim.target() + Fq::from(1u128));
    let mut transcript = verifier_transcript(&proof);
    assert_eq!(
        instance.verifier.receive_fold(&retargeted, &mut transcript),
        Err(ReceiveError::TargetMismatch)
    );
}

#[test]
fn a_witness_of_a_different_shape_is_refused_before_anything_is_written() {
    let instance = Instance::honest(narrow_shape(), 8);
    let other = Instance::honest(wide_shape(), 8);

    let mut transcript = prover_transcript();
    assert_eq!(
        instance
            .prover
            .send_fold(&instance.claim, &other.table(), &mut transcript),
        Err(SendError::ShapeMismatch)
    );
    assert!(transcript.finish().narg_string.is_empty());
}

#[test]
fn a_truncated_proof_is_refused_rather_than_read_past() {
    let instance = Instance::honest(narrow_shape(), 10);
    let (_, proof) = prove(&instance);

    let mut short = proof.clone();
    short.narg_string.truncate(proof.narg_string.len() - 16);
    let mut transcript = verifier_transcript(&short);
    assert_eq!(
        instance
            .verifier
            .receive_fold(&instance.claim, &mut transcript),
        Err(ReceiveError::MalformedProof)
    );
}

#[test]
fn an_all_zero_witness_folds_to_zero_and_still_round_trips() {
    let shape = narrow_shape();
    let packed = vec![F128::ZERO; (1 << shape.log_bits()) / 128];
    let params = BitZParams::<Q>::new(shape, smallest_generator()).unwrap();
    let prover = BitZProver::new(params, WINDOW);
    let verifier = BitZVerifier::new(params, WINDOW);
    let claim = LinearClaim::new(
        &params,
        vec![Fq::from(Q - 1); shape.rows()],
        vec![Fq::from(3u128); shape.columns()],
        Fq::from(0u128),
    )
    .unwrap();
    let table = params.table(&packed).unwrap();

    let mut transcript = prover_transcript();
    let round = prover.send_fold(&claim, &table, &mut transcript).unwrap();
    assert!(round.folds.iter().all(|&fold| fold == 0));
    assert!(round.images.iter().all(|&image| image == F128::ONE));

    let proof = transcript.finish();
    let mut transcript = verifier_transcript(&proof);
    assert_eq!(
        verifier.receive_fold(&claim, &mut transcript).unwrap(),
        round
    );
}

#[test]
fn a_round_is_refused_when_its_parts_do_not_match_the_shape() {
    // `Fold::new` is public and reachable outside the two openers, which
    // derive every length from one shape and so cannot trigger this.
    let shape = narrow_shape();
    assert_eq!(
        common::Fold::new(
            &shape,
            vec![0; shape.columns()],
            vec![F128::ONE; shape.columns()],
            Vec::new(),
            vec![F128::ONE; shape.log_columns()],
        ),
        Err(FoldError::RowCountMismatch)
    );
}
