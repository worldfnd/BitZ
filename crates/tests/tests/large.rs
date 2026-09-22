//! Tests for the large fixture, that does NOT do the opening.
//!
//! At `m = 28` the round trip is out of reach for a test. The reduction
//! materializes one field element per committed bit and every product-tree
//! layer above them, some 10 GiB, and an unoptimised build needs over an hour
//! for it.
//!
//! This test is designed to test what's possible in under a minute.

use common::LinearClaim;
use field::Fq;
use num_traits::ConstOne;
use prover::BitZProver;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use tests::{HonestClaim, WINDOW, large_shape, prover_transcript, verifier_transcript};
use verifier::{BitZVerifier, ReceiveError};

#[test]
fn the_fold_round_trips_on_the_large_shape() {
    let shape = large_shape();
    let honest = HonestClaim::new(shape, &mut ChaCha8Rng::seed_from_u64(31));
    let prover = BitZProver::new(honest.params, WINDOW);
    let verifier = BitZVerifier::new(honest.params, WINDOW);

    let mut transcript = prover_transcript();
    let sent = prover
        .send_fold(&honest.claim, &honest.table(), &mut transcript)
        .unwrap();
    let proof = transcript.finish();
    assert_eq!(proof.narg_string.len(), 16 * shape.columns());
    assert!(proof.hints.is_empty());

    let mut transcript = verifier_transcript(&proof);
    let received = verifier
        .receive_fold(&honest.claim, &mut transcript)
        .expect("honest proof");
    assert_eq!(sent, received);
    assert_eq!(received.row_images.len(), shape.rows());
    assert_eq!(received.zeta.len(), shape.log_columns());
    transcript.check_eof().expect("both streams exhausted");

    // `k_1 (q - 1)` is 117 bits wide here; one past it is still refused.
    let mut transcript = prover_transcript();
    for _ in 0..shape.columns() {
        transcript.prover_message(&(verifier.fold_bound() + 1).to_le_bytes());
    }
    let over = transcript.finish();
    assert_eq!(
        verifier.receive_fold(&honest.claim, &mut verifier_transcript(&over)),
        Err(ReceiveError::FoldOutOfRange)
    );

    // The honest folds against a claim off by one.
    let retargeted = LinearClaim::new(
        &honest.params,
        honest.claim.row_weights().to_vec(),
        honest.claim.column_weights().to_vec(),
        honest.claim.target() + Fq::ONE,
    )
    .unwrap();
    assert_eq!(
        verifier.receive_fold(&retargeted, &mut verifier_transcript(&proof)),
        Err(ReceiveError::TargetMismatch)
    );
}
