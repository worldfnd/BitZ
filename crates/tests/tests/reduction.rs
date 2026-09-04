//! Step 4 paired across both roles.

use common::{OpeningQuery, ReductionInput};
use reduction::GrandProduct;
use tests::{Instance, evaluate, narrow_shape, wide_shape};
use transcript::{build_prover, build_verifier};

const SESSION: &[u8] = b"f2z-tests";
const INSTANCE: &[u8] = b"reduction-round-trip";

/// Runs the fold and the reduction on both sides and returns what each
/// produced.
fn round_trip(instance: &Instance) -> (OpeningQuery, OpeningQuery) {
    let table = instance.table();

    let mut prover_transcript = build_prover(SESSION, INSTANCE);
    prover_transcript.public_message(&instance.com.0);
    prover_transcript.public_message(&instance.params);
    let fold = instance
        .prover
        .send_fold(&instance.claim, &table, &mut prover_transcript)
        .unwrap();
    let proved = prover::Reduction::reduce(
        &GrandProduct,
        &ReductionInput {
            params: &instance.params,
            claim: &instance.claim,
            commitment: instance.com,
            fold: &fold,
        },
        &table,
        &mut prover_transcript,
    )
    .unwrap();
    let proof = prover_transcript.finish();

    let mut verifier_transcript = build_verifier(SESSION, INSTANCE, &proof);
    verifier_transcript.public_message(&instance.com.0);
    verifier_transcript.public_message(&instance.params);
    let replayed_fold = instance
        .verifier
        .receive_fold(&instance.claim, &mut verifier_transcript)
        .unwrap();
    let replayed = verifier::Reduction::reduce(
        &GrandProduct,
        &ReductionInput {
            params: &instance.params,
            claim: &instance.claim,
            commitment: instance.com,
            fold: &replayed_fold,
        },
        &mut verifier_transcript,
    )
    .unwrap();
    verifier_transcript.check_eof().unwrap();

    (proved, replayed)
}

/// The reduction is only worth anything if its query is a claim about the
/// committed bits. `evaluate` reads them directly, so it agrees only if every
/// step from the fold to the row sumcheck lines up.
fn assert_query_is_the_bit_extension(instance: &Instance, query: &OpeningQuery) {
    let OpeningQuery::Mle { point, target } = query else {
        panic!("the plain path reduces to an evaluation claim");
    };
    assert_eq!(point.len(), instance.params.shape().log_bits());
    assert_eq!(
        *target,
        evaluate(&instance.table(), point),
        "the reduction's target is not the committed bits at its own point"
    );
}

#[test]
fn both_roles_reduce_to_the_same_claim_on_the_committed_bits() {
    for (shape, seed) in [(wide_shape(), 71), (narrow_shape(), 72)] {
        let instance = Instance::honest(shape, seed);
        let (proved, replayed) = round_trip(&instance);

        assert_eq!(proved, replayed, "the roles disagree on the opening query");
        assert_query_is_the_bit_extension(&instance, &proved);
    }
}

/// The whole protocol: step 1 through step 6, with the grand product standing
/// where the stub used to, and the real opening discharging its claim.
#[test]
fn an_honest_proof_verifies_end_to_end() {
    for (shape, seed) in [(wide_shape(), 81), (narrow_shape(), 82)] {
        let instance = Instance::honest(shape, seed);

        let mut transcript = tests::prover_transcript();
        instance
            .prover
            .prove(
                &instance.claim,
                &instance.pcs,
                &instance.data,
                instance.packed.clone(),
                &GrandProduct,
                &mut transcript,
            )
            .unwrap_or_else(|error| panic!("t = {}: {error:?}", shape.log_rows()));
        let proof = transcript.finish();

        instance
            .verifier
            .verify(
                &instance.claim,
                &instance.pcs,
                instance.com,
                &GrandProduct,
                tests::verifier_transcript(&proof),
            )
            .unwrap_or_else(|error| panic!("t = {}: {error:?}", shape.log_rows()));
    }
}

/// A proof carries its own instance. Replaying it against a different witness
/// changes the row images the forest's leaves are built from, so the roots the
/// verifier derives no longer match what was proved.
#[test]
fn a_proof_does_not_verify_against_another_instance() {
    let proved = Instance::honest(wide_shape(), 91);
    let other = Instance::honest(wide_shape(), 92);

    let mut transcript = tests::prover_transcript();
    proved
        .prover
        .prove(
            &proved.claim,
            &proved.pcs,
            &proved.data,
            proved.packed.clone(),
            &GrandProduct,
            &mut transcript,
        )
        .unwrap();
    let proof = transcript.finish();

    assert!(
        other
            .verifier
            .verify(
                &other.claim,
                &other.pcs,
                other.com,
                &GrandProduct,
                tests::verifier_transcript(&proof),
            )
            .is_err()
    );
}
