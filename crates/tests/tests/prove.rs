//! The top-level prove and verify, through the real opening.

use common::{Root, TableError};
use field::{F128, Fq};
use pcs::{HashKind, LigeritoProfile, Pcs, VerifyError as PcsVerifyError};
use prover::ProveError;
use tests::{
    Instance, large_shape, narrow_shape, prover_transcript, verifier_transcript, wide_shape,
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
                verifier_transcript(&proof),
            )
            .unwrap_or_else(|error| panic!("t = {}: {error:?}", shape.log_rows()));
    }
}

/// The direct claim under a prime sampled from a transcript: the modulus is
/// installed, the parameters, weights and target are built in
/// `Fq<RUNTIME>`, and prove and verify agree — the parameter frame carries
/// the sampled prime, so a proof under one prime is a proof under it alone.
#[test]
fn an_honest_proof_verifies_under_a_sampled_prime() {
    use common::{BitZParams, LinearClaim};
    use crypto_primitives::LiftElement;
    use field::{FqRuntime, RUNTIME, gf128::smallest_generator};
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    let mut sampler = prover_transcript();
    let prime = sampler.squeeze_prime(100);
    assert_ne!(prime, field::Q100);
    field::set_modulus(prime).unwrap();

    let shape = narrow_shape();
    let params = BitZParams::<RUNTIME>::new(shape, smallest_generator()).unwrap();
    assert_eq!(params.fold_bound(), 128 * (prime - 1));
    let mut rng = ChaCha8Rng::seed_from_u64(35);
    let packed: Vec<F128> = (0..(1 << shape.log_bits()) / 128)
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let mut residue = || {
        FqRuntime::from(u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64()))
    };
    let row_weights: Vec<FqRuntime> = (0..shape.rows()).map(|_| residue()).collect();
    let column_weights: Vec<FqRuntime> = (0..shape.columns()).map(|_| residue()).collect();
    let table = params.table(&packed).unwrap();
    let exponents: Vec<u128> = row_weights.iter().map(|weight| weight.lift()).collect();
    assert!(exponents.iter().all(|&exponent| exponent < prime));
    let target: FqRuntime = (0..shape.columns())
        .map(|column| {
            let fold: u128 = (0..shape.rows())
                .filter(|&row| table.bit(column, row))
                .map(|row| exponents[row])
                .sum();
            column_weights[column] * FqRuntime::from(fold)
        })
        .sum();
    let claim = LinearClaim::new(&params, row_weights, column_weights, target).unwrap();
    let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let (com, data) = pcs.commit(&packed).unwrap();

    let mut transcript = prover_transcript();
    prover::BitZProver::new(params, tests::WINDOW)
        .prove(&claim, &pcs, &data, packed.clone(), &mut transcript)
        .expect("honest instance under the sampled prime");
    let proof = transcript.finish();
    verifier::BitZVerifier::new(params, tests::WINDOW)
        .verify(&claim, &pcs, com, verifier_transcript(&proof))
        .expect("the verifier under the same prime accepts");

    field::set_modulus(field::Q100).unwrap();
}

#[test]
fn a_proof_replayed_under_a_different_commitment_is_refused() {
    let instance = Instance::honest(narrow_shape(), 32);
    let proof = prove(&instance);

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
    let instance = Instance::honest(narrow_shape(), 37);
    let mut proof = prove(&instance);
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
            verifier_transcript(&proof)
        ),
        Err(VerifyError::Opening(PcsVerifyError::VerificationFailed))
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
            &mut transcript,
        ),
        Err(ProveError::Witness(TableError::BitCountMismatch))
    );

    // The shape is checked before the first absorb, so a rejected witness
    // leaves no half-written proof behind.
    let proof = transcript.finish();
    assert!(proof.narg_string.is_empty() && proof.hints.is_empty());
}
