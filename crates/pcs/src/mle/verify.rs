//! Verifier flow for multilinear openings.

use field::F128;
use flock_core::pcs::BatchOpeningProofLigerito;
use flock_core::pcs::LOG_PACKING;
use flock_core::pcs::ligerito::{VerifierConfig, recursive_verifier_with_basis_succinct};
use transcript::VerifierState;

use super::ring_switch::{MleRingSwitch, RingSwitchClaims};
use crate::challenger::VerifierChallenger;
use crate::utils::{
    bind_ring_switch_message, bind_statement, observe_opening_target, read_opening_proof,
    sample_ring_switch_point,
};
use crate::validation::validate_ligerito_proof_shape;
use crate::{CommitError, Commitment, Pcs, StatementBinding};

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Commitment,
    point: &[F128],
    target: F128,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    let ring_switch = MleRingSwitch::new(point, pcs.params().m)?;
    let log_n = ring_switch.suffix_dimension();
    let ligerito_config = pcs.verifier_config();

    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, commitment.root(), point, target, transcript);
    }

    let proof = read_opening_proof(transcript)?;
    validate_mle_proof_shape(
        &proof,
        ligerito_config,
        pcs.final_log_n(),
        commitment.root(),
    )?;

    let proof_claims = &proof.ring_switches[0].s_hat_v;
    let claims = RingSwitchClaims::from_proof(proof_claims)?;
    bind_ring_switch_message(transcript, claims.as_slice())?;
    if !ring_switch.target_matches(&claims, target) {
        return Err(CommitError::VerificationFailed);
    }

    let challenge_point = sample_ring_switch_point(transcript);
    let reduced_claim = ring_switch.reduce_verifier(&claims, &challenge_point);
    let packed_target = reduced_claim.packed_target;

    observe_opening_target(transcript, pcs.opening_log_n(), packed_target);
    let mut challenger = VerifierChallenger::new_ligerito(transcript, packed_target);
    let evaluate_basis = |ris: &[_], yr_log_n| reduced_claim.evaluate_basis(ris, yr_log_n);
    let valid = recursive_verifier_with_basis_succinct(
        ligerito_config,
        &proof.ligerito,
        log_n,
        packed_target,
        commitment.root(),
        evaluate_basis,
        &mut challenger,
    );
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }
    if !valid {
        return Err(CommitError::VerificationFailed);
    }
    Ok(())
}

fn validate_mle_proof_shape(
    proof: &BatchOpeningProofLigerito,
    config: &VerifierConfig,
    final_log_n: usize,
    expected_root: &[u8; 32],
) -> Result<(), CommitError> {
    if proof.ring_switches.len() != 1
        || proof.ring_switches[0].s_hat_v.len() != 1usize << LOG_PACKING
    {
        return Err(CommitError::VerificationFailed);
    }

    validate_ligerito_proof_shape(&proof.ligerito, config, final_log_n, expected_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::validate_config;
    use crate::{CommitScheme, HashKind, LigeritoProfile, OpeningQuery};
    use common::Shape;
    use field::F128 as LocalF128;
    use transcript::{build_prover, build_verifier};

    #[test]
    fn proof_shape_validation_rejects_prover_supplied_dimension_mismatches() {
        const SESSION: &[u8] = b"pcs-proof-shape-test";
        const INSTANCE: &[u8] = b"zero-polynomial";
        let shape = Shape::new(7, 15).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let packed_witness = vec![LocalF128::default(); pcs.packed_len()];
        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let query = OpeningQuery::Mle {
            point: vec![LocalF128::from(2u64); 22],
            target: LocalF128::default(),
        };
        let mut prover = build_prover(SESSION, INSTANCE);
        pcs.prove_lin(
            &data,
            packed_witness,
            &query,
            StatementBinding::Bind,
            &mut prover,
        )
        .unwrap();
        let transcript_proof = prover.finish();
        let mut verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        let valid = read_opening_proof(&mut verifier).unwrap();
        let config = pcs.params().ligerito_verifier_config().unwrap();
        let final_log_n =
            validate_config(&config, 22 - LOG_PACKING, pcs.params().log_batch_size).unwrap();
        let root = *commitment.root();
        assert_eq!(
            validate_mle_proof_shape(&valid, &config, final_log_n, &root),
            Ok(())
        );

        type ProofMutation = fn(&mut BatchOpeningProofLigerito);
        let mutations: [(&str, ProofMutation); 3] = [
            ("ring-switch count", |proof| proof.ring_switches.clear()),
            ("recursive-root count", |proof| {
                proof.ligerito.recursive_roots.pop();
            }),
            ("opened-row width", |proof| {
                proof.ligerito.initial_proof.opened_rows[0].pop();
            }),
        ];

        for (case, mutate) in mutations {
            let mut proof = valid.clone();
            mutate(&mut proof);
            assert_eq!(
                validate_mle_proof_shape(&proof, &config, final_log_n, &root),
                Err(CommitError::VerificationFailed),
                "{case}",
            );
        }
    }
}
