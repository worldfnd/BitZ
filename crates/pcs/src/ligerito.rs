//! Shared Ligerito tail for reduced linear claims.
//!
//! Query modules derive one packed basis and target through their ring switch.
//! This module validates the prover input and runs the common opening protocol.

use field::F128;
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::ligerito::{
    LigeritoProof, recursive_prover_with_basis, recursive_verifier_with_basis,
    recursive_verifier_with_basis_succinct,
};
use flock_core::pcs::{BatchOpeningProofLigerito, RingSwitchProof};
use transcript::{ProverState, VerifierState};

use crate::bridge::into_flock_f128s;
use crate::challenger::{ProverChallenger, VerifierChallenger};
use crate::ring_switch::CLAIM_COUNT;
use crate::utils::{observe_opening_target, read_opening_proof, write_opening_proof};
use crate::validation::{validate_ligerito_proof_shape, validate_prover_data};
use crate::{CommitError, Commitment, Pcs, ProverData};

/// One packed-field claim produced by a query-specific ring switch.
pub(crate) struct ReducedClaim {
    pub(crate) packed_basis: Vec<FlockF128>,
    pub(crate) packed_target: FlockF128,
}

/// Expected query-specific payload in the shared opening proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RingSwitchPayloadShape {
    /// An explicit inner-product opening has no Flock ring-switch payload.
    None,
    /// An MLE opening has one fixed-width Flock ring-switch payload.
    Single,
}

/// Validated prover input for the shared Ligerito tail.
pub(crate) struct ReducedProver<'a> {
    pcs: &'a Pcs,
    data: &'a ProverData,
    packed_witness: Vec<FlockF128>,
}

impl<'a> ReducedProver<'a> {
    /// Validates and converts the packed witness before transcript mutation.
    pub(crate) fn new(
        pcs: &'a Pcs,
        data: &'a ProverData,
        packed_witness: Vec<F128>,
    ) -> Result<Self, CommitError> {
        if packed_witness.len() != pcs.packed_len() {
            return Err(CommitError::InvalidBitLength);
        }
        validate_prover_data(pcs, data)?;
        Ok(Self {
            pcs,
            data,
            packed_witness: into_flock_f128s(packed_witness),
        })
    }

    /// Returns the packed witness for the query-specific ring switch.
    pub(crate) fn witness(&self) -> &[FlockF128] {
        &self.packed_witness
    }

    /// Proves one reduced claim and writes the completed opening proof.
    pub(crate) fn prove(
        self,
        claim: ReducedClaim,
        ring_switches: Vec<RingSwitchProof>,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError> {
        let ReducedClaim {
            packed_basis,
            packed_target,
        } = claim;
        observe_opening_target(transcript, self.pcs.opening_log_n(), packed_target);
        let mut challenger = ProverChallenger::new_ligerito(transcript, packed_target);
        let flock_data = self.data.flock_data();
        let ligerito = recursive_prover_with_basis(
            self.pcs.prover_config(),
            self.packed_witness,
            packed_basis,
            packed_target,
            &flock_data.codeword,
            &flock_data.merkle_tree,
            &mut challenger,
        );
        if challenger.failed() {
            return Err(CommitError::invalid_configuration(
                "missing Ligerito opening-target prefix",
            ));
        }

        write_opening_proof(
            &BatchOpeningProofLigerito {
                ring_switches,
                ligerito,
            },
            transcript,
        )
    }
}

/// Reads one proof and validates its query-specific and Ligerito shapes.
pub(crate) fn read_proof(
    pcs: &Pcs,
    commitment: &Commitment,
    ring_switch_shape: RingSwitchPayloadShape,
    transcript: &mut VerifierState<'_>,
) -> Result<BatchOpeningProofLigerito, CommitError> {
    let proof = read_opening_proof(transcript)?;
    validate_proof_shape(pcs, commitment, &proof, ring_switch_shape)?;
    Ok(proof)
}

pub(crate) fn validate_proof_shape(
    pcs: &Pcs,
    commitment: &Commitment,
    proof: &BatchOpeningProofLigerito,
    ring_switch_shape: RingSwitchPayloadShape,
) -> Result<(), CommitError> {
    let ring_switch_shape_matches = match ring_switch_shape {
        RingSwitchPayloadShape::None => proof.ring_switches.is_empty(),
        RingSwitchPayloadShape::Single => {
            proof.ring_switches.len() == 1 && proof.ring_switches[0].s_hat_v.len() == CLAIM_COUNT
        }
    };
    if !ring_switch_shape_matches {
        return Err(CommitError::VerificationFailed);
    }

    validate_ligerito_proof_shape(
        &proof.ligerito,
        pcs.verifier_config(),
        pcs.final_log_n(),
        commitment.root(),
    )
}

/// Verifies a reduced claim with a materialized packed basis.
pub(crate) fn verify_dense(
    pcs: &Pcs,
    commitment: &Commitment,
    proof: &LigeritoProof,
    claim: ReducedClaim,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    let ReducedClaim {
        packed_basis,
        packed_target,
    } = claim;
    finish_verification(
        pcs.opening_log_n(),
        packed_target,
        transcript,
        |challenger| {
            recursive_verifier_with_basis(
                pcs.verifier_config(),
                proof,
                &packed_basis,
                packed_target,
                commitment.root(),
                challenger,
            )
        },
    )
}

/// Verifies a reduced claim with a succinct packed-basis evaluator.
pub(crate) fn verify_succinct<F>(
    pcs: &Pcs,
    commitment: &Commitment,
    proof: &LigeritoProof,
    log_n: usize,
    packed_target: FlockF128,
    evaluate_basis: F,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError>
where
    F: Fn(&[FlockF128], usize) -> Vec<FlockF128>,
{
    finish_verification(
        pcs.opening_log_n(),
        packed_target,
        transcript,
        |challenger| {
            recursive_verifier_with_basis_succinct(
                pcs.verifier_config(),
                proof,
                log_n,
                packed_target,
                commitment.root(),
                evaluate_basis,
                challenger,
            )
        },
    )
}

fn finish_verification<'proof>(
    opening_log_n: u32,
    packed_target: FlockF128,
    transcript: &mut VerifierState<'proof>,
    verify: impl FnOnce(&mut VerifierChallenger<'_, 'proof>) -> bool,
) -> Result<(), CommitError> {
    observe_opening_target(transcript, opening_log_n, packed_target);
    let mut challenger = VerifierChallenger::new_ligerito(transcript, packed_target);
    let valid = verify(&mut challenger);
    if challenger.failed() {
        return Err(CommitError::MalformedProof);
    }
    if !valid {
        return Err(CommitError::VerificationFailed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommitScheme, HashKind, LigeritoProfile, OpeningQuery, StatementBinding};
    use common::Shape;
    use transcript::{build_prover, build_verifier};

    #[test]
    fn proof_shape_validation_rejects_prover_supplied_dimension_mismatches() {
        const SESSION: &[u8] = b"pcs-proof-shape-test";
        const INSTANCE: &[u8] = b"zero-polynomial";
        let shape = Shape::new(7, 15).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let packed_witness = vec![F128::default(); pcs.packed_len()];
        let (commitment, data) = pcs.commit(&packed_witness).unwrap();
        let query = OpeningQuery::Mle {
            point: vec![F128::from(2u64); 22],
            target: F128::default(),
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
        let shape = RingSwitchPayloadShape::Single;
        assert_eq!(
            validate_proof_shape(&pcs, &commitment, &valid, shape),
            Ok(())
        );
        assert_eq!(
            validate_proof_shape(&pcs, &commitment, &valid, RingSwitchPayloadShape::None),
            Err(CommitError::VerificationFailed),
        );

        type ProofMutation = fn(&mut BatchOpeningProofLigerito);
        let mutations: [(&str, ProofMutation); 4] = [
            ("ring-switch count", |proof| proof.ring_switches.clear()),
            ("ring-switch claim count", |proof| {
                proof.ring_switches[0].s_hat_v.pop();
            }),
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
                validate_proof_shape(&pcs, &commitment, &proof, shape),
                Err(CommitError::VerificationFailed),
                "{case}",
            );
        }
    }
}
