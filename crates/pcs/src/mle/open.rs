//! Prover flow for multilinear openings.

use field::F128;
use flock_core::pcs::ligerito::recursive_prover_with_basis;
use flock_core::pcs::{BatchOpeningProofLigerito, RingSwitchProof};
use transcript::ProverState;

use super::ring_switch::{DenseReducedClaim, MleRingSwitch};
use crate::bridge::into_flock_f128s;
use crate::challenger::ProverChallenger;
use crate::utils::{
    bind_ring_switch_message, bind_statement, observe_opening_target, sample_ring_switch_point,
    write_opening_proof,
};
use crate::validation::validate_prover_data;
use crate::{CommitError, Pcs, ProverData, StatementBinding};

pub(crate) fn open(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    point: &[F128],
    target: F128,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    let ring_switch = MleRingSwitch::new(point, pcs.params().m)?;
    if packed_witness.len() != pcs.packed_len() {
        return Err(CommitError::InvalidBitLength);
    }
    validate_prover_data(pcs, data)?;
    let ligerito_config = pcs.prover_config();

    if statement_binding == StatementBinding::Bind {
        bind_statement(pcs, &data.commitment().root, point, target, transcript);
    }

    let packed_witness = into_flock_f128s(packed_witness);
    let prover_ring_switch = ring_switch.prepare_prover(&packed_witness, target)?;
    bind_ring_switch_message(transcript, prover_ring_switch.claims().as_slice())?;

    let challenge_point = sample_ring_switch_point(transcript);
    let (claims, reduced_claim) = prover_ring_switch.reduce(&challenge_point);
    let DenseReducedClaim {
        packed_basis,
        packed_target,
    } = reduced_claim;

    observe_opening_target(transcript, pcs.opening_log_n(), packed_target);
    let mut challenger = ProverChallenger::new_ligerito(transcript, packed_target);
    let flock_data = data.flock_data();
    let ligerito = recursive_prover_with_basis(
        ligerito_config,
        packed_witness,
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
            ring_switches: vec![RingSwitchProof {
                s_hat_v: claims.into_vec(),
            }],
            ligerito,
        },
        transcript,
    )
}
