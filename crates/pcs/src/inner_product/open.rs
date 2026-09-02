//! Prover flow for explicit inner-product openings.

use field::F128;
use flock_core::pcs::BatchOpeningProofLigerito;
use flock_core::pcs::ligerito::recursive_prover_with_basis;
use transcript::ProverState;

use super::ring_switch::{InnerProductRingSwitch, ReducedClaim};
use super::validate_profile;
use crate::bridge::into_flock_f128s;
use crate::challenger::ProverChallenger;
use crate::utils::{
    bind_inner_product_statement, observe_opening_target, sample_inner_product_batching_challenges,
    write_inner_product_coordinates, write_opening_proof,
};
use crate::validation::validate_prover_data;
use crate::{CommitError, Pcs, ProverData, StatementBinding};

pub(crate) fn open(
    pcs: &Pcs,
    data: &ProverData,
    packed_witness: Vec<F128>,
    weights: &[F128],
    claimed_target: F128,
    statement_binding: StatementBinding,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    validate_profile(pcs)?;
    let ring_switch = InnerProductRingSwitch::new(weights, pcs.packed_len())?;
    validate_prover_data(pcs, data)?;
    let ligerito_config = pcs.prover_config();

    if statement_binding == StatementBinding::Bind {
        bind_inner_product_statement(
            pcs,
            &data.commitment().root,
            weights,
            claimed_target,
            transcript,
        );
    }

    let packed_witness = into_flock_f128s(packed_witness);
    let coordinate_claims = ring_switch.coordinate_claims(&packed_witness)?;
    if coordinate_claims.reconstructed_target() != claimed_target {
        return Err(CommitError::InvalidClaim);
    }

    write_inner_product_coordinates(transcript, coordinate_claims.as_array());
    let batching_challenges = sample_inner_product_batching_challenges(transcript);
    let ReducedClaim {
        packed_basis,
        packed_target,
    } = ring_switch.reduce(&coordinate_claims, &batching_challenges);

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
            ring_switches: Vec::new(),
            ligerito,
        },
        transcript,
    )
}
