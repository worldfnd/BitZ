//! Arbitrary `F128` inner products over the original committed bits.
//!
//! The ring-switch algebra lives in [`ring_switch`]. This module only defines
//! the transcript order and the final generic Ligerito opening.

mod ring_switch;

use field::F128;
use flock_core::pcs::ligerito::{recursive_prover_with_basis, recursive_verifier_with_basis};
use flock_core::pcs::{BatchOpeningProofLigerito, LOG_PACKING};
use transcript::{ProverState, VerifierState};

use self::ring_switch::{CoordinateClaims, InnerProductRingSwitch, ReducedClaim};
use crate::bridge::into_flock_f128s;
use crate::challenger::{ProverChallenger, VerifierChallenger};
use crate::utils::{
    bind_inner_product_statement, observe_opening_target, read_inner_product_coordinates,
    read_opening_proof, sample_inner_product_batching_challenges, write_inner_product_coordinates,
    write_opening_proof,
};
use crate::verify::validate_ligerito_proof_shape;
use crate::{CommitError, Commitment, LigeritoProfile, Pcs, ProverData, StatementBinding};

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
    if !params_match(pcs, data) {
        return Err(CommitError::invalid_configuration(format!(
            "prover data parameters do not match the active PCS: expected {:?}, got {:?}",
            pcs.params(),
            data.commitment().params,
        )));
    }
    let log_n = pcs.params().m.checked_sub(LOG_PACKING).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "PCS variable count {} is smaller than the packing width {LOG_PACKING}",
            pcs.params().m,
        ))
    })?;
    let ligerito_config = pcs
        .params()
        .ligerito_prover_config()
        .map_err(CommitError::InvalidConfiguration)?;

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

    observe_opening_target(transcript, log_n, packed_target)?;
    let mut challenger = ProverChallenger::new_ligerito(transcript, packed_target);
    let flock_data = data.flock_data();
    let ligerito = recursive_prover_with_basis(
        &ligerito_config,
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

pub(crate) fn verify(
    pcs: &Pcs,
    commitment: &Commitment,
    weights: &[F128],
    claimed_target: F128,
    statement_binding: StatementBinding,
    transcript: &mut VerifierState<'_>,
) -> Result<(), CommitError> {
    validate_profile(pcs)?;
    let ring_switch = InnerProductRingSwitch::new(weights, pcs.packed_len())?;
    let log_n = pcs.params().m.checked_sub(LOG_PACKING).ok_or_else(|| {
        CommitError::invalid_configuration(format!(
            "PCS variable count {} is smaller than the packing width {LOG_PACKING}",
            pcs.params().m,
        ))
    })?;
    let ligerito_config = pcs
        .params()
        .ligerito_verifier_config()
        .map_err(CommitError::InvalidConfiguration)?;

    if statement_binding == StatementBinding::Bind {
        bind_inner_product_statement(pcs, commitment.root(), weights, claimed_target, transcript);
    }

    let proof = read_opening_proof(transcript)?;
    if !proof.ring_switches.is_empty() {
        return Err(CommitError::VerificationFailed);
    }
    validate_ligerito_proof_shape(
        &proof.ligerito,
        &ligerito_config,
        pcs.final_log_n(),
        commitment.root(),
    )?;

    let coordinate_claims =
        CoordinateClaims::from_array(read_inner_product_coordinates(transcript)?);
    if coordinate_claims.reconstructed_target() != claimed_target {
        return Err(CommitError::VerificationFailed);
    }
    let batching_challenges = sample_inner_product_batching_challenges(transcript);
    let ReducedClaim {
        packed_basis,
        packed_target,
    } = ring_switch.reduce(&coordinate_claims, &batching_challenges);

    observe_opening_target(transcript, log_n, packed_target)?;
    let mut challenger = VerifierChallenger::new_ligerito(transcript, packed_target);
    let valid = recursive_verifier_with_basis(
        &ligerito_config,
        &proof.ligerito,
        &packed_basis,
        packed_target,
        commitment.root(),
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

fn validate_profile(pcs: &Pcs) -> Result<(), CommitError> {
    if pcs.params().profile != LigeritoProfile::Secure {
        return Err(CommitError::UnsupportedInnerProductProfile);
    }
    Ok(())
}

fn params_match(pcs: &Pcs, data: &ProverData) -> bool {
    let expected = pcs.params();
    let actual = &data.commitment().params;

    expected.m == actual.m
        && expected.log_inv_rate == actual.log_inv_rate
        && expected.log_batch_size == actual.log_batch_size
        && expected.profile == actual.profile
        && expected.merkle_hash == actual.merkle_hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_inner_products_require_the_secure_profile() {
        let shape = common::Shape::new(7, 15).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, crate::HashKind::Blake3).unwrap();
        assert_eq!(
            validate_profile(&pcs),
            Err(CommitError::UnsupportedInnerProductProfile),
        );
    }
}
