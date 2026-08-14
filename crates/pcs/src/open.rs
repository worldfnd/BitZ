//! Standard multilinear openings over the FLoCK commitment.
//!
//! Prover steps:
//! 1. Validate the prover data and require `query.point.len() == params.m`.
//! 2. Bind the commitment root, trusted parameters, point, and target to the transcript.
//! 3. Split the point into seven low coordinates and `m - 7` high coordinates.
//! 4. Build the high-coordinate equality table with FLoCK's `build_eq`.
//! 5. Compute the 128 partial evaluations with `fold_1b_rows_naive`.
//! 6. Check the target against the low-coordinate equality table.
//! 7. Absorb the ring-switch domain label and all partial evaluations.
//! 8. Sample seven ring-switch challenges and build their equality table.
//! 9. Transpose the partial evaluations and compute the packed target `beta0`.
//! 10. Build the packed Ligerito basis with `fold_b128_elems`.
//! 11. Call `recursive_prover_with_basis` with the retained codeword and Merkle tree.
//! 12. Write a bounded opening proof to the transcript.

use transcript::ProverState;

use crate::{CommitError, OpeningQuery, Pcs, ProverData};

#[allow(dead_code)]
pub(crate) fn prove_lin(
    pcs: &Pcs,
    data: ProverData,
    query: &OpeningQuery,
    transcript: &mut ProverState,
) -> Result<(), CommitError> {
    let expected_m = pcs.params().m;
    if query.point.len() != expected_m {
        return Err(CommitError::PointLengthMismatch);
    }
    if data.bit_len() != pcs.bit_len() || !params_match(pcs, &data) {
        return Err(CommitError::InvalidConfiguration);
    }

    let ligerito_config = pcs
        .params()
        .ligerito_prover_config()
        .map_err(|_| CommitError::InvalidConfiguration)?;

    let commitment = data.commitment();
    let (packed_witness, flock_data) = data.into_opening_parts();

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
