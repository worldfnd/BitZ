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
    _pcs: &Pcs,
    _data: ProverData,
    _query: &OpeningQuery,
    _transcript: &mut ProverState,
) -> Result<(), CommitError> {
    todo!("implement the standard FLoCK opening prover")
}
