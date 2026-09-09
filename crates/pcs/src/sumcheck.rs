//! Temporary reference implementation of the inner-product reduction.
//!
//! Replace the reduction internals here, while preserving `prove`, `verify`, and the `MleClaim` contract.
//! The current quadratic sumcheck uses a dense witness table with one `F128` element per bit.
//! Its tests own the round encoding and byte offsets, so those details can change together.
//!
//! The input contains arbitrary row and column weights over `F128`.
//! Bit `column * row_weights.len() + row` has weight `row_weights[row] * column_weights[column]`.
//! Each packed witness element stores bits 0 through 63 in `lo`, then bits 64 through 127 in `hi`.
//! Packed element `i`, local bit `v`, supplies logical bit `128 * i + v`.
//! The target is the claimed weighted sum of the original committed bits.
//! An optimized reduction over rows alone needs an explicit column evaluation point, which this input does not provide.
//!
//! `LinearClaim` ensures nonempty factors whose lengths are powers of two.
//! Opening code checks the total weight count, packed witness length, and retained prover parameters.
//! Opening code or the outer caller binds the original statement before this stage.
//! Opening code always adds the sumcheck domain before calling `prove` or `verify`.
//!
//! This module owns the round messages, challenges, and terminal product check in the supplied transcript.
//! It returns the witness MLE claim and leaves the remaining transcript for the opening protocol.
//! Opening code binds that claim and performs the final PCS opening, which authenticates the witness evaluation.

use common::LinearClaim;
use field::F128;
use transcript::{ProverState, VerifierState};

use crate::{ProveError, VerifyError};

/// A pending evaluation claim over the original committed bit polynomial.
pub(super) struct MleClaim {
    /// The evaluation point, with row coordinates before column coordinates, in low-bit-first order.
    /// Its length is `log2(row_weights.len()) + log2(column_weights.len())`.
    pub(super) point: Vec<F128>,
    /// The witness MLE evaluation, rather than the terminal product evaluation.
    pub(super) target: F128,
}

/// Proves the reduction using a temporary dense table of witness evaluations.
pub(super) fn prove(
    claim: &LinearClaim<F128>,
    packed_witness: &[F128],
    transcript: &mut ProverState,
) -> Result<MleClaim, ProveError> {
    let mut rows = claim.row_weights().to_vec();
    let mut columns = claim.column_weights().to_vec();
    let bit_len = rows.len() * columns.len();
    if packed_witness.len().checked_mul(128) != Some(bit_len) {
        return Err(ProveError::PackedWitnessLengthMismatch);
    }
    let mut witness = Vec::with_capacity(bit_len);
    for packed in packed_witness {
        let bits = u128::from(packed.lo) | (u128::from(packed.hi) << 64);
        for bit in 0..128 {
            witness.push(F128::from(((bits >> bit) & 1) as u64));
        }
    }

    let mut target = claim.target();
    let mut point = Vec::with_capacity(bit_len.ilog2() as usize);
    while witness.len() > 1 {
        let coefficients = round_polynomial(&witness, &rows, &columns);
        // In characteristic two, g(0) + g(1) = a1 + a2.
        if coefficients[1] + coefficients[2] != target {
            return Err(ProveError::InvalidClaim);
        }
        transcript.prover_message(&coefficients);
        let challenge = transcript.verifier_message::<F128>();
        target = evaluate_round(coefficients, challenge);
        point.push(challenge);
        fold(&mut witness, challenge);
        if rows.len() > 1 {
            fold(&mut rows, challenge);
        } else {
            fold(&mut columns, challenge);
        }
    }

    let evaluation = witness[0];
    if target != evaluation * rows[0] * columns[0] {
        return Err(ProveError::InvalidClaim);
    }
    transcript.prover_message(&evaluation);
    Ok(MleClaim {
        point,
        target: evaluation,
    })
}

/// Verifies sumcheck and returns a witness claim that still requires the final PCS opening.
pub(super) fn verify(
    claim: &LinearClaim<F128>,
    transcript: &mut VerifierState<'_>,
) -> Result<MleClaim, VerifyError> {
    let mut rows = claim.row_weights().to_vec();
    let mut columns = claim.column_weights().to_vec();
    let rounds = rows.len().ilog2() as usize + columns.len().ilog2() as usize;
    let mut point = Vec::with_capacity(rounds);
    let mut target = claim.target();
    for _ in 0..rounds {
        let coefficients = transcript
            .prover_message::<[F128; 3]>()
            .map_err(|_| VerifyError::MalformedProof)?;
        if coefficients[1] + coefficients[2] != target {
            return Err(VerifyError::VerificationFailed);
        }
        let challenge = transcript.verifier_message::<F128>();
        target = evaluate_round(coefficients, challenge);
        point.push(challenge);
        if rows.len() > 1 {
            fold(&mut rows, challenge);
        } else {
            fold(&mut columns, challenge);
        }
    }

    let evaluation = transcript
        .prover_message::<F128>()
        .map_err(|_| VerifyError::MalformedProof)?;
    // Multiplication also handles zero weights; the MLE opening authenticates the witness value.
    if target != evaluation * rows[0] * columns[0] {
        return Err(VerifyError::VerificationFailed);
    }
    Ok(MleClaim {
        point,
        target: evaluation,
    })
}

/// Returns the coefficients of the next degree-two round polynomial.
fn round_polynomial(witness: &[F128], rows: &[F128], columns: &[F128]) -> [F128; 3] {
    let mut at_zero = F128::default();
    let mut at_one = F128::default();
    let mut quadratic = F128::default();
    for (pair_index, pair) in witness.chunks_exact(2).enumerate() {
        let index = 2 * pair_index;
        let weight_zero = rows[index % rows.len()] * columns[index / rows.len()];
        let weight_one = rows[(index + 1) % rows.len()] * columns[(index + 1) / rows.len()];
        at_zero += pair[0] * weight_zero;
        at_one += pair[1] * weight_one;
        quadratic += (pair[0] + pair[1]) * (weight_zero + weight_one);
    }
    [at_zero, at_zero + at_one + quadratic, quadratic]
}

fn evaluate_round([constant, linear, quadratic]: [F128; 3], challenge: F128) -> F128 {
    constant + challenge * (linear + challenge * quadratic)
}

/// Fixes the lowest remaining coordinate without changing the order of the remaining coordinates.
fn fold(values: &mut Vec<F128>, challenge: F128) {
    let len = values.len() / 2;
    for index in 0..len {
        let zero = values[2 * index];
        let one = values[2 * index + 1];
        values[index] = zero + challenge * (zero + one);
    }
    values.truncate(len);
}

#[cfg(test)]
mod tests;
