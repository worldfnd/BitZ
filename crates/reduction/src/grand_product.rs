//! The `Reduction` both roles run at step 4.

use common::{BitTable, OpeningQuery, ReductionInput};
use field::F128;
use gkr::{ForestError, TreeClaim, prove_product_forest, verify_product_forest};
use num_traits::{ConstOne, ConstZero, Inv};
use poly::eq_table;
use transcript::{ProverState, VerifierState};

/// A reduction that cannot run, or a proof of one the verifier rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionError {
    /// The grand product failed.
    Forest(ForestError),
    /// A record is missing or does not decode.
    MalformedProof,
    /// The row sumcheck does not reduce the batched column claim.
    RowClaimMismatch,
    /// `R(r)` is zero, so the terminal value says nothing about the bits.
    ///
    /// `R` is a public polynomial and `r` comes from the transcript, so this
    /// is a chance event rather than something a prover chooses.
    PublicWeightVanished,
}

impl From<ForestError> for ReductionError {
    fn from(error: ForestError) -> Self {
        Self::Forest(error)
    }
}

/// Step 4, as a grand-product forest followed by one row sumcheck.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrandProduct;

/// `y_i - 1`, the factor a set bit contributes over the unset bit's `1`.
fn leaf_offsets(row_images: &[F128]) -> Vec<F128> {
    row_images.iter().map(|&y| y - F128::ONE).collect()
}

/// `1 + f_ij (y_i - 1)` for every row of one column.
fn column_leaves(table: &BitTable<'_>, offsets: &[F128], column: usize) -> Vec<F128> {
    (0..offsets.len())
        .map(|row| {
            if table.bit(column, row) {
                F128::ONE + offsets[row]
            } else {
                F128::ONE
            }
        })
        .collect()
}

/// `sum_j eq(j, xi) (leaf_j(rho) - 1)`, the columns batched into one claim.
fn batch_columns(claims: &[TreeClaim<F128>], equality: &[F128]) -> F128 {
    claims
        .iter()
        .zip(equality)
        .fold(F128::ZERO, |sum, (claim, &weight)| {
            sum + weight * (claim.evaluation - F128::ONE)
        })
}

/// `R(i) = eq(i, rho) (y_i - 1)`, the public side of the row sumcheck.
fn row_weights(rho: &[F128], offsets: &[F128]) -> Vec<F128> {
    eq_table(rho)
        .into_iter()
        .zip(offsets)
        .map(|(weight, &offset)| weight * offset)
        .collect()
}

/// `m(i) = sum_j eq(j, xi) f_ij`, the committed bits batched across columns.
fn batched_bits(table: &BitTable<'_>, equality: &[F128], rows: usize) -> Vec<F128> {
    let mut batched = vec![F128::ZERO; rows];
    for (column, &weight) in equality.iter().enumerate() {
        for (row, entry) in batched.iter_mut().enumerate() {
            if table.bit(column, row) {
                *entry += weight;
            }
        }
    }
    batched
}

/// `[c0, c1, c2]` of `(a0 + X da)(b0 + X db)`.
#[inline]
fn quadratic_coefficients(left: [F128; 2], right: [F128; 2]) -> [F128; 3] {
    let (left_zero, left_delta) = (left[0], left[1] - left[0]);
    let (right_zero, right_delta) = (right[0], right[1] - right[0]);
    [
        left_zero * right_zero,
        left_zero * right_delta + left_delta * right_zero,
        left_delta * right_delta,
    ]
}

fn fold_in_place(table: &mut Vec<F128>, challenge: F128) {
    let half = table.len() / 2;
    for i in 0..half {
        let (zero, one) = (table[2 * i], table[2 * i + 1]);
        table[i] = zero + challenge * (one - zero);
    }
    table.truncate(half);
}

fn horner(coefficients: &[F128; 3], point: F128) -> F128 {
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(F128::ZERO, |value, coefficient| value * point + coefficient)
}

/// The claim `<v, h>` reduces to, as a point over `row ++ column` and a value.
///
/// The bit table indexes rows in the low coordinates, so the row point the
/// sumcheck produced comes first.
fn opening_point(row_point: &[F128], column_point: &[F128]) -> Vec<F128> {
    row_point.iter().chain(column_point).copied().collect()
}

impl<const Q: u128> prover::Reduction<Q> for GrandProduct {
    type Error = ReductionError;

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
    ) -> Result<OpeningQuery, Self::Error> {
        let shape = input.params.shape();
        let offsets = leaf_offsets(&input.fold.row_images);

        let leaves: Vec<Vec<F128>> = {
            let _guard = prof::scope("reduce/leaves");
            (0..shape.columns())
                .map(|column| column_leaves(table, &offsets, column))
                .collect()
        };
        let claims = {
            let _guard = prof::scope("reduce/forest");
            prove_product_forest(transcript, leaves, &input.fold.images)?
        };

        let equality = eq_table(&input.fold.zeta);
        let mut claim = batch_columns(&claims, &equality);

        let rho = &claims[0].point;
        let mut weights = row_weights(rho, &offsets);
        let mut bits = {
            let _guard = prof::scope("reduce/batch-bits");
            batched_bits(table, &equality, shape.rows())
        };

        let _guard = prof::scope("reduce/row-sumcheck");
        let mut row_point = Vec::with_capacity(shape.log_rows());
        for _ in 0..shape.log_rows() {
            let mut coefficients = [F128::ZERO; 3];
            for pair in 0..weights.len() / 2 {
                let (lo, hi) = (2 * pair, 2 * pair + 1);
                let term = quadratic_coefficients([weights[lo], weights[hi]], [bits[lo], bits[hi]]);
                for (accumulated, added) in coefficients.iter_mut().zip(term) {
                    *accumulated += added;
                }
            }

            transcript.prover_message(&coefficients);
            let challenge = transcript.squeeze::<F128>();
            claim = horner(&coefficients, challenge);
            row_point.push(challenge);

            fold_in_place(&mut weights, challenge);
            fold_in_place(&mut bits, challenge);
        }

        // The sumcheck ended where its own tables did, which is the identity
        // the verifier will re-derive `m(r)` from.
        debug_assert_eq!(claim, weights[0] * bits[0]);

        Ok(OpeningQuery::Mle {
            point: opening_point(&row_point, &input.fold.zeta),
            target: bits[0],
        })
    }
}

impl<const Q: u128> verifier::Reduction<Q> for GrandProduct {
    type Error = ReductionError;

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningQuery, Self::Error> {
        let shape = input.params.shape();
        let offsets = leaf_offsets(&input.fold.row_images);
        let depths = vec![shape.log_rows(); shape.columns()];

        let claims = verify_product_forest(transcript, &depths, &input.fold.images)?;

        let equality = eq_table(&input.fold.zeta);
        let mut claim = batch_columns(&claims, &equality);

        let rho = &claims[0].point;
        let weights = row_weights(rho, &offsets);

        let mut row_point = Vec::with_capacity(shape.log_rows());
        for _ in 0..shape.log_rows() {
            let coefficients = transcript
                .prover_message::<[F128; 3]>()
                .map_err(|_| ReductionError::MalformedProof)?;
            let at_one = coefficients
                .iter()
                .copied()
                .fold(F128::ZERO, |sum, coefficient| sum + coefficient);
            if coefficients[0] + at_one != claim {
                return Err(ReductionError::RowClaimMismatch);
            }
            let challenge = transcript.squeeze::<F128>();
            claim = horner(&coefficients, challenge);
            row_point.push(challenge);
        }

        // Nothing here checks the sumcheck's terminal value against
        // `R(r) m(r)`, because `m(r)` is exactly what the opening goes on to
        // establish. Dividing the terminal by the public `R(r)` states what
        // `m(r)` must be, and step 6 refuses the proof unless the committed
        // bits agree.
        //
        // `R(r)` is public, so the verifier evaluates it rather than reading it.
        let public_weight = weights
            .iter()
            .zip(eq_table(&row_point))
            .fold(F128::ZERO, |sum, (&weight, at_row)| sum + weight * at_row);
        let inverse = public_weight
            .inv()
            .ok_or(ReductionError::PublicWeightVanished)?;

        Ok(OpeningQuery::Mle {
            point: opening_point(&row_point, &input.fold.zeta),
            target: claim * inverse,
        })
    }
}
