//! The linear claim F2Z is asked to discharge.

use crypto_primitives::LiftElement;
use field::Fq;
use num_traits::ConstZero;
use poly::{ScaledMleEvaluationClaim, eq_table};

use crate::F2ZParams;

/// A Merkle root over the committed codeword.
///
/// A digest, never an operand: nothing in the protocol does arithmetic on it.
/// The newtype keeps it from being passed where a field element belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Root(pub [u8; 32]);

/// A claim one of the pre-transcript checks rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimError {
    /// There is not one weight per row.
    RowWeightCountMismatch,
    /// There is not one weight per column.
    ColumnWeightCountMismatch,
    /// The evaluation point carries more coordinates than the shape indexes.
    PointTooWide,
    /// The scale is zero, so the claim says nothing about the vector.
    ScaleVanished,
}

/// The caller's `x_core`: the weights and the value they are claimed to give.
///
/// F2Z verifies nothing upstream of this. The caller runs its own PIOP, and
/// establishes that its claim holds, that `q` is prime, and that the
/// coefficient factors as `v = v^(1) (x) v^(2)`. A claim whose coefficient
/// does not split that way cannot use this profile: the fold exponentiates
/// `v^(1)` and reconstructs over `v^(2)`, so it consumes the two separately.
///
/// The parameters live in [`F2ZParams`]; this is only the claim against them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearClaim<const Q: u128> {
    row_weights: Vec<Fq<Q>>,
    column_weights: Vec<Fq<Q>>,
    target: Fq<Q>,
}

impl<const Q: u128> LinearClaim<Q> {
    /// Checks the weights against `config` and returns the claim.
    ///
    /// `row_weights` is `v^(1)`, one element per row; `column_weights` is
    /// `v^(2)`, one per column; `target` is the claimed value `mu`.
    pub fn new(
        params: &F2ZParams<Q>,
        row_weights: Vec<Fq<Q>>,
        column_weights: Vec<Fq<Q>>,
        target: Fq<Q>,
    ) -> Result<Self, ClaimError> {
        if row_weights.len() != params.shape().rows() {
            return Err(ClaimError::RowWeightCountMismatch);
        }
        if column_weights.len() != params.shape().columns() {
            return Err(ClaimError::ColumnWeightCountMismatch);
        }

        Ok(Self {
            row_weights,
            column_weights,
            target,
        })
    }

    /// The per-row weights `v^(1)`, over `F_q`.
    pub fn row_weights(&self) -> &[Fq<Q>] {
        &self.row_weights
    }

    /// `pi_q^{-1}(v^(1)_i)`, the canonical representatives the fold
    /// exponentiates.
    ///
    /// The claim lives in `F_q` while the exponent is an integer, and taking
    /// the representative is what bounds it: each is below `q`, so a fold over
    /// `k_1` rows lands in `[0, k_1(q-1)]`.
    pub fn row_exponents(&self) -> Vec<u128> {
        self.row_weights
            .iter()
            .map(|weight| weight.lift())
            .collect()
    }

    /// The per-column weights `v^(2)`, over `F_q`.
    ///
    /// These stay in the field: they are applied only in the reconstruction
    /// that ties the folds back to `mu`, never in the exponent.
    pub fn column_weights(&self) -> &[Fq<Q>] {
        &self.column_weights
    }

    /// The claimed value `mu`.
    pub fn target(&self) -> Fq<Q> {
        self.target
    }

    /// Rewrites a PIOP's terminal evaluation claim as the claim F2Z takes.
    ///
    /// A PIOP ends holding `scale * w(point) = value` for the vector `w` its
    /// constraints ran over. F2Z takes weights that split as
    /// `v^(1) (x) v^(2)`, and an equality weight does split, at the row/column
    /// boundary the shape fixes:
    ///
    /// ```text
    /// w(point) = sum_{i,j} eq(point_row, i) eq(point_col, j) w_{ij}.
    /// ```
    ///
    /// `point` may be narrower than the shape. `w` is zero above its own
    /// length, so appending zero coordinates leaves the value alone: the added
    /// equality factor selects the low half, which is all of `w`.
    ///
    /// The scale multiplies into `v^(1)` rather than dividing out of the
    /// value, which keeps the whole conversion free of inversion. Each scaled
    /// weight is still a field element below `q`, so the fold stays inside the
    /// bound admissibility fixed.
    ///
    /// A zero scale is refused. It would zero every row weight and leave a
    /// claim of `0 = 0` that any witness satisfies.
    pub fn from_evaluation(
        params: &F2ZParams<Q>,
        claim: &ScaledMleEvaluationClaim<Fq<Q>>,
    ) -> Result<Self, ClaimError> {
        let shape = params.shape();
        if claim.point().len() > shape.log_bits() {
            return Err(ClaimError::PointTooWide);
        }
        let scale = claim.scale();
        if scale == Fq::ZERO {
            return Err(ClaimError::ScaleVanished);
        }

        let mut point = claim.point().to_vec();
        point.resize(shape.log_bits(), Fq::ZERO);
        let (rows, columns) = point.split_at(shape.log_rows());

        let row_weights = eq_table(rows)
            .into_iter()
            .map(|weight| weight * scale)
            .collect();

        Self::new(params, row_weights, eq_table(columns), claim.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;
    use field::gf128::smallest_generator;

    /// The largest prime below `2^114`, the top of the sampling range.
    const Q114: u128 = (1 << 114) - 11;

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn params() -> F2ZParams<Q114> {
        F2ZParams::new(Shape::new(7, 15).unwrap(), smallest_generator()).unwrap()
    }

    fn claim(row_weights: Vec<Fq<Q114>>) -> Result<LinearClaim<Q114>, ClaimError> {
        let params = params();
        LinearClaim::new(
            &params,
            row_weights,
            vec![Fq::from(1u128); params.shape().columns()],
            Fq::from(0u128),
        )
    }

    fn weights() -> Vec<Fq<Q114>> {
        (0..params().shape().rows())
            .map(|row| Fq::from(row as u128))
            .collect()
    }

    #[test]
    fn accepts_a_well_formed_statement() {
        let accepted = claim(weights()).unwrap();
        assert_eq!(accepted.row_weights().len(), 1 << 7);
        assert_eq!(accepted.column_weights().len(), 1 << 15);
    }

    #[test]
    fn rejects_weight_vectors_that_do_not_fit_the_shape() {
        assert_eq!(
            claim(vec![Fq::from(1u128)]).err(),
            Some(ClaimError::RowWeightCountMismatch)
        );

        let params = params();
        assert_eq!(
            LinearClaim::new(&params, weights(), vec![Fq::from(1u128)], Fq::from(0u128)).err(),
            Some(ClaimError::ColumnWeightCountMismatch)
        );
    }

    /// The conversion has to agree with what F2Z then does with the claim.
    /// `reconstruct` is the verifier's own tie-back to `mu`, so running it
    /// against the converted weights checks the split at the row/column
    /// boundary, the padding, and the scale division at once.
    #[test]
    fn the_converted_claim_reconstructs_to_the_evaluation() {
        use crate::{fold::fold_columns, reconstruct};

        let params = params();
        let shape = *params.shape();
        let packed: Vec<field::F128> = (0..shape.rows() * shape.columns() / 128)
            .map(|i| field::F128::new((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15), i as u64))
            .collect();
        let table = params.table(&packed).unwrap();

        // One coordinate short of the shape, so the padding path runs.
        let point: Vec<Fq<Q114>> = (0..shape.log_bits() - 1)
            .map(|i| Fq::from((i as u128 + 2) * 0x5bd1_e995))
            .collect();

        // The evaluation, straight from the definition, over the padded point.
        let mut padded = point.clone();
        padded.resize(shape.log_bits(), Fq::from(0u128));
        let (row_point, column_point) = padded.split_at(shape.log_rows());
        let (eq_rows, eq_columns) = (eq_table(row_point), eq_table(column_point));
        let mut evaluation = Fq::from(0u128);
        for (column, &column_weight) in eq_columns.iter().enumerate() {
            for (row, &row_weight) in eq_rows.iter().enumerate() {
                if table.bit(column, row) {
                    evaluation += row_weight * column_weight;
                }
            }
        }

        let scale = Fq::from(0x1234_5678_9abcu128);
        let converted = LinearClaim::from_evaluation(
            &params,
            &ScaledMleEvaluationClaim::new(point.into_boxed_slice(), scale, scale * evaluation),
        )
        .unwrap();

        assert_eq!(
            converted.target(),
            scale * evaluation,
            "the target is the claim's own value",
        );
        assert_eq!(
            reconstruct(
                &converted,
                &fold_columns(&table, &converted.row_exponents())
            )
            .unwrap(),
            scale * evaluation,
            "F2Z's own reconstruction disagrees with the evaluation claim",
        );
    }

    #[test]
    fn a_zero_scale_is_rejected() {
        assert_eq!(
            LinearClaim::from_evaluation(
                &params(),
                &ScaledMleEvaluationClaim::new(
                    vec![Fq::<Q114>::from(1u128); 22].into_boxed_slice(),
                    Fq::from(0u128),
                    Fq::from(0u128),
                ),
            ),
            Err(ClaimError::ScaleVanished)
        );
    }

    #[test]
    fn a_point_wider_than_the_shape_is_rejected() {
        assert_eq!(
            LinearClaim::from_evaluation(
                &params(),
                &ScaledMleEvaluationClaim::new(
                    vec![Fq::<Q114>::from(1u128); 23].into_boxed_slice(),
                    Fq::from(1u128),
                    Fq::from(0u128),
                ),
            ),
            Err(ClaimError::PointTooWide)
        );
    }

    #[test]
    fn a_row_weight_lifts_to_its_canonical_representative() {
        // The type makes an out-of-range weight unrepresentable, so there is
        // no range check to test: `Fq::from` reduces on the way in.
        let mut weights = weights();
        weights[3] = Fq::from(Q114 - 1);
        weights[4] = Fq::from(Q114);
        weights[5] = Fq::from(Q114 + 6);

        let exponents = claim(weights).unwrap().row_exponents();
        assert_eq!(exponents[3], Q114 - 1);
        assert_eq!(exponents[4], 0);
        assert_eq!(exponents[5], 6);
        assert!(exponents.iter().all(|&exponent| exponent < Q114));
    }
}
