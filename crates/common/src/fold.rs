//! The column fold, and the round state both sides hold once it closes.

use field::{F128, FixedBasePow, Fq};
use poly::DenseMultilinearExtension;

use crate::{BitTable, CoreStatement, Shape, table::WORD_BITS};

/// A round whose parts do not describe the shape they belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldError {
    /// There is not one fold and one image per column, or the challenge is not
    /// one coordinate per column variable.
    ColumnCountMismatch,
    /// There is not one row image per row.
    RowCountMismatch,
}

/// `eta_j = sum_i pi_q^{-1}(v^(1)_i) f_ij`, over the integers.
///
/// Walks the column's words and adds the exponent of each set bit, rather than
/// reading `k_1` bits one at a time.
///
/// `exponents` must be the statement's own — [`CoreStatement::row_exponents`].
/// That is what makes the sum safe: each is below `q` and there are `k_1` of
/// them, so it is at most `k_1 (q - 1)`, which admissibility put below `|K|`.
pub fn fold_column(table: &BitTable<'_>, exponents: &[u128], column: usize) -> u128 {
    table
        .column(column)
        .iter()
        .enumerate()
        .map(|(index, &word)| {
            let base = index * WORD_BITS;
            let mut remaining = word;
            let mut total = 0u128;
            while remaining != 0 {
                total += exponents[base + remaining.trailing_zeros() as usize];
                // Clears the lowest set bit.
                remaining &= remaining - 1;
            }
            total
        })
        .sum()
}

/// `y_i = g^{pi_q^{-1}(v^(1)_i)}`, one per row.
///
/// Derived, never transmitted. There are only `k_1` of these — at most `2^14`
/// under the sizing constraint — so unlike the columns they are cheap to hold.
pub fn row_images<const Q: u128>(
    statement: &CoreStatement<Q>,
    generator: &FixedBasePow,
) -> Vec<F128> {
    statement
        .row_exponents()
        .into_iter()
        .map(|exponent| generator.pow(exponent))
        .collect()
}

/// The reconstruction `sum_j v^(2)_j pi_q(eta_j)`, which V requires to equal
/// the claimed `mu`.
///
/// This is what ties the folds back to the caller's claim. Until it holds the
/// folds say nothing about `y`.
///
/// The length is checked rather than zipped away. A short `folds` would
/// otherwise sum over a prefix and return a value that is right for no
/// instance -- and for the common `y = 0` it would look correct.
pub fn reconstruct<const Q: u128>(
    statement: &CoreStatement<Q>,
    folds: &[u128],
) -> Result<Fq<Q>, FoldError> {
    if folds.len() != statement.shape().columns() {
        return Err(FoldError::ColumnCountMismatch);
    }
    Ok(statement
        .column_weights()
        .iter()
        .zip(folds)
        .map(|(&weight, &fold)| weight * Fq::from(fold))
        .sum())
}

/// The state both sides hold once the fold round closes.
///
/// Only `folds` is transmitted. Everything else is derived from it, from the
/// statement, or from the transcript, so the two sides must arrive at
/// identical values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// The column folds `eta_j`, as integers.
    pub folds: Vec<u128>,
    /// `g^{eta_j}`, derived on both sides rather than transmitted.
    pub images: Vec<F128>,
    /// `y_i = g^{pi_q^{-1}(v^(1)_i)}`, one per row.
    pub row_images: Vec<F128>,
    /// The challenge drawn after the images are bound.
    pub zeta: Vec<F128>,
    /// The batched output claim: the multilinear extension of
    /// `j -> g^{eta_j}` evaluated at `zeta`.
    pub e0: F128,
}

impl Fold {
    /// Derives `e0` and takes ownership of the round.
    ///
    /// Every length is checked against `shape`, including `row_images`, which
    /// nothing downstream would otherwise notice was short until Step 4
    /// consumed it.
    pub fn new(
        shape: &Shape,
        folds: Vec<u128>,
        images: Vec<F128>,
        row_images: Vec<F128>,
        zeta: Vec<F128>,
    ) -> Result<Self, FoldError> {
        if row_images.len() != shape.rows() {
            return Err(FoldError::RowCountMismatch);
        }
        if folds.len() != shape.columns() || images.len() != shape.columns() {
            return Err(FoldError::ColumnCountMismatch);
        }
        let e0 = DenseMultilinearExtension::evaluate_at(&images, &zeta)
            .map_err(|_| FoldError::ColumnCountMismatch)?;

        Ok(Self {
            folds,
            images,
            row_images,
            zeta,
            e0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;
    use field::gf128::smallest_generator;

    const Q114: u128 = (1 << 114) - 11;
    /// Comb window: `FixedBasePow` always covers the full 128-bit exponent
    /// range, and `win` trades table size against multiplies per call.
    const WINDOW: u32 = 8;

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    fn statement(row_weights: Vec<Fq<Q114>>, column_weights: Vec<Fq<Q114>>) -> CoreStatement<Q114> {
        CoreStatement::new(
            shape(),
            smallest_generator(),
            row_weights,
            column_weights,
            Fq::from(0u128),
        )
        .unwrap()
    }

    fn witness(shape: &Shape, bits: &[(usize, usize)]) -> Vec<u64> {
        let mut words = vec![0u64; (1 << shape.m()) / WORD_BITS];
        for &(row, column) in bits {
            let index = (column << shape.t()) | row;
            words[index >> 6] |= 1u64 << (index % WORD_BITS);
        }
        words
    }

    #[test]
    fn a_fold_adds_the_weights_of_the_set_rows() {
        let shape = shape();
        let weights: Vec<u128> = (0..shape.rows()).map(|row| (row as u128) * 1_000).collect();
        let field_weights: Vec<Fq<Q114>> = weights.iter().map(|&w| Fq::from(w)).collect();
        let statement = statement(field_weights, vec![Fq::from(1u128); shape.columns()]);

        let words = witness(&shape, &[(1, 0), (5, 0), (127, 0), (64, 4)]);
        let table = BitTable::new(shape, &words).unwrap();

        assert_eq!(
            fold_column(&table, &statement.row_exponents(), 0),
            weights[1] + weights[5] + weights[127]
        );
        assert_eq!(
            fold_column(&table, &statement.row_exponents(), 4),
            weights[64]
        );
        assert_eq!(fold_column(&table, &statement.row_exponents(), 9), 0);
    }

    #[test]
    fn the_widest_fold_reaches_the_shape_bound_without_overflowing() {
        let shape = shape();
        // Every weight at `q - 1` and every bit of the column set.
        let statement = statement(
            vec![Fq::from(Q114 - 1); shape.rows()],
            vec![Fq::from(1u128); shape.columns()],
        );
        let all: Vec<(usize, usize)> = (0..shape.rows()).map(|row| (row, 0)).collect();
        let words = witness(&shape, &all);
        let table = BitTable::new(shape, &words).unwrap();

        assert_eq!(
            fold_column(&table, &statement.row_exponents(), 0),
            statement.fold_bound()
        );
    }

    #[test]
    fn the_fold_agrees_with_the_bit_by_bit_definition() {
        let shape = shape();
        let weights: Vec<Fq<Q114>> = (0..shape.rows())
            .map(|row| Fq::from((row as u128 + 1) * (Q114 / 137)))
            .collect();
        let statement = statement(weights, vec![Fq::from(1u128); shape.columns()]);

        let bits: Vec<(usize, usize)> = (0..shape.rows())
            .filter(|row| row % 3 == 0 || row % 7 == 1)
            .map(|row| (row, 6))
            .collect();
        let words = witness(&shape, &bits);
        let table = BitTable::new(shape, &words).unwrap();

        let expected: u128 = (0..shape.rows())
            .filter(|&row| table.bit(6, row))
            .map(|row| statement.row_exponents()[row])
            .sum();
        assert_eq!(fold_column(&table, &statement.row_exponents(), 6), expected);
    }

    #[test]
    fn the_reconstruction_reduces_a_fold_that_runs_past_the_modulus() {
        let shape = shape();
        let statement = statement(
            vec![Fq::from(1u128); shape.rows()],
            vec![Fq::from(1u128); shape.columns()],
        );

        // One column folding to exactly `q` contributes nothing.
        let mut folds = vec![0u128; shape.columns()];
        folds[0] = Q114;
        assert_eq!(reconstruct(&statement, &folds), Ok(Fq::from(0u128)));

        folds[0] = Q114 + 5;
        assert_eq!(reconstruct(&statement, &folds), Ok(Fq::from(5u128)));
    }

    #[test]
    fn the_row_images_are_the_generator_raised_to_each_weight() {
        let shape = shape();
        let weights: Vec<Fq<Q114>> = (0..shape.rows()).map(|row| Fq::from(row as u128)).collect();
        let statement = statement(weights, vec![Fq::from(1u128); shape.columns()]);
        let generator = FixedBasePow::new(smallest_generator(), WINDOW);

        let images = row_images(&statement, &generator);
        assert_eq!(images.len(), shape.rows());
        assert_eq!(images[0], F128::new(1, 0));
        assert_eq!(images[1], smallest_generator());
        assert_eq!(images[2], smallest_generator() * smallest_generator());
    }

    #[test]
    fn the_reconstruction_refuses_a_fold_vector_that_is_not_one_per_column() {
        let shape = shape();
        let statement = statement(
            vec![Fq::from(1u128); shape.rows()],
            vec![Fq::from(1u128); shape.columns()],
        );

        // The dangerous case: a short vector would sum over a prefix, and at
        // the common `y = 0` an empty one would look correct.
        assert_eq!(
            reconstruct(&statement, &[]),
            Err(FoldError::ColumnCountMismatch)
        );
        assert_eq!(
            reconstruct(&statement, &vec![0u128; shape.columns() - 1]),
            Err(FoldError::ColumnCountMismatch)
        );
        assert_eq!(
            reconstruct(&statement, &vec![0u128; shape.columns() + 1]),
            Err(FoldError::ColumnCountMismatch)
        );
    }

    /// A round whose lengths all match `shape`, so a test can vary one.
    fn parts(shape: &Shape) -> (Vec<u128>, Vec<F128>, Vec<F128>, Vec<F128>) {
        (
            vec![1u128; shape.columns()],
            vec![F128::new(2, 0); shape.columns()],
            vec![F128::new(3, 0); shape.rows()],
            vec![F128::new(5, 0); shape.s()],
        )
    }

    #[test]
    fn a_round_rejects_parts_that_disagree_with_the_shape() {
        let shape = shape();
        let (folds, images, row_images, zeta) = parts(&shape);

        assert!(
            Fold::new(
                &shape,
                folds.clone(),
                images.clone(),
                row_images.clone(),
                zeta.clone()
            )
            .is_ok()
        );

        let mut short_rows = row_images.clone();
        short_rows.pop();
        assert_eq!(
            Fold::new(
                &shape,
                folds.clone(),
                images.clone(),
                short_rows,
                zeta.clone()
            ),
            Err(FoldError::RowCountMismatch)
        );

        let mut short_folds = folds.clone();
        short_folds.pop();
        assert_eq!(
            Fold::new(
                &shape,
                short_folds,
                images.clone(),
                row_images.clone(),
                zeta.clone()
            ),
            Err(FoldError::ColumnCountMismatch)
        );

        let mut short_images = images.clone();
        short_images.pop();
        assert_eq!(
            Fold::new(
                &shape,
                folds.clone(),
                short_images,
                row_images.clone(),
                zeta.clone()
            ),
            Err(FoldError::ColumnCountMismatch)
        );

        let mut short_zeta = zeta.clone();
        short_zeta.pop();
        assert_eq!(
            Fold::new(&shape, folds, images, row_images, short_zeta),
            Err(FoldError::ColumnCountMismatch)
        );
    }

    #[test]
    fn the_batched_claim_is_the_image_extension_at_the_challenge() {
        // Two variables, so coordinate order is observable -- with one, a
        // swapped challenge could not be told apart.
        let shape = Shape::new(7, 15).unwrap();
        let images: Vec<F128> = (0..shape.columns())
            .map(|column| F128::new(column as u64 + 1, 0))
            .collect();
        let zeta: Vec<F128> = (0..shape.s())
            .map(|index| F128::new(index as u64 + 2, 0))
            .collect();
        let round = Fold::new(
            &shape,
            vec![0; shape.columns()],
            images.clone(),
            vec![F128::new(1, 0); shape.rows()],
            zeta.clone(),
        )
        .unwrap();

        // `sum_c nu_c * eq(c, zeta)`, the extension written out directly.
        let expected: F128 = (0..shape.columns())
            .map(|column| {
                let weight = zeta
                    .iter()
                    .enumerate()
                    .fold(F128::new(1, 0), |acc, (bit, &z)| {
                        let set = (column >> bit) & 1 == 1;
                        acc * if set { z } else { F128::new(1, 0) + z }
                    });
                images[column] * weight
            })
            .fold(F128::new(0, 0), |acc, term| acc + term);
        assert_eq!(round.e0, expected);
    }
}
