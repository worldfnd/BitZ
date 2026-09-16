//! Grand-product reduction to a factored inner-product claim over committed bits.
//!
//! Each leaf is `1 + (fold.row_images[row] - 1) * table.bit(column, row)`.
//! At the terminal GKR point, subtracting one leaves the weighted sum of bits:
//! `sum(row, column) u1[row] * u2[column] * table.bit(column, row)`.
//! The caller must discharge this claim through the commitment opening.

use common::{BitTable, ClaimError, Fold, LinearClaim, OpeningQuery, TransposeError};
use field::F128;
use gkr::{GrandProductCircuit, gpgkr_prove};
use num_traits::ConstOne;
use poly::eq_table;
use transcript::ProverState;

#[inline(never)]
fn init_circuit(table: &BitTable, fold: &Fold) -> GrandProductCircuit {
    let columns = table.shape().columns();
    let dim = columns * table.shape().rows();
    let mut leafs = F128::zeroed_vec(dim);

    // TODO optimisation: Handle the leafs and the two layers above it lazily.
    // Columns occupy the low index bits, so each product tree reduces one column.
    match table.transpose() {
        Ok(transposed) => {
            // Transpose wide tables so each row can be read sequentially.
            let transposed = transposed.as_table();
            for (b, &row_image) in fold.row_images.iter().enumerate() {
                let leafs = &mut leafs[b * columns..(b + 1) * columns];
                for (leaf, bit) in leafs.iter_mut().zip(transposed.column_bits(b)) {
                    *leaf = if bit { row_image } else { F128::ONE };
                }
            }
        }
        Err(TransposeError::ColumnCountTooNarrow) => {
            // Narrow tables cannot form packed columns after transposition.
            for (b, &row_image) in fold.row_images.iter().enumerate() {
                let leafs = &mut leafs[b * columns..(b + 1) * columns];
                for (c, leaf) in leafs.iter_mut().enumerate() {
                    *leaf = if table.bit(c, b) {
                        row_image
                    } else {
                        F128::ONE
                    };
                }
            }
        }
    }

    GrandProductCircuit::new(leafs)
}

/// Reduces the grand-product circuit to a factored claim on the committed bits.
pub fn gkr_reduce(
    transcript: &mut ProverState,
    fold: &Fold,
    table: &BitTable,
) -> Result<OpeningQuery, ClaimError> {
    let circuit = init_circuit(table, fold);
    let (_last_value, witnesses) = circuit.batched_eval(table.shape().columns());

    let (mut point, claim) = gpgkr_prove(transcript, &fold.zeta, witnesses);

    // The multilinear extension of the constant-one table is one at every point.
    let inner_product_claim = claim - F128::ONE;

    let r1 = fold.row_images.len().max(1).ilog2();
    let r2 = point.len() - r1 as usize;
    // GKR returns coordinates in low-index-bit-first order: columns, then rows.
    let alfa_b = point.split_off(r2);
    let alfa_c = point;

    let u1: Vec<_> = fold
        .row_images
        .iter()
        .zip(poly::eq_table(&alfa_b))
        .map(|(a, b)| (*a - F128::ONE) * b) // Does the later step benefit from wide mul?
        .collect();

    let u2 = eq_table(&alfa_c);
    let claim = LinearClaim::from_shape(table.shape(), u1, u2, inner_product_claim)?;
    Ok(OpeningQuery::InnerProduct { claim })
}

#[cfg(test)]
mod order_check_ai_test {
    use super::*;
    use common::{BitZParams, Fold, Shape};
    use field::gf128::smallest_generator;
    use num_traits::ConstZero;

    const Q: u128 = (1 << 114) - 11;

    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    fn params() -> BitZParams<Q> {
        BitZParams::new(shape(), smallest_generator()).unwrap()
    }

    fn packed_witness(shape: &Shape, bit_fn: impl Fn(usize, usize) -> bool) -> Vec<F128> {
        let mut packed = vec![F128::ZERO; (1 << shape.log_bits()) / 128];
        for column in 0..shape.columns() {
            for row in 0..shape.rows() {
                if bit_fn(column, row) {
                    let index = (column << shape.log_rows()) | row;
                    let element = &mut packed[index >> 7];
                    let offset = index % 128;
                    if offset < 64 {
                        element.lo |= 1u64 << offset;
                    } else {
                        element.hi |= 1u64 << (offset - 64);
                    }
                }
            }
        }
        packed
    }

    #[test]
    fn factored_claim_matches_the_bit_table() {
        let shape = shape();

        // Vary both dimensions to exercise the row/column split and coordinate order.
        let packed = packed_witness(&shape, |c, b| {
            let h =
                (b as u64).wrapping_mul(2654435761) ^ (c as u64).wrapping_mul(0x9E3779B97F4A7C15);
            (h >> 5) & 1 == 1
        });
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        let images = vec![F128::ONE; shape.columns()];
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();

        let mut prover = transcript::build_prover("order-check", &F128::ZERO);
        let query = gkr_reduce(&mut prover, &fold, &table).unwrap();
        check_query(&query, &table);
    }

    #[test]
    fn factored_claim_matches_for_narrow_tables() {
        const Q100: u128 = (1 << 100) - 15;

        // Cover one column and both sides of the 128-column transpose boundary.
        for log_columns in [0, 6, 7] {
            let shape = Shape::new(22 - log_columns, log_columns).unwrap();
            let params = BitZParams::<Q100>::new(shape, smallest_generator()).unwrap();
            let packed = packed_witness(&shape, |column, row| {
                let bits = (row as u64).wrapping_mul(0x9E3779B97F4A7C15)
                    ^ (column as u64).wrapping_mul(0xD1B54A32D192ED03);
                bits >> 63 & 1 == 1
            });
            let table = params.table(&packed).unwrap();
            let row_images = (0..shape.rows())
                .map(|row| F128::from((row as u128 + 1) * 0x9E3779B97F4A7C15 + 7))
                .collect();
            let zeta = (0..shape.log_columns())
                .map(|index| F128::from((index as u128 + 3) * 0xABCDEF12345 + 1))
                .collect();
            let fold = Fold::new(
                &shape,
                vec![0; shape.columns()],
                vec![F128::ONE; shape.columns()],
                row_images,
                zeta,
            )
            .unwrap();

            let mut prover = transcript::build_prover("narrow-table", &F128::ZERO);
            let query = gkr_reduce(&mut prover, &fold, &table).unwrap();
            check_query(&query, &table);
        }
    }

    // Identical columns make the leaf evaluation independent of column coordinates.
    #[test]
    fn b_only_pattern_isolates_the_row_pathway() {
        let shape = shape();

        let packed = packed_witness(&shape, |_c, b| (b * 2654435761) >> 5 & 1 == 1);
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        let images = vec![F128::ONE; shape.columns()];
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();

        let mut prover = transcript::build_prover("order-check-b", &F128::ZERO);
        let query = gkr_reduce(&mut prover, &fold, &table).unwrap();
        check_query(&query, &table);
    }

    fn check_query(query: &OpeningQuery, table: &BitTable<'_>) {
        let OpeningQuery::InnerProduct { claim } = query else {
            panic!("expected a factored inner-product claim");
        };
        assert_eq!(claim.row_weights().len(), table.shape().rows());
        assert_eq!(claim.column_weights().len(), table.shape().columns());
        let mut target = F128::ZERO;
        for (column, &column_weight) in claim.column_weights().iter().enumerate() {
            for (row, &row_weight) in claim.row_weights().iter().enumerate() {
                if table.bit(column, row) {
                    target += row_weight * column_weight;
                }
            }
        }
        assert_eq!(target, claim.target());
    }
}
