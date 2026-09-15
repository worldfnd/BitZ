//! Grand-product reduction to a factored inner-product claim over committed bits.
//!
//! `fold.e0` is the column-product evaluation at `fold.zeta`.
//! GKR reduces it to a leaf evaluation. Removing the constant-one term yields
//! row weights `(fold.row_images[row] - 1) * eq_table(alfa_b)[row]` and
//! column weights `eq_table(&point[..r2])[column]`. The terminal point stores
//! column coordinates first; `alfa_b` contains the remaining row coordinates.
//! The caller must verify the returned claim against the commitment.

use common::{ClaimError, Fold, LinearClaim, OpeningQuery, Shape};
use field::F128;
use num_traits::ConstOne;
use transcript::VerifierState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceError {
    /// The GKR transcript does not satisfy a sumcheck or child-product relation.
    GKR,
    /// The derived weight counts do not match the configured shape.
    Claim(ClaimError),
}

pub(crate) fn gkr_reduce(
    transcript: &mut VerifierState,
    fold: &Fold,
    shape: &Shape,
) -> Result<OpeningQuery, ReduceError> {
    // Each layer halves the row count, leaving one product per column.
    let r1 = fold.row_images.len().max(1).ilog2();

    let (point, mle_leaf_claim) =
        gkr::gpgkr_verify(transcript, fold.e0, &fold.zeta, r1).ok_or(ReduceError::GKR)?;

    let r2 = point.len() - r1 as usize;
    // Columns occupy the low index bits of the GKR leaf table.
    let alfa_b = &point[r2..];

    let inner_product_claim = mle_leaf_claim - F128::ONE;
    // Allocates 2*l1 space if the compiler doesn't fuse.
    let u1: Vec<_> = fold
        .row_images
        .iter()
        .zip(poly::eq_table(alfa_b))
        .map(|(a, b)| (*a - F128::ONE) * b) // Does the later step benefit from wide mul?
        .collect();

    let u2 = poly::eq_table(&point[..r2]);
    let claim =
        LinearClaim::from_shape(shape, u1, u2, inner_product_claim).map_err(ReduceError::Claim)?;
    Ok(OpeningQuery::InnerProduct { claim })
}

#[cfg(test)]
mod round_trip_ai_test {
    use common::{BitZParams, Fold, Shape};
    use field::gf128::smallest_generator;
    use gkr::{GrandProductCircuit, gpgkr_prove};
    use num_traits::{ConstOne, ConstZero, identities::Zero};

    use super::*;

    const Q: u128 = (1 << 114) - 11;

    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    fn params() -> BitZParams<Q> {
        BitZParams::new(shape(), smallest_generator()).unwrap()
    }

    #[test]
    fn verifier_agrees_with_a_real_prover_transcript() {
        let shape = shape();

        let mut packed = vec![F128::ZERO; (1 << shape.log_bits()) / 128];
        for column in 0..shape.columns() {
            for row in 0..shape.rows() {
                let h = (row as u64).wrapping_mul(2654435761)
                    ^ (column as u64).wrapping_mul(0x9E3779B97F4A7C15);
                if (h >> 5) & 1 == 1 {
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
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        // Build affine leaves directly so the verifier test needs no prover dependency.
        let mut leafs = vec![F128::zero(); shape.columns() * shape.rows()];
        for b in 0..shape.rows() {
            for c in 0..shape.columns() {
                leafs[b * shape.columns() + c] = if table.bit(c, b) {
                    row_images[b]
                } else {
                    F128::ONE
                };
            }
        }
        let circuit = GrandProductCircuit::new(leafs);
        let (top_layer, witnesses) = circuit.batched_eval(shape.columns());

        // Fold::new evaluates top_layer at zeta to obtain GKR's initial claim.
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, top_layer, row_images.clone(), zeta.clone()).unwrap();

        let mut prover = transcript::build_prover("verifier-round-trip", &F128::ZERO);
        let (mut point, claim) = gpgkr_prove(&mut prover, &zeta, witnesses);
        let proof = prover.finish();

        // Derive the expected factors from the prover's terminal point.
        let r1 = row_images.len().max(1).ilog2();
        let r2 = point.len() - r1 as usize;
        let expected_alfa_b = point.split_off(r2);
        let expected_u1: Vec<F128> = row_images
            .iter()
            .zip(poly::eq_table(&expected_alfa_b))
            .map(|(a, b)| (*a - F128::ONE) * b)
            .collect();
        let expected_inner_product_claim = claim - F128::ONE;

        let mut verifier = transcript::build_verifier("verifier-round-trip", &F128::ZERO, &proof);
        let query = gkr_reduce(&mut verifier, &fold, &shape).unwrap();
        verifier.check_eof().unwrap();
        let expected = LinearClaim::from_shape(
            &shape,
            expected_u1,
            poly::eq_table(&point),
            expected_inner_product_claim,
        )
        .unwrap();
        assert_eq!(query, OpeningQuery::InnerProduct { claim: expected });
    }
}
