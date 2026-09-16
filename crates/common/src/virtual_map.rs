//! The public map `h = M (1 ‖ f)`, where `f` is the committed bit witness and
//! `h` is the vector the incoming claim is about.
//!
//! BitZ is handed a claim about `h`, but the oracle commits to `f`. The fold and
//! the grand product never read the oracle, so they run against `h` as it is.
//! The opening does read it, so the claim is rewritten first:
//!
//! ```text
//! <v, h> = <v, M (1 || f)> = <M^T v, (1 || f)>.
//! ```
//!
//! `M` stays with the caller. This crate needs only `M^T v` and a digest.

use field::{F128, Fq};
use num_traits::ConstZero;

use crate::{
    BitZParams, ClaimError, LinearClaim, OpeningQuery, Shape, VirtualParams, VirtualParamsError,
};

/// A claim on virtual bits `h = M (1 || f)` with checked dimensions.
///
/// Both roles bind the virtual domain, commitment root, shapes, modulus,
/// generator, map digest, and input claim to the transcript before folding.
#[derive(Debug)]
pub struct VirtualStatement<'a, const Q: u128, M: VirtualMap> {
    params: VirtualParams<Q>,
    map: &'a M,
    claim: &'a LinearClaim<Fq<Q>>,
}

/// A map or claim with mismatched dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualStatementError {
    /// The map does not fit the virtual or committed witness shape.
    Parameters(VirtualParamsError),
    /// The input claim's factors do not match the virtual witness shape.
    Claim(ClaimError),
}

/// A map that does not describe the protocol it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualMapError {
    /// The evaluation point does not index the padded virtual witness.
    PointLengthMismatch,
    /// The inner-product coefficients do not cover the padded virtual witness.
    ClaimWeightCountMismatch,
    /// The input omits coordinates of `h`, or the output does not have one weight per bit of `f`.
    WeightCountMismatch,
}

/// What the protocol asks of `M`.
pub trait VirtualMap {
    /// Computes `M^T weights`, splitting off the constant-one coordinate.
    ///
    /// `weights[i]` multiplies bit `h[i]`. At least `h_len()` weights are required;
    /// any remaining weights multiply zero-padded bits of `h` and are ignored.
    /// The result has `f_len() - 1` bit weights and the constant-column weight.
    fn transpose(&self, weights: &[F128]) -> Result<TransposedWeights, VirtualMapError>;

    /// Binds the map into the statement frame.
    ///
    /// Must cover the shape and every nonzero: equal digests are treated as
    /// the same public input.
    fn digest(&self) -> [u8; 32];

    /// The number of coordinates of `h`.
    fn h_len(&self) -> usize;

    /// The number of coordinates of `1 ‖ f`, one more than the committed bits.
    fn f_len(&self) -> usize;
}

/// `M^T v`, split at the coordinate that multiplies the constant one.
///
/// `M`'s first column multiplies the constant one, so `M^T v` is indexed over `1 ‖ f`.
/// Holding the leading coordinate back leaves `f` as the committed vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransposedWeights {
    weights: Vec<F128>,
    constant_weight: F128,
}

impl TransposedWeights {
    /// Callers check the lengths against the shape.
    pub fn new(weights: Vec<F128>, constant_weight: F128) -> Self {
        Self {
            weights,
            constant_weight,
        }
    }

    pub fn weights(&self) -> &[F128] {
        &self.weights
    }

    /// The weight on the constant-one coordinate. A field element, not the
    /// one it multiplies.
    pub fn constant_weight(&self) -> F128 {
        self.constant_weight
    }

    /// `<M^T v, (1 || f)> = constant_weight + <weights, f>`, so the claim on
    /// the committed bits carries `target - constant_weight`.
    pub fn adjusted_target(&self, target: F128) -> F128 {
        target - self.constant_weight
    }

    pub fn into_weights(self) -> Vec<F128> {
        self.weights
    }
}

impl<'a, const Q: u128, M: VirtualMap> VirtualStatement<'a, Q, M> {
    /// Checks the map and claim against both witness shapes.
    ///
    /// The statement retains the checked inputs. The map must keep the same
    /// linear transformation while the statement borrows it.
    pub fn new(
        claim_params: BitZParams<Q>,
        committed_shape: Shape,
        map: &'a M,
        claim: &'a LinearClaim<Fq<Q>>,
    ) -> Result<Self, VirtualStatementError> {
        let params = VirtualParams::new(claim_params, committed_shape, map)
            .map_err(VirtualStatementError::Parameters)?;
        if claim.row_weights().len() != claim_params.shape().rows() {
            return Err(VirtualStatementError::Claim(
                ClaimError::RowWeightCountMismatch,
            ));
        }
        if claim.column_weights().len() != claim_params.shape().columns() {
            return Err(VirtualStatementError::Claim(
                ClaimError::ColumnWeightCountMismatch,
            ));
        }
        Ok(Self { params, map, claim })
    }

    pub fn params(&self) -> &VirtualParams<Q> {
        &self.params
    }

    pub fn map(&self) -> &M {
        self.map
    }

    pub fn claim(&self) -> &LinearClaim<Fq<Q>> {
        self.claim
    }

    /// Rewrites a claim on padded virtual bits into an opening on committed bits.
    ///
    /// Coefficients use `column * row_count + row` order. The map drops weights on
    /// virtual padding. The opening subtracts the constant-column weight from the
    /// target and zero-pads the remaining weights to the commitment size. A
    /// single-column `InnerProduct` holds the dense weights with column weight one.
    pub fn transpose_query(&self, query: OpeningQuery) -> Result<OpeningQuery, VirtualMapError> {
        let log_bits = self.params.claim().shape().log_bits();
        let (weights, target) = match query {
            OpeningQuery::Mle { point, target } => {
                if point.len() != log_bits {
                    return Err(VirtualMapError::PointLengthMismatch);
                }
                (poly::eq_table(&point), target)
            }
            OpeningQuery::InnerProduct { claim } => {
                if claim
                    .row_weights()
                    .len()
                    .checked_mul(claim.column_weights().len())
                    != Some(1 << log_bits)
                {
                    return Err(VirtualMapError::ClaimWeightCountMismatch);
                }
                let weights = claim
                    .column_weights()
                    .iter()
                    .flat_map(|column| claim.row_weights().iter().map(move |row| *row * *column))
                    .collect();
                (weights, claim.target())
            }
        };
        let transposed = self.map.transpose(&weights)?;
        if Some(transposed.weights().len()) != self.map.f_len().checked_sub(1) {
            return Err(VirtualMapError::WeightCountMismatch);
        }
        let target = transposed.adjusted_target(target);
        let mut weights = transposed.into_weights();
        weights.resize(1 << self.params.committed_shape().log_bits(), F128::ZERO);
        let shape = Shape::new(self.params.committed_shape().log_bits(), 0)
            .expect("a valid committed bit count permits a single-column shape");
        let claim = LinearClaim::from_shape(&shape, weights, vec![F128::from(1u64)], target)
            .map_err(|_| VirtualMapError::WeightCountMismatch)?;
        Ok(OpeningQuery::InnerProduct { claim })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{ConstOne, ConstZero};

    /// Dense `M`, the reference the sparse implementations are checked against.
    struct DenseMap {
        /// Row `i` is the `1 ‖ f` indicator of the bits `h_i` sums.
        rows: Vec<Vec<bool>>,
        columns: usize,
    }

    impl VirtualMap for DenseMap {
        fn transpose(&self, weights: &[F128]) -> Result<TransposedWeights, VirtualMapError> {
            if weights.len() < self.h_len() {
                return Err(VirtualMapError::WeightCountMismatch);
            }
            let mut transposed = vec![F128::ZERO; self.columns];
            for (row, weight) in self.rows.iter().zip(weights) {
                for (column, set) in row.iter().enumerate() {
                    if *set {
                        transposed[column] += *weight;
                    }
                }
            }
            let constant_weight = transposed[0];
            Ok(TransposedWeights::new(
                transposed[1..].to_vec(),
                constant_weight,
            ))
        }

        fn digest(&self) -> [u8; 32] {
            [0; 32]
        }

        fn h_len(&self) -> usize {
            self.rows.len()
        }

        fn f_len(&self) -> usize {
            self.columns
        }
    }

    /// `h_0 = 1`, `h_1 = f_0`, `h_2 = f_1`, `h_3 = f_0 + f_1`.
    fn map() -> DenseMap {
        DenseMap {
            rows: vec![
                vec![true, false, false],
                vec![false, true, false],
                vec![false, false, true],
                vec![false, true, true],
            ],
            columns: 3,
        }
    }

    /// `<M^T v, (1 || f)>` must equal `<v, h>` for the `h` the map defines.
    #[test]
    fn transposing_preserves_the_claim() {
        let map = map();
        let weights = [
            F128::new(7, 11),
            F128::new(3, 5),
            F128::new(13, 17),
            F128::new(19, 23),
        ];

        for bits in 0..4_u32 {
            let f = [bits & 1 == 1, bits >> 1 & 1 == 1];
            let h = [true, f[0], f[1], f[0] ^ f[1]];

            let direct = h
                .iter()
                .zip(&weights)
                .filter(|(bit, _)| **bit)
                .fold(F128::ZERO, |sum, (_, weight)| sum + *weight);

            let transposed = map.transpose(&weights).unwrap();
            let through_f = f
                .iter()
                .zip(transposed.weights())
                .filter(|(bit, _)| **bit)
                .fold(transposed.constant_weight(), |sum, (_, weight)| {
                    sum + *weight
                });

            assert_eq!(direct, through_f, "bits {bits:02b}");
        }
    }

    /// The constant weight is folded into the target.
    #[test]
    fn the_adjusted_target_removes_the_constant_weight() {
        let transposed = TransposedWeights::new(vec![F128::ONE], F128::new(9, 0));
        let target = F128::new(13, 2);

        assert_eq!(
            transposed.adjusted_target(target),
            target - F128::new(9, 0),
            "the constant coefficient is public and leaves the claim"
        );
    }

    #[test]
    fn missing_virtual_witness_weights_are_rejected() {
        assert_eq!(
            map().transpose(&[F128::ONE; 3]),
            Err(VirtualMapError::WeightCountMismatch)
        );
    }

    #[test]
    fn weights_on_zero_padding_do_not_change_the_transpose() {
        let map = map();
        let weights = [F128::new(7, 11); 4];
        let mut padded = weights.to_vec();
        padded.extend([F128::new(13, 17); 4]);
        assert_eq!(map.transpose(&weights), map.transpose(&padded));
    }

    const Q: u128 = (1 << 114) - 11;

    fn params() -> BitZParams<Q> {
        let shape = Shape::new(7, 15).unwrap();
        BitZParams::new(shape, field::gf128::smallest_generator()).unwrap()
    }

    fn input_claim(params: &BitZParams<Q>) -> LinearClaim<Fq<Q>> {
        LinearClaim::new(
            params,
            vec![Fq::ONE; params.shape().rows()],
            vec![Fq::ONE; params.shape().columns()],
            Fq::from(0u128),
        )
        .unwrap()
    }

    #[test]
    fn mle_and_factored_queries_transpose_to_the_same_dense_opening() {
        let map = map();
        let params = params();
        let input_claim = input_claim(&params);
        let statement = VirtualStatement::new(params, *params.shape(), &map, &input_claim).unwrap();
        let point = vec![F128::new(7, 11); params.shape().log_bits()];
        let rows = poly::eq_table(&point[..7]);
        let columns = poly::eq_table(&point[7..]);
        let target = F128::new(13, 17);
        let claim =
            LinearClaim::from_shape(params.shape(), rows.clone(), columns.clone(), target).unwrap();
        let factored = statement
            .transpose_query(OpeningQuery::InnerProduct { claim })
            .unwrap();
        let mle = statement
            .transpose_query(OpeningQuery::Mle { point, target })
            .unwrap();
        assert_eq!(factored, mle);

        let OpeningQuery::InnerProduct { claim } = factored else {
            panic!("transposition must produce an inner-product opening");
        };
        assert_eq!(claim.column_weights(), &[F128::ONE]);
        assert_eq!(
            claim.row_weights().len(),
            1 << statement.params().committed_shape().log_bits()
        );
        assert_eq!(claim.row_weights()[0], (rows[1] + rows[3]) * columns[0]);
        assert_eq!(claim.row_weights()[1], (rows[2] + rows[3]) * columns[0]);
        assert!(claim.row_weights()[2..].iter().all(|w| *w == F128::ZERO));
        assert_eq!(claim.target(), target - rows[0] * columns[0]);
    }

    #[test]
    fn opening_queries_must_cover_the_configured_virtual_shape() {
        let map = map();
        let params = params();
        let input_claim = input_claim(&params);
        let statement = VirtualStatement::new(params, *params.shape(), &map, &input_claim).unwrap();
        assert_eq!(
            statement.transpose_query(OpeningQuery::Mle {
                point: vec![F128::ONE; 21],
                target: F128::ZERO,
            }),
            Err(VirtualMapError::PointLengthMismatch),
        );
        let shape = Shape::new(8, 15).unwrap();
        let claim = LinearClaim::from_shape(
            &shape,
            vec![F128::ONE; shape.rows()],
            vec![F128::ONE; shape.columns()],
            F128::ZERO,
        )
        .unwrap();
        assert_eq!(
            statement.transpose_query(OpeningQuery::InnerProduct { claim }),
            Err(VirtualMapError::ClaimWeightCountMismatch),
        );
    }

    #[test]
    fn statement_rejects_input_claims_with_mismatched_factors() {
        let params = params();
        let map = map();
        for (shape, expected) in [
            (
                Shape::new(8, 14).unwrap(),
                ClaimError::RowWeightCountMismatch,
            ),
            (
                Shape::new(7, 16).unwrap(),
                ClaimError::ColumnWeightCountMismatch,
            ),
        ] {
            let other_params = BitZParams::new(shape, params.generator()).unwrap();
            let claim = input_claim(&other_params);
            assert_eq!(
                VirtualStatement::new(params, *params.shape(), &map, &claim).err(),
                Some(VirtualStatementError::Claim(expected)),
            );
        }
    }

    #[test]
    fn statement_rejects_maps_that_do_not_fit_the_committed_shape() {
        let params = params();
        let claim = input_claim(&params);
        let mut map = map();
        for (columns, expected) in [
            (0, VirtualParamsError::MissingConstantColumn),
            (
                (1 << params.shape().log_bits()) + 2,
                VirtualParamsError::CommittedShapeTooSmall,
            ),
        ] {
            map.columns = columns;
            assert_eq!(
                VirtualStatement::new(params, *params.shape(), &map, &claim).err(),
                Some(VirtualStatementError::Parameters(expected)),
            );
        }
    }
}
