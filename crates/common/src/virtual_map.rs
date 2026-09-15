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

use field::F128;

/// A map that does not describe the protocol it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualMapError {
    /// The point does not have enough coordinates to index every entry of `h`.
    PointLengthMismatch,
    /// The transposed weights are not one per coordinate of `1 ‖ f`.
    WeightCountMismatch,
}

/// What the protocol asks of `M`.
pub trait VirtualMap {
    /// `M^T eq(point, .)`, indexed over `1 ‖ f`.
    ///
    /// Takes a point rather than a weight vector because the reduction
    /// normalises its claim to a point before handing it over. The point may
    /// index zero-padded entries beyond `h_len()`; their weights are ignored.
    fn transpose_eq(&self, point: &[F128]) -> Result<TransposedWeights, VirtualMapError>;

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
/// `M`'s first row is the constant one, so `M^T v` is indexed over `1 ‖ f`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{ConstOne, ConstZero};
    use poly::eq_table;

    /// Dense `M`, the reference the sparse implementations are checked against.
    struct DenseMap {
        /// Row `i` is the `1 ‖ f` indicator of the bits `h_i` sums.
        rows: Vec<Vec<bool>>,
        columns: usize,
    }

    impl VirtualMap for DenseMap {
        fn transpose_eq(&self, point: &[F128]) -> Result<TransposedWeights, VirtualMapError> {
            if point.len() < self.h_len().next_power_of_two().trailing_zeros() as usize {
                return Err(VirtualMapError::PointLengthMismatch);
            }
            let weights = eq_table(point);
            let mut transposed = vec![F128::ZERO; self.columns];
            for (row, weight) in self.rows.iter().zip(weights) {
                for (column, set) in row.iter().enumerate() {
                    if *set {
                        transposed[column] += weight;
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
        let point = [F128::new(7, 11), F128::new(3, 5)];
        let weights = eq_table(&point);

        for bits in 0..4_u32 {
            let f = [bits & 1 == 1, bits >> 1 & 1 == 1];
            let h = [true, f[0], f[1], f[0] ^ f[1]];

            let direct = h
                .iter()
                .zip(&weights)
                .filter(|(bit, _)| **bit)
                .fold(F128::ZERO, |sum, (_, weight)| sum + *weight);

            let transposed = map.transpose_eq(&point).unwrap();
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
    fn a_point_of_the_wrong_width_is_rejected() {
        assert_eq!(
            map().transpose_eq(&[F128::ONE]),
            Err(VirtualMapError::PointLengthMismatch)
        );
    }

    #[test]
    fn a_point_that_indexes_zero_padding_is_accepted() {
        assert!(map().transpose_eq(&[F128::ONE; 3]).is_ok());
    }
}
