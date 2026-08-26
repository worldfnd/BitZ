//! The core statement: the claim F2Z is asked to discharge.

use crypto_primitives::LiftElement;
use field::Fq;

use crate::F2ZConfig;

/// A Merkle root over the committed codeword.
///
/// A digest, never an operand: nothing in the protocol does arithmetic on it.
/// The newtype keeps it from being passed where a field element belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Root(pub [u8; 32]);

/// A statement one of the pre-transcript checks rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementError {
    /// There is not one weight per row.
    RowWeightCountMismatch,
    /// There is not one weight per column.
    ColumnWeightCountMismatch,
}

/// The caller's `x_core`: the weights and the value they are claimed to give.
///
/// F2Z verifies nothing upstream of this. The caller runs its own PIOP, and
/// establishes that its claim holds, that `q` is prime, and that the
/// coefficient factors as `v = v^(1) (x) v^(2)`. A claim whose coefficient
/// does not split that way cannot use this profile: the fold exponentiates
/// `v^(1)` and reconstructs over `v^(2)`, so it consumes the two separately.
///
/// The parameters live in [`F2ZConfig`]; this is only the claim against them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreStatement<const Q: u128> {
    row_weights: Vec<Fq<Q>>,
    column_weights: Vec<Fq<Q>>,
    target: Fq<Q>,
}

impl<const Q: u128> CoreStatement<Q> {
    /// Checks the weights against `config` and returns the claim.
    ///
    /// `row_weights` is `v^(1)`, one element per row; `column_weights` is
    /// `v^(2)`, one per column; `target` is the claimed value `mu`.
    pub fn new(
        config: &F2ZConfig<Q>,
        row_weights: Vec<Fq<Q>>,
        column_weights: Vec<Fq<Q>>,
        target: Fq<Q>,
    ) -> Result<Self, StatementError> {
        if row_weights.len() != config.shape().rows() {
            return Err(StatementError::RowWeightCountMismatch);
        }
        if column_weights.len() != config.shape().columns() {
            return Err(StatementError::ColumnWeightCountMismatch);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;
    use field::gf128::smallest_generator;

    /// The largest prime below `2^114`, the top of the sampling range.
    const Q114: u128 = (1 << 114) - 11;

    const WINDOW: u32 = 8;

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn config() -> F2ZConfig<Q114> {
        F2ZConfig::new(Shape::new(7, 15).unwrap(), smallest_generator(), WINDOW).unwrap()
    }

    fn statement(row_weights: Vec<Fq<Q114>>) -> Result<CoreStatement<Q114>, StatementError> {
        let config = config();
        CoreStatement::new(
            &config,
            row_weights,
            vec![Fq::from(1u128); config.shape().columns()],
            Fq::from(0u128),
        )
    }

    fn weights() -> Vec<Fq<Q114>> {
        (0..config().shape().rows())
            .map(|row| Fq::from(row as u128))
            .collect()
    }

    #[test]
    fn accepts_a_well_formed_statement() {
        let accepted = statement(weights()).unwrap();
        assert_eq!(accepted.row_weights().len(), 1 << 7);
        assert_eq!(accepted.column_weights().len(), 1 << 15);
    }

    #[test]
    fn rejects_weight_vectors_that_do_not_fit_the_shape() {
        assert_eq!(
            statement(vec![Fq::from(1u128)]).err(),
            Some(StatementError::RowWeightCountMismatch)
        );

        let config = config();
        assert_eq!(
            CoreStatement::new(&config, weights(), vec![Fq::from(1u128)], Fq::from(0u128)).err(),
            Some(StatementError::ColumnWeightCountMismatch)
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

        let exponents = statement(weights).unwrap().row_exponents();
        assert_eq!(exponents[3], Q114 - 1);
        assert_eq!(exponents[4], 0);
        assert_eq!(exponents[5], 6);
        assert!(exponents.iter().all(|&exponent| exponent < Q114));
    }
}
