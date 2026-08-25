//! The core statement: the claim F2Z is asked to discharge.

use crypto_primitives::LiftElement;
use field::{F128, Fq, gf128::is_generator};
use spongefish::Encoding;

use crate::Shape;

/// A Merkle root over the committed codeword.
///
/// A digest, never an operand: nothing in the protocol does arithmetic on it.
/// The newtype keeps it from being passed where a field element belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Root(pub [u8; 32]);

/// `log2 |K|`. The extension is `F_2^128`, so a fold has 128 bits of room.
pub const EXTENSION_BITS: u32 = 128;

/// A statement one of the pre-transcript checks rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementError {
    /// `Q` is at or above `|K| / k_1`, so two folds could collide in the
    /// exponent. This couples the modulus to the shape and is the only gate
    /// that needs both.
    FoldBoundExceeded,
    /// The generator does not generate the whole multiplicative group, so a
    /// fold would not be the only exponent producing its image.
    GeneratorOrderNotFull,
    /// There is not one weight per row.
    RowWeightCountMismatch,
    /// There is not one weight per column.
    ColumnWeightCountMismatch,
}

/// The caller's `x_core` together with `com`.
///
/// F2Z verifies nothing upstream of this. The caller runs its own PIOP, and
/// establishes that its claim holds, that `q` is prime, and that the
/// coefficient factors as `v = v^(1) (x) v^(2)`. A claim whose coefficient
/// does not split that way cannot use this profile: the fold exponentiates
/// `v^(1)` and reconstructs over `v^(2)`, so it consumes the two separately.
/// This type re-checks only what it can see for itself.
///
/// `Q` is a const parameter rather than a profile constant because the caller
/// samples it: it draws a prime from a public set, projects its claim into
/// `F_q`, and hands the result to F2Z. Fixing it in the type keeps Barrett's
/// reciprocal a compile-time constant at the cost of one runtime check that
/// the shape agrees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreStatement<const Q: u128> {
    shape: Shape,
    generator: F128,
    row_weights: Vec<Fq<Q>>,
    column_weights: Vec<Fq<Q>>,
    target: Fq<Q>,
}

impl<const Q: u128> CoreStatement<Q> {
    /// Runs every check the statement owns and returns it.
    ///
    /// `row_weights` is `v^(1)`, one element per row; `column_weights` is
    /// `v^(2)`, one per column; `target` is the claimed value `mu`.
    pub fn new(
        shape: Shape,
        generator: F128,
        row_weights: Vec<Fq<Q>>,
        column_weights: Vec<Fq<Q>>,
        target: Fq<Q>,
    ) -> Result<Self, StatementError> {
        // No modulus gate here. `Fq<Q>` asserts that Q is an odd prime below
        // 2^126 on its own behalf, so naming the type is what enforces it and
        // an inadmissible modulus never reaches this constructor.

        // `Q < |K| / k_1`, equivalently `t + log2 Q < 128`. A fold is at most
        // `k_1 (Q - 1)` and the verifier only ever sees it modulo `ord(g)`, so
        // the accepted exponent is unique exactly while the whole range fits
        // inside the group. Written as a shift because `|K| = 2^128` does not
        // fit a u128; `t >= 7` keeps it in range.
        if Q >> (EXTENSION_BITS - shape.t() as u32) != 0 {
            return Err(StatementError::FoldBoundExceeded);
        }
        if !is_generator(generator) {
            return Err(StatementError::GeneratorOrderNotFull);
        }
        if row_weights.len() != shape.rows() {
            return Err(StatementError::RowWeightCountMismatch);
        }
        if column_weights.len() != shape.columns() {
            return Err(StatementError::ColumnWeightCountMismatch);
        }

        Ok(Self {
            shape,
            generator,
            row_weights,
            column_weights,
            target,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The largest value a column fold may take, `k_1 (Q - 1)`.
    ///
    /// The verifier requires every claimed fold to land at or below this.
    /// Checking it is not redundant with the gate in [`Self::new`]: the gate
    /// leaves an honest prover room, this bound is what holds a dishonest one
    /// to the same range.
    pub fn fold_bound(&self) -> u128 {
        // The gate put `k_1 (Q - 1)` strictly below `|K|`.
        (self.shape.rows() as u128) * (Q - 1)
    }

    pub fn generator(&self) -> F128 {
        self.generator
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
    ///
    /// At most `2^14` of these, so unlike the columns they are cheap to hold.
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

/// Absorbed as a typed value rather than hand-serialised at each call site.
///
/// Every field is fixed width, so distinct statements cannot encode alike and
/// no length prefix is needed. The caller has already domain-separated the
/// transcript and absorbed its own statement into it, so neither is repeated
/// here.
///
/// Note what is **not** here: the claim itself. Neither `v^(1)`, `v^(2)` nor
/// `mu` is encoded, so nothing below binds them; they reach the transcript
/// only through the caller's own events.
impl<const Q: u128> Encoding<[u8]> for CoreStatement<Q> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut frame = [0u8; 48];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            frame[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };

        put(&(self.shape.t() as u64).to_le_bytes());
        put(&(self.shape.s() as u64).to_le_bytes());
        put(&Q.to_le_bytes());
        put(&self.generator.to_bytes());
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::gf128::smallest_generator;
    use num_traits::ConstOne;

    /// The largest prime below `2^114`, the top of the sampling range.
    const Q114: u128 = (1 << 114) - 11;

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    /// A well-formed statement at an arbitrary shape, for gating tests.
    fn statement_at(shape: Shape) -> Result<CoreStatement<Q114>, StatementError> {
        CoreStatement::new(
            shape,
            smallest_generator(),
            vec![Fq::from(1u128); shape.rows()],
            vec![Fq::from(1u128); shape.columns()],
            Fq::from(0u128),
        )
    }

    fn statement(row_weights: Vec<Fq<Q114>>) -> Result<CoreStatement<Q114>, StatementError> {
        let shape = shape();
        CoreStatement::new(
            shape,
            smallest_generator(),
            row_weights,
            vec![Fq::from(1u128); shape.columns()],
            Fq::from(0u128),
        )
    }

    fn weights() -> Vec<Fq<Q114>> {
        (0..shape().rows())
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
    fn rejects_a_shape_the_modulus_is_too_large_for() {
        // `t = 14` is the widest row count this prime admits; 15 is not.
        assert!(statement_at(Shape::new(14, 21).unwrap()).is_ok());
        assert_eq!(
            statement_at(Shape::new(15, 20).unwrap()).err(),
            Some(StatementError::FoldBoundExceeded)
        );
    }

    #[test]
    fn rejects_a_generator_of_partial_order() {
        let shape = shape();
        assert_eq!(
            CoreStatement::<Q114>::new(
                shape,
                F128::ONE,
                weights(),
                vec![Fq::from(1u128); shape.columns()],
                Fq::from(0u128),
            ),
            Err(StatementError::GeneratorOrderNotFull)
        );
    }

    #[test]
    fn rejects_weight_vectors_that_do_not_fit_the_shape() {
        assert_eq!(
            statement(vec![Fq::from(1u128)]).err(),
            Some(StatementError::RowWeightCountMismatch)
        );

        let shape = shape();
        assert_eq!(
            CoreStatement::<Q114>::new(
                shape,
                smallest_generator(),
                weights(),
                vec![Fq::from(1u128)],
                Fq::from(0u128),
            )
            .err(),
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
