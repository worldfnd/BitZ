//! Sparse constraint generation for the F2Z circuit language.
//!
//! The generated matrices follow Freigen's convention. `M` maps the Boolean
//! witness, prefixed by a constant one, to the integer witness. Its first row
//! is the implicit integer constant one. `A`, `B`, and `C` then encode the
//! rank-1 constraints `(A z) * (B z) = C z` over that integer witness. Every
//! integer coefficient is an arbitrary-precision signed [`BigInt`].

use std::array;
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::witgen::PackedWitness;
use crate::{BoolWitness, Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

/// One row of a sparse matrix, sorted by increasing column index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseRow<C> {
    entries: Vec<(usize, C)>,
}

impl<C> SparseRow<C> {
    /// Nonzero `(column, coefficient)` entries in increasing column order.
    pub fn entries(&self) -> &[(usize, C)] {
        &self.entries
    }
}

/// A row-major sparse matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseMatrix<C> {
    rows: Vec<SparseRow<C>>,
    columns: usize,
}

impl<C> SparseMatrix<C> {
    /// Matrix rows.
    pub fn rows(&self) -> &[SparseRow<C>] {
        &self.rows
    }

    /// Number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns, including the constant column zero.
    pub const fn column_count(&self) -> usize {
        self.columns
    }
}

/// One sparse F2 row, represented solely by its nonzero column positions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseBoolRow {
    positions: Vec<usize>,
}

impl SparseBoolRow {
    /// Nonzero column positions in increasing order.
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }
}

/// A row-major sparse matrix over F2.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseBoolMatrix {
    rows: Vec<SparseBoolRow>,
    columns: usize,
}

impl SparseBoolMatrix {
    /// Matrix rows.
    pub fn rows(&self) -> &[SparseBoolRow] {
        &self.rows
    }

    /// Number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns, including the constant column zero.
    pub const fn column_count(&self) -> usize {
        self.columns
    }
}

/// The four sparse matrices generated for an F2Z circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintMatrices {
    /// Boolean-to-integer witness matrix.
    pub m: SparseBoolMatrix,
    /// Left R1CS matrix.
    pub a: SparseMatrix<BigInt>,
    /// Right R1CS matrix.
    pub b: SparseMatrix<BigInt>,
    /// Output R1CS matrix.
    pub c: SparseMatrix<BigInt>,
}

/// Why a Boolean witness does not satisfy a generated constraint system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SatisfactionError {
    /// The packed Boolean witness has the wrong number of entries.
    WitnessLength { expected: usize, actual: usize },
    /// The indicated R1CS row does not satisfy `a * b = c`.
    Constraint { row: usize },
}

impl Display for SatisfactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WitnessLength { expected, actual } => write!(
                formatter,
                "Boolean witness has length {actual}, expected {expected}"
            ),
            Self::Constraint { row } => write!(formatter, "R1CS row {row} is unsatisfied"),
        }
    }
}

impl Error for SatisfactionError {}

impl ConstraintMatrices {
    /// Applies `M` to a packed Boolean witness.
    ///
    /// The returned vector starts with the implicit constant one and is the
    /// witness consumed by `A`, `B`, and `C`.
    pub fn integer_witness(
        &self,
        witness: &PackedWitness,
    ) -> Result<Vec<BigInt>, SatisfactionError> {
        let expected = self.m.column_count().saturating_sub(1);
        if witness.bit_len() != expected {
            return Err(SatisfactionError::WitnessLength {
                expected,
                actual: witness.bit_len(),
            });
        }

        Ok(self
            .m
            .rows()
            .iter()
            .map(|row| {
                let value = row.positions().iter().fold(false, |value, column| {
                    value
                        ^ if *column == 0 {
                            true
                        } else {
                            witness.bit(column - 1)
                        }
                });
                BigInt::from(value)
            })
            .collect())
    }

    /// Checks every materialized R1CS row against a packed Boolean witness.
    pub fn check_witness(&self, witness: &PackedWitness) -> Result<(), SatisfactionError> {
        let integer_witness = self.integer_witness(witness)?;

        for row in 0..self.a.row_count() {
            let a = evaluate_integer_row(&self.a.rows[row], &integer_witness);
            let b = evaluate_integer_row(&self.b.rows[row], &integer_witness);
            let c = evaluate_integer_row(&self.c.rows[row], &integer_witness);
            if a * b != c {
                return Err(SatisfactionError::Constraint { row });
            }
        }

        Ok(())
    }

    /// Whether every materialized row is satisfied by the witness.
    pub fn is_satisfied(&self, witness: &PackedWitness) -> bool {
        self.check_witness(witness).is_ok()
    }
}

fn evaluate_integer_row(row: &SparseRow<BigInt>, witness: &[BigInt]) -> BigInt {
    row.entries()
        .iter()
        .map(|(column, coefficient)| witness[*column].clone() * coefficient.clone())
        .sum()
}

/// A symbolic linear combination over F2.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BoolLinearCombination {
    constant: bool,
    witnesses: BTreeSet<usize>,
}

impl BoolLinearCombination {
    /// The Boolean constant term.
    pub const fn constant(&self) -> bool {
        self.constant
    }

    /// Zero-based Boolean witness indices with coefficient one.
    pub fn witnesses(&self) -> &BTreeSet<usize> {
        &self.witnesses
    }

    fn witness(index: usize) -> Self {
        Self {
            constant: false,
            witnesses: BTreeSet::from([index]),
        }
    }
}

impl From<bool> for BoolLinearCombination {
    fn from(constant: bool) -> Self {
        Self {
            constant,
            witnesses: BTreeSet::new(),
        }
    }
}

impl BoolWitness for BoolLinearCombination {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;

    fn xor(mut self, rhs: Self) -> Self {
        self.constant ^= rhs.constant;
        for witness in rhs.witnesses {
            if !self.witnesses.insert(witness) {
                self.witnesses.remove(&witness);
            }
        }
        self
    }
}

/// A symbolic integer linear combination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearCombination {
    constant: BigInt,
    witnesses: BTreeMap<usize, BigInt>,
}

impl LinearCombination {
    /// The integer constant term.
    pub fn constant(&self) -> &BigInt {
        &self.constant
    }

    /// Nonzero coefficients keyed by zero-based integer witness index.
    pub fn witnesses(&self) -> &BTreeMap<usize, BigInt> {
        &self.witnesses
    }

    fn witness(index: usize) -> Self {
        Self {
            constant: BigInt::zero(),
            witnesses: BTreeMap::from([(index, BigInt::one())]),
        }
    }

    fn add_term(&mut self, index: usize, coefficient: BigInt) {
        if coefficient.is_zero() {
            return;
        }
        match self.witnesses.entry(index) {
            Entry::Vacant(entry) => {
                entry.insert(coefficient);
            }
            Entry::Occupied(mut entry) => {
                *entry.get_mut() += coefficient;
                if entry.get().is_zero() {
                    entry.remove();
                }
            }
        }
    }

    fn into_sparse_row(self) -> SparseRow<BigInt> {
        let mut entries =
            Vec::with_capacity(self.witnesses.len() + usize::from(!self.constant.is_zero()));
        if !self.constant.is_zero() {
            entries.push((0, self.constant));
        }
        entries.extend(
            self.witnesses
                .into_iter()
                .map(|(witness, coefficient)| (witness + 1, coefficient)),
        );
        SparseRow { entries }
    }
}

impl From<BigInt> for LinearCombination {
    fn from(constant: BigInt) -> Self {
        Self {
            constant,
            witnesses: BTreeMap::new(),
        }
    }
}

impl Zero for LinearCombination {
    fn zero() -> Self {
        Self::from(BigInt::zero())
    }

    fn is_zero(&self) -> bool {
        self.constant.is_zero() && self.witnesses.is_empty()
    }
}

impl Add for LinearCombination {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.constant += rhs.constant;
        for (witness, coefficient) in rhs.witnesses {
            self.add_term(witness, coefficient);
        }
        self
    }
}

impl AddAssign for LinearCombination {
    fn add_assign(&mut self, rhs: Self) {
        self.constant += rhs.constant;
        for (witness, coefficient) in rhs.witnesses {
            self.add_term(witness, coefficient);
        }
    }
}

impl Neg for LinearCombination {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.constant = -self.constant;
        self.witnesses = self
            .witnesses
            .into_iter()
            .map(|(witness, coefficient)| (witness, -coefficient))
            .collect();
        self
    }
}

impl Sub for LinearCombination {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl SubAssign for LinearCombination {
    fn sub_assign(&mut self, rhs: Self) {
        *self += -rhs;
    }
}

impl Mul<BigInt> for LinearCombination {
    type Output = Self;

    fn mul(mut self, rhs: BigInt) -> Self::Output {
        self.constant *= rhs.clone();
        self.witnesses = self
            .witnesses
            .into_iter()
            .filter_map(|(witness, coefficient)| {
                let coefficient = coefficient * rhs.clone();
                (!coefficient.is_zero()).then_some((witness, coefficient))
            })
            .collect();
        self
    }
}

impl Sum for LinearCombination {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

/// Circuit backend that records sparse M/A/B/C matrices without evaluating hints.
#[derive(Clone, Debug)]
pub struct ConstraintGenerator {
    input_witnesses: usize,
    next_boolean_witness: usize,
    m_rows: Vec<BoolLinearCombination>,
    r1cs: Vec<(LinearCombination, LinearCombination, LinearCombination)>,
}

impl ConstraintGenerator {
    /// Starts a generator with `input_witnesses` preallocated Boolean inputs.
    pub const fn new(input_witnesses: usize) -> Self {
        Self {
            input_witnesses,
            next_boolean_witness: input_witnesses,
            m_rows: Vec::new(),
            r1cs: Vec::new(),
        }
    }

    /// Returns one of the preallocated input witnesses.
    pub fn input(&self, index: usize) -> BoolLinearCombination {
        assert!(
            index < self.input_witnesses,
            "input witness is not allocated"
        );
        BoolLinearCombination::witness(index)
    }

    /// Returns all preallocated inputs as an array.
    pub fn inputs<const N: usize>(&self) -> [BoolLinearCombination; N] {
        assert_eq!(N, self.input_witnesses, "input witness count mismatch");
        array::from_fn(BoolLinearCombination::witness)
    }

    /// Returns every preallocated input in one heap allocation.
    ///
    /// Large gadgets should use this instead of constructing a large symbolic
    /// array on the stack merely to pass it by reference.
    pub fn boxed_inputs<const N: usize>(&self) -> Box<[BoolLinearCombination; N]> {
        assert_eq!(N, self.input_witnesses, "input witness count mismatch");
        let inputs: Box<[BoolLinearCombination]> = (0..self.input_witnesses)
            .map(BoolLinearCombination::witness)
            .collect();
        match inputs.try_into() {
            Ok(inputs) => inputs,
            Err(_) => unreachable!("boxed input length was checked"),
        }
    }

    /// Finishes generation and materializes the four sparse matrices.
    pub fn into_matrices(self) -> ConstraintMatrices {
        let Self {
            input_witnesses: _,
            next_boolean_witness,
            m_rows,
            r1cs,
        } = self;
        let integer_columns = m_rows.len() + 1;

        let mut materialized_m = Vec::with_capacity(integer_columns);
        materialized_m.push(SparseBoolRow { positions: vec![0] });
        materialized_m.extend(m_rows.into_iter().map(bool_sparse_row));

        let mut a = Vec::with_capacity(r1cs.len());
        let mut b = Vec::with_capacity(r1cs.len());
        let mut c = Vec::with_capacity(r1cs.len());
        for (left, right, output) in r1cs {
            a.push(left.into_sparse_row());
            b.push(right.into_sparse_row());
            c.push(output.into_sparse_row());
        }

        ConstraintMatrices {
            m: SparseBoolMatrix {
                rows: materialized_m,
                columns: next_boolean_witness + 1,
            },
            a: SparseMatrix {
                rows: a,
                columns: integer_columns,
            },
            b: SparseMatrix {
                rows: b,
                columns: integer_columns,
            },
            c: SparseMatrix {
                rows: c,
                columns: integer_columns,
            },
        }
    }
}

fn bool_sparse_row(value: BoolLinearCombination) -> SparseBoolRow {
    let mut positions = Vec::with_capacity(value.witnesses.len() + usize::from(value.constant));
    if value.constant {
        positions.push(0);
    }
    positions.extend(value.witnesses.into_iter().map(|witness| witness + 1));
    SparseBoolRow { positions }
}

impl Circuit for ConstraintGenerator {
    type Bool = BoolLinearCombination;
    type Coefficient<const LIMBS: usize> = BigInt;
    type Z<const LIMBS: usize> = LinearCombination;

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<BoolLinearCombination, N>
    where
        H: Fn(
                &dyn WitnessContext<LinearCombination, BoolLinearCombination, BigInt>,
            ) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        let first = self.next_boolean_witness;
        self.next_boolean_witness = self
            .next_boolean_witness
            .checked_add(N)
            .expect("Boolean witness count overflow");
        ScalarBits(array::from_fn(|index| {
            BoolLinearCombination::witness(first + index)
        }))
    }

    fn f2z<const LIMBS: usize>(&mut self, value: BoolLinearCombination) -> LinearCombination {
        let witness = self.m_rows.len();
        self.m_rows.push(value);
        LinearCombination::witness(witness)
    }

    fn assert_r1c<const LIMBS: usize>(
        &mut self,
        a: LinearCombination,
        b: LinearCombination,
        c: LinearCombination,
    ) {
        self.r1cs.push((a, b, c));
    }

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: LinearCombination,
    ) -> LinearCombination {
        assert!(
            TO_LIMBS >= FROM_LIMBS,
            "cannot sign-extend into fewer limbs"
        );
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witgen::Witgen;

    #[test]
    fn materializes_freigen_matrix_conventions_and_checks_witnesses() {
        let mut generator = ConstraintGenerator::new(2);
        let [x, y] = generator.inputs();
        let sum = x.clone().xor(y.clone());
        let z_sum = generator.f2z::<1>(sum);
        let z_x = generator.f2z::<1>(x);
        let z_y = generator.f2z::<1>(y);
        generator.assert_r1c::<1>(
            z_x.clone() * BigInt::from(2),
            z_y.clone(),
            z_x + z_y - z_sum,
        );
        let mut matrices = generator.into_matrices();

        assert_eq!(matrices.m.row_count(), 4);
        assert_eq!(matrices.m.column_count(), 3);
        assert_eq!(matrices.a.row_count(), 1);
        assert_eq!(matrices.a.column_count(), 4);
        assert_eq!(matrices.m.rows()[0].positions(), &[0]);
        assert_eq!(matrices.m.rows()[1].positions(), &[1, 2]);
        assert_eq!(matrices.a.rows()[0].entries(), &[(2, BigInt::from(2))]);
        assert_eq!(matrices.b.rows()[0].entries(), &[(3, BigInt::from(1))]);
        assert_eq!(
            matrices.c.rows()[0].entries(),
            &[
                (1, BigInt::from(-1)),
                (2, BigInt::from(1)),
                (3, BigInt::from(1)),
            ]
        );

        let satisfying = Witgen::with_inputs(&[true, false]);
        // This circuit has no hints, so its packed witness consists of inputs.
        assert!(matrices.is_satisfied(satisfying.witness()));

        matrices.c.rows[0].entries.push((0, BigInt::one()));
        assert_eq!(
            matrices.check_witness(satisfying.witness()),
            Err(SatisfactionError::Constraint { row: 0 })
        );
    }

    #[test]
    fn coefficients_are_arbitrary_precision_integers() {
        let huge = BigInt::one() << 512_usize;
        let mut generator = ConstraintGenerator::new(0);
        generator.assert_r1c::<1>(
            LinearCombination::from(BigInt::one()),
            LinearCombination::from(huge.clone()),
            LinearCombination::from(huge.clone()),
        );
        let matrices = generator.into_matrices();

        assert_eq!(matrices.b.rows()[0].entries(), &[(0, huge.clone())]);
        assert_eq!(matrices.c.rows()[0].entries(), &[(0, huge)]);
        assert!(matrices.is_satisfied(Witgen::with_inputs(&[]).witness()));
    }
}
