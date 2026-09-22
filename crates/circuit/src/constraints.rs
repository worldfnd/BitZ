//! Sparse constraint generation for the BitZ circuit language.
//!
//! The generated matrices follow Freigen's convention. `M` maps the Boolean
//! witness, prefixed by a constant one, to the integer witness. Its first row
//! is the implicit integer constant one. `A`, `B`, and `C` then encode the
//! rank-1 constraints `(A z) * (B z) = C z` over that integer witness. Every
//! integer coefficient is an arbitrary-precision signed [`BigInt`].

use num_bigint::BigInt;
use num_traits::{One, Zero};
use rayon::prelude::*;
use std::array;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::Sum;
use std::mem;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

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
    /// Constructs a sparse matrix from row-local `(column, coefficient)`
    /// entries.
    ///
    /// Every row must use strictly increasing in-bounds column indices.
    pub fn try_from_rows(
        columns: usize,
        rows: Vec<Vec<(usize, C)>>,
    ) -> Result<Self, SparseMatrixError> {
        for (row, entries) in rows.iter().enumerate() {
            let mut previous = None;
            for (column, _) in entries {
                if *column >= columns {
                    return Err(SparseMatrixError::ColumnOutOfBounds {
                        row,
                        column: *column,
                        columns,
                    });
                }
                if let Some(previous) = previous
                    && previous >= *column
                {
                    return Err(SparseMatrixError::ColumnsNotStrictlyIncreasing {
                        row,
                        previous,
                        column: *column,
                    });
                }
                previous = Some(*column);
            }
        }

        Ok(Self {
            rows: rows
                .into_iter()
                .map(|entries| SparseRow { entries })
                .collect(),
            columns,
        })
    }

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

impl<C: Send + Sync> SparseMatrix<C> {
    fn map_values_with<D, M>(self, map: M) -> SparseMatrix<D>
    where
        D: Send + Sync,
        M: Fn(C) -> D + Send + Sync,
    {
        SparseMatrix {
            rows: self
                .rows
                .into_par_iter()
                .map(|row| SparseRow {
                    entries: row
                        .entries
                        .into_iter()
                        .map(|(column, coefficient)| (column, map(coefficient)))
                        .collect(),
                })
                .collect(),
            columns: self.columns,
        }
    }
}

/// Why a sparse row representation is malformed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseMatrixError {
    ColumnOutOfBounds {
        row: usize,
        column: usize,
        columns: usize,
    },
    ColumnsNotStrictlyIncreasing {
        row: usize,
        previous: usize,
        column: usize,
    },
}

impl Display for SparseMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnOutOfBounds {
                row,
                column,
                columns,
            } => write!(
                formatter,
                "column {column} in row {row} is outside a {columns}-column matrix"
            ),
            Self::ColumnsNotStrictlyIncreasing {
                row,
                previous,
                column,
            } => write!(
                formatter,
                "columns in row {row} are not strictly increasing: {previous}, {column}"
            ),
        }
    }
}

impl Error for SparseMatrixError {}

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
    /// Constructs an F2 sparse matrix from row-local nonzero positions.
    ///
    /// Every row must use strictly increasing in-bounds column indices.
    pub fn try_from_rows(columns: usize, rows: Vec<Vec<usize>>) -> Result<Self, SparseMatrixError> {
        for (row, positions) in rows.iter().enumerate() {
            let mut previous = None;
            for column in positions {
                if *column >= columns {
                    return Err(SparseMatrixError::ColumnOutOfBounds {
                        row,
                        column: *column,
                        columns,
                    });
                }
                if let Some(previous) = previous
                    && previous >= *column
                {
                    return Err(SparseMatrixError::ColumnsNotStrictlyIncreasing {
                        row,
                        previous,
                        column: *column,
                    });
                }
                previous = Some(*column);
            }
        }

        Ok(Self {
            rows: rows
                .into_iter()
                .map(|positions| SparseBoolRow { positions })
                .collect(),
            columns,
        })
    }

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

/// The four sparse matrices generated for a BitZ circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstraintMatrices<C = BigInt> {
    /// Boolean-to-integer witness matrix.
    pub m: SparseBoolMatrix,
    /// Left R1CS matrix.
    pub a: SparseMatrix<C>,
    /// Right R1CS matrix.
    pub b: SparseMatrix<C>,
    /// Output R1CS matrix.
    pub c: SparseMatrix<C>,
}

/// Why the four constraint matrices cannot describe one R1CS relation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintMatrixShapeError {
    R1csRowCountMismatch { a: usize, b: usize, c: usize },
    R1csColumnCountMismatch { a: usize, b: usize, c: usize },
    AssignmentLengthMismatch { m_rows: usize, r1cs_columns: usize },
}

impl Display for ConstraintMatrixShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::R1csRowCountMismatch { a, b, c } => {
                write!(
                    formatter,
                    "A, B, and C have different row counts: {a}, {b}, {c}"
                )
            }
            Self::R1csColumnCountMismatch { a, b, c } => write!(
                formatter,
                "A, B, and C have different column counts: {a}, {b}, {c}"
            ),
            Self::AssignmentLengthMismatch {
                m_rows,
                r1cs_columns,
            } => write!(
                formatter,
                "M produces {m_rows} assignment entries but A, B, and C expect {r1cs_columns}"
            ),
        }
    }
}

impl Error for ConstraintMatrixShapeError {}

impl<C: Send + Sync> ConstraintMatrices<C> {
    /// Checks that A, B, and C share a shape and consume the assignment
    /// produced by M.
    pub fn validate_shape(&self) -> Result<(), ConstraintMatrixShapeError> {
        let a_rows = self.a.row_count();
        let b_rows = self.b.row_count();
        let c_rows = self.c.row_count();
        if a_rows != b_rows || a_rows != c_rows {
            return Err(ConstraintMatrixShapeError::R1csRowCountMismatch {
                a: a_rows,
                b: b_rows,
                c: c_rows,
            });
        }

        let a_columns = self.a.column_count();
        let b_columns = self.b.column_count();
        let c_columns = self.c.column_count();
        if a_columns != b_columns || a_columns != c_columns {
            return Err(ConstraintMatrixShapeError::R1csColumnCountMismatch {
                a: a_columns,
                b: b_columns,
                c: c_columns,
            });
        }

        let m_rows = self.m.row_count();
        if m_rows != a_columns {
            return Err(ConstraintMatrixShapeError::AssignmentLengthMismatch {
                m_rows,
                r1cs_columns: a_columns,
            });
        }

        Ok(())
    }

    /// Consumes the matrices and maps every A/B/C coefficient.
    ///
    /// The Boolean `M` matrix and sparse topology are moved unchanged.
    pub fn map_coefficients<D, M>(self, map: M) -> ConstraintMatrices<D>
    where
        D: Send + Sync,
        M: Fn(C) -> D + Send + Sync,
    {
        ConstraintMatrices {
            m: self.m,
            a: self.a.map_values_with(&map),
            b: self.b.map_values_with(&map),
            c: self.c.map_values_with(&map),
        }
    }
}

/// Why a Boolean witness does not satisfy a generated constraint system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SatisfactionError {
    /// The four matrices do not share a compatible R1CS shape.
    InvalidShape(ConstraintMatrixShapeError),
    /// The packed Boolean witness has the wrong number of entries.
    WitnessLength { expected: usize, actual: usize },
    /// The indicated R1CS row does not satisfy `a * b = c`.
    Constraint { row: usize },
}

impl Display for SatisfactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(error) => Display::fmt(error, formatter),
            Self::WitnessLength { expected, actual } => write!(
                formatter,
                "Boolean witness has length {actual}, expected {expected}"
            ),
            Self::Constraint { row } => write!(formatter, "R1CS row {row} is unsatisfied"),
        }
    }
}

impl Error for SatisfactionError {}

impl From<ConstraintMatrixShapeError> for SatisfactionError {
    fn from(error: ConstraintMatrixShapeError) -> Self {
        Self::InvalidShape(error)
    }
}

impl ConstraintMatrices<BigInt> {
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
        self.validate_shape()?;
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
    /// Witness indices, sorted and deduplicated.
    witnesses: Vec<usize>,
}

impl BoolLinearCombination {
    /// The Boolean constant term.
    pub const fn constant(&self) -> bool {
        self.constant
    }

    /// Sorted zero-based Boolean witness indices with coefficient one.
    pub fn witnesses(&self) -> &[usize] {
        &self.witnesses
    }

    pub(crate) fn witness(index: usize) -> Self {
        Self {
            constant: false,
            witnesses: vec![index],
        }
    }

    pub(crate) fn xor(mut self, rhs: Self) -> Self {
        self.constant ^= rhs.constant;
        self.witnesses = merge_sorted_vecs(
            self.witnesses,
            rhs.witnesses,
            |lhs, rhs| lhs.cmp(rhs),
            |_lhs, _rhs| None, // Drop overlaps
        );
        self
    }
}

impl From<bool> for BoolLinearCombination {
    fn from(constant: bool) -> Self {
        Self {
            constant,
            witnesses: Vec::new(),
        }
    }
}

impl BoolWitness for BoolLinearCombination {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
}

/// A symbolic integer linear combination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearCombination {
    constant: BigInt,
    /// Indices of witness elements with nonzero coefficients.
    /// Sorted and deduplicated by witness index.
    witnesses: Vec<(usize, BigInt)>,
}

impl LinearCombination {
    /// The integer constant term.
    pub fn constant(&self) -> &BigInt {
        &self.constant
    }

    /// Nonzero coefficients sorted by zero-based integer witness index.
    pub fn witnesses(&self) -> &[(usize, BigInt)] {
        &self.witnesses
    }

    fn witness(index: usize) -> Self {
        Self {
            constant: BigInt::zero(),
            witnesses: vec![(index, BigInt::one())],
        }
    }

    fn into_sparse_row(self) -> SparseRow<BigInt> {
        let Self {
            constant,
            mut witnesses,
        } = self;
        for (column, _) in &mut witnesses {
            *column += 1;
        }
        if !constant.is_zero() {
            witnesses.reserve_exact(1);
            witnesses.insert(0, (0, constant));
        }
        SparseRow { entries: witnesses }
    }
}

impl From<BigInt> for LinearCombination {
    fn from(constant: BigInt) -> Self {
        Self {
            constant,
            witnesses: Vec::new(),
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
        self += rhs;
        self
    }
}

impl AddAssign for LinearCombination {
    fn add_assign(&mut self, rhs: Self) {
        self.constant += rhs.constant;
        self.witnesses = merge_sorted_vecs(
            mem::take(&mut self.witnesses),
            rhs.witnesses,
            |(lhs_idx, _), (rhs_idx, _)| lhs_idx.cmp(rhs_idx),
            |(wit_idx, lhs_coeff), (_, rhs_coeff)| {
                let coeff = lhs_coeff + rhs_coeff;
                if coeff.is_zero() {
                    None
                } else {
                    Some((wit_idx, coeff))
                }
            },
        );
    }
}

impl Neg for LinearCombination {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.constant = -self.constant;
        for (_, coefficient) in &mut self.witnesses {
            *coefficient = -mem::take(coefficient);
        }
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
        if rhs.is_zero() {
            return Self::zero();
        }
        self.constant *= &rhs;
        for (_, coefficient) in &mut self.witnesses {
            *coefficient *= &rhs;
        }
        self
    }
}

impl Sum for LinearCombination {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

/// Merges two sorted vectors with no duplicates. Compares elements using `cmp` function,
/// and if two elements are equal, element produced by the `merge` function is added instead.
///
/// Concatenation will only reserve as much extra space as needed.
fn merge_sorted_vecs<T>(
    mut lhs: Vec<T>,
    mut rhs: Vec<T>,
    cmp: impl Fn(&T, &T) -> Ordering,
    merge: impl Fn(T, T) -> Option<T>,
) -> Vec<T> {
    let (Some(lhs_first), Some(lhs_last)) = (lhs.first(), lhs.last()) else {
        return rhs;
    };
    let (Some(rhs_first), Some(rhs_last)) = (rhs.first(), rhs.last()) else {
        return lhs;
    };
    // Disjoint ranges concatenate without a merge.
    if cmp(lhs_last, rhs_first).is_lt() {
        lhs.reserve_exact(rhs.len());
        lhs.extend(rhs);
        return lhs;
    }
    if cmp(rhs_last, lhs_first).is_lt() {
        rhs.reserve_exact(lhs.len());
        rhs.extend(lhs);
        return rhs;
    }
    let mut merged = Vec::with_capacity(lhs.len() + rhs.len());
    let mut lhs = lhs.into_iter().peekable();
    let mut rhs = rhs.into_iter().peekable();
    while let (Some(left), Some(right)) = (lhs.peek(), rhs.peek()) {
        match cmp(left, right) {
            Ordering::Less => merged.extend(lhs.next()),
            Ordering::Greater => merged.extend(rhs.next()),
            Ordering::Equal => {
                let left = lhs.next().expect("impossible");
                let right = rhs.next().expect("impossible");
                if let Some(new) = merge(left, right) {
                    merged.push(new);
                }
            }
        }
    }
    merged.extend(lhs);
    merged.extend(rhs);
    merged
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

    /// Records `value` as the next M row and returns its integer witness index.
    fn record_bitz(&mut self, value: BoolLinearCombination) -> usize {
        let witness = self.m_rows.len();
        self.m_rows.push(value);
        witness
    }

    /// Finishes generation and materializes the four sparse matrices.
    pub fn into_matrices(self) -> ConstraintMatrices<BigInt> {
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
    let BoolLinearCombination {
        constant,
        mut witnesses,
    } = value;
    for position in &mut witnesses {
        *position += 1;
    }
    if constant {
        witnesses.reserve_exact(1);
        witnesses.insert(0, 0);
    }
    SparseBoolRow {
        positions: witnesses,
    }
}

impl Circuit for ConstraintGenerator {
    type Bool = BoolLinearCombination;
    type Coefficient<const LIMBS: usize> = BigInt;
    type Z<const LIMBS: usize> = LinearCombination;

    fn xor(
        &mut self,
        lhs: BoolLinearCombination,
        rhs: BoolLinearCombination,
    ) -> BoolLinearCombination {
        lhs.xor(rhs)
    }

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

    fn bitz<const LIMBS: usize>(&mut self, value: BoolLinearCombination) -> LinearCombination {
        LinearCombination::witness(self.record_bitz(value))
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        bits_le: &ScalarBits<BoolLinearCombination, N>,
    ) -> (LinearCombination, LinearCombination) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        // Same layout as the default, one `bitz` per bit with coefficient
        // 2^index, but each sum is built once with exact capacity.
        let mut witnesses = Vec::with_capacity(N);
        for (index, bit) in bits_le.0.iter().enumerate() {
            witnesses.push((self.record_bitz(bit.clone()), BigInt::one() << index));
        }
        let low = LinearCombination {
            constant: BigInt::zero(),
            witnesses: witnesses[..LOW].to_vec(),
        };
        let full = LinearCombination {
            constant: BigInt::zero(),
            witnesses,
        };
        (full, low)
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

    fn terms(pairs: &[(usize, i64)]) -> Vec<(usize, BigInt)> {
        pairs
            .iter()
            .map(|&(witness, coefficient)| (witness, BigInt::from(coefficient)))
            .collect()
    }

    #[test]
    fn materializes_freigen_matrix_conventions_and_checks_witnesses() {
        let mut generator = ConstraintGenerator::new(2);
        let [x, y] = generator.inputs();
        let sum = generator.xor(x.clone(), y.clone());
        let z_sum = generator.bitz::<1>(sum);
        let z_x = generator.bitz::<1>(x);
        let z_y = generator.bitz::<1>(y);
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

    #[test]
    fn sparse_constructors_reject_unsorted_and_out_of_bounds_columns() {
        assert_eq!(
            SparseMatrix::<BigInt>::try_from_rows(
                3,
                vec![vec![(1, BigInt::one()), (1, BigInt::one())]],
            ),
            Err(SparseMatrixError::ColumnsNotStrictlyIncreasing {
                row: 0,
                previous: 1,
                column: 1,
            })
        );
        assert_eq!(
            SparseBoolMatrix::try_from_rows(2, vec![vec![2]]),
            Err(SparseMatrixError::ColumnOutOfBounds {
                row: 0,
                column: 2,
                columns: 2,
            })
        );
    }

    #[test]
    fn malformed_constraint_shape_returns_an_error_instead_of_panicking() {
        let matrices = ConstraintMatrices {
            m: SparseBoolMatrix::try_from_rows(1, vec![vec![0]]).unwrap(),
            a: SparseMatrix::try_from_rows(1, vec![vec![(0, BigInt::one())]]).unwrap(),
            b: SparseMatrix::try_from_rows(1, Vec::new()).unwrap(),
            c: SparseMatrix::try_from_rows(1, vec![vec![(0, BigInt::one())]]).unwrap(),
        };

        assert_eq!(
            matrices.check_witness(&PackedWitness::from_bits(&[])),
            Err(SatisfactionError::InvalidShape(
                ConstraintMatrixShapeError::R1csRowCountMismatch { a: 1, b: 0, c: 1 }
            ))
        );
    }

    #[test]
    fn linear_combination_terms_stay_sorted_and_merge_shared_witnesses() {
        let term = |index: usize, coefficient: i64| {
            LinearCombination::witness(index) * BigInt::from(coefficient)
        };
        let exact = |value: &LinearCombination| value.witnesses.capacity() == value.witnesses.len();

        // Disjoint ranges append on either side without spare capacity.
        let ascending = term(1, 1) + term(3, 1);
        assert_eq!(ascending.witnesses(), terms(&[(1, 1), (3, 1)]));
        assert!(exact(&ascending));
        let descending = term(3, 1) + term(1, 1);
        assert_eq!(descending.witnesses(), terms(&[(1, 1), (3, 1)]));
        assert!(exact(&descending));

        // Interleaved ranges merge, shared witnesses sum, cancellations vanish.
        let interleaved = (term(0, 1) + term(2, 1)) + (term(1, 1) + term(3, 1) + term(2, 3));
        assert_eq!(
            interleaved.witnesses(),
            terms(&[(0, 1), (1, 1), (2, 4), (3, 1)])
        );
        let cancelled = (term(1, 1) + term(2, 1)) - term(1, 1);
        assert_eq!(cancelled.witnesses(), terms(&[(2, 1)]));
        assert!((term(1, 2) - term(1, 2)).is_zero());
        assert!((term(5, 7) * BigInt::zero()).is_zero());

        let scaled = -(term(1, 2) + LinearCombination::from(BigInt::from(3))) * BigInt::from(5);
        assert_eq!(*scaled.constant(), BigInt::from(-15));
        assert_eq!(scaled.witnesses(), terms(&[(1, -10)]));
    }

    #[test]
    fn bitz_unsigned_lifts_bits_with_powers_of_two_into_exact_rows() {
        let mut generator = ConstraintGenerator::new(3);
        let bits = ScalarBits(generator.inputs::<3>());
        let (full, low) = generator.bitz_unsigned::<1, 3, 1, 2>(&bits);

        assert_eq!(generator.m_rows.len(), 3);
        assert_eq!(generator.m_rows[2].witnesses(), &[2]);
        assert!(full.constant().is_zero());
        assert_eq!(full.witnesses(), terms(&[(0, 1), (1, 2), (2, 4)]));
        assert_eq!(full.witnesses.capacity(), 3);
        assert_eq!(low.witnesses(), terms(&[(0, 1), (1, 2)]));
        assert_eq!(low.witnesses.capacity(), 2);
    }

    #[test]
    fn bool_linear_combination_xor_is_a_sorted_symmetric_difference() {
        let bit = BoolLinearCombination::witness;

        let ascending = bit(1).xor(bit(3));
        assert_eq!(ascending.witnesses(), &[1, 3]);
        assert_eq!(bit(3).xor(bit(1)).witnesses(), &[1, 3]);
        assert_eq!(
            ascending.clone().xor(bit(3).xor(bit(5))).witnesses(),
            &[1, 5]
        );
        assert_eq!(
            bit(0).xor(bit(2)).xor(bit(1).xor(bit(3))).witnesses(),
            &[0, 1, 2, 3]
        );

        let cancelled = ascending
            .clone()
            .xor(ascending)
            .xor(BoolLinearCombination::from(true));
        assert!(cancelled.witnesses().is_empty());
        assert!(cancelled.constant());
    }
}
