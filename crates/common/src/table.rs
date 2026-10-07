//! The committed bit table.

use crate::{
    Shape,
    matrix::{BitMatrix, Word},
};

/// A witness that does not match its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableError {
    /// The witness does not hold `2^m` bits.
    BitCountMismatch,
}

/// The committed bits `B[c][b]`, read through a shape.
///
/// A borrowed view over the prover's packed witness: at `m = 35` a byte per
/// bit would be 32 GiB. The index of a bit is `(c << t) | b` — column major,
/// then row — which is the order every coefficient vector in the protocol is
/// written in. Getting it wrong produces a valid-looking proof that fails only
/// at the final opening.
///
/// The packing is §2's `P`: `P[j * 2^(t-7) + i_hi] = sum_v f[(i_hi << 7) | v, j] * beta_v`
/// for the monomial basis `beta_v = X^v`. In that basis bit `v` of an `F128`'s
/// little-endian `lo || hi` *is* the coefficient of `X^v`, so the committed
/// form and the bit form are the same bytes, and the caller holds one copy.
#[derive(Debug, Clone, Copy)]
pub struct BitTable<'a> {
    shape: Shape,
    packed: &'a [Word],
}

impl<'a> BitTable<'a> {
    pub const BITS: usize = Word::BITS as usize;

    /// Wraps `packed` in `shape`, least significant bit first inside `lo`.
    ///
    /// Crate-private: [`crate::BitZParams::table`] is the only way in, so a
    /// table is always shaped by a checked parameter set.
    pub(crate) fn new(shape: Shape, packed: &'a [Word]) -> Result<Self, TableError> {
        // `m >= 22`, so the bit count is always a whole number of words.
        if packed.len() != (1 << shape.log_bits()) / BitTable::BITS {
            return Err(TableError::BitCountMismatch);
        }
        Ok(Self { shape, packed })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The bit `B[column][row]`.
    ///
    /// A `row` at or above `2^t` would alias into the next column rather than
    /// panicking, since the index is formed by `or` and not by addition. The
    /// assertion catches that in tests and costs nothing in release.
    pub fn bit(&self, column: usize, row: usize) -> bool {
        debug_assert!(row < self.shape.rows(), "row {row} is outside the column");
        debug_assert!(
            column < self.shape.columns(),
            "column {column} is outside the table"
        );

        let index = (column << self.shape.log_rows()) | row;
        // /% on compile time constant should be properly optimised away
        let element = self.packed[index / BitTable::BITS];
        let offset = index % BitTable::BITS;

        ((element >> offset) & 1) == 1
    }

    /// The `2^(t-7)` elements holding one column, in ascending row order.
    ///
    /// A column starts at bit `c * 2^t` and `t >= 7`, so it begins on an
    /// element boundary and spans whole elements. The fold walks these directly
    /// rather than calling [`BitTable::bit`] once per row: at `m = 35` that is
    /// `2^35` calls, each of them a bounds check and two shifts.
    pub fn column(&self, column: usize) -> &'a [Word] {
        let elements = self.shape.rows() / BitTable::BITS;
        &self.packed[column * elements..(column + 1) * elements]
    }

    /// Swaps rows and columns into an owned [`BitMatrix`] with `dim1` the
    /// rows and `dim2` the columns: `result.bit(b, c) == self.bit(c, b)`.
    ///
    /// Built for callers like a leaf construction that must walk the table
    /// row by row: the packing is column major, so a row-major walk over the
    /// table is a scatter, one `bit()` call and cache miss per cell. After
    /// the transpose each row is contiguous, read by [`BitMatrix::bits`]. See
    /// [`BitMatrix::transpose`] for the constraints; the table's own `t >= 7`
    /// is what the first one asks for.
    pub fn transpose(&self) -> BitMatrix {
        BitMatrix::transposed(self.packed, self.shape.log_rows())
    }
}

#[cfg(test)]
mod tests {
    use num_traits::ConstZero;

    // We want to use compare against
    use field::F128;

    /// Bits in a packed element.
    const PACKED_BITS: usize = 128;

    use super::*;

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn small_shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    /// Sets the rows named by `bits` as `(row, column)` in a zeroed witness.
    fn with_bits(shape: &Shape, bits: &[(usize, usize)]) -> Vec<F128> {
        let mut packed = vec![F128::ZERO; (1 << shape.log_bits()) / PACKED_BITS];
        for &(row, column) in bits {
            let index = (column << shape.log_rows()) | row;
            let element = &mut packed[index >> 7];
            let offset = index % PACKED_BITS;
            if offset < 64 {
                element.lo |= 1u64 << offset;
            } else {
                element.hi |= 1u64 << (offset - 64);
            }
        }
        packed
    }

    #[test]
    fn a_bit_lands_where_the_index_order_says_it_does() {
        let shape = small_shape();
        let set = [(0, 0), (2, 0), (3, 0), (65, 0), (1, 9)];
        let packed = with_bits(&shape, &set);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        for column in [0, 9, 10] {
            for row in 0..shape.rows() {
                assert_eq!(
                    table.bit(column, row),
                    set.contains(&(row, column)),
                    "B[{column}][{row}]"
                );
            }
        }
    }

    #[test]
    fn bit_v_of_each_element_is_row_v_of_its_group() {
        // Bit v of the element is the coefficient of X^v, so it must be the
        // row at offset v of the 128 that element covers. This is what lets
        // the committed form and the bit form be the same bytes.
        let shape = small_shape();
        let set = [(0, 0), (5, 0), (127, 0), (64, 3), (2, 9)];
        let packed = with_bits(&shape, &set);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        assert_eq!(packed.len(), 1 << shape.log_packed_len());

        let groups_per_column = shape.rows() / PACKED_BITS;
        for (index, element) in packed.iter().enumerate() {
            let column = index / groups_per_column;
            let i_hi = index % groups_per_column;
            let bits = u128::from(element.lo) | (u128::from(element.hi) << 64);
            for v in 0..PACKED_BITS {
                assert_eq!(
                    (bits >> v) & 1 == 1,
                    table.bit(column, (i_hi << 7) | v),
                    "P[{index}] bit {v}"
                );
            }
        }
    }

    #[test]
    fn rejects_a_witness_of_the_wrong_length() {
        let shape = small_shape();
        assert_eq!(
            BitTable::new(shape, bytemuck::cast_slice(&[F128::ZERO; 8])).err(),
            Some(TableError::BitCountMismatch)
        );
    }

    #[test]
    fn a_column_is_element_aligned_and_spans_whole_elements() {
        let shape = small_shape();
        let packed = with_bits(&shape, &[(0, 3), (2, 3), (64, 3)]);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        assert_eq!(table.column(3).len(), shape.rows() / BitTable::BITS);
        assert_eq!(table.column(3), [0b101 | (1 << 64)]);
        assert!(table.column(4).iter().all(|&element| element == 0));
    }

    #[test]
    fn the_column_elements_agree_with_the_bit_accessor() {
        let shape = small_shape();
        let rows: Vec<(usize, usize)> = (0..shape.rows()).step_by(7).map(|row| (row, 2)).collect();
        let packed = with_bits(&shape, &rows);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        for (index, &element) in table.column(2).iter().enumerate() {
            for offset in 0..BitTable::BITS {
                let row = index * BitTable::BITS + offset;
                assert_eq!(table.bit(2, row), (element >> offset) & 1 == 1, "row {row}");
            }
        }
    }

    /// The transpose has the table's rows along `dim1`, each one contiguous.
    #[test]
    fn the_transpose_has_the_rows_contiguous() {
        let shape = small_shape();
        let set = [(0, 0), (127, 0), (64, 3), (5, 9), (1, shape.columns() - 1)];
        let packed = with_bits(&shape, &set);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        let transposed = table.transpose();
        assert_eq!(transposed.dim1(), shape.rows());
        assert_eq!(transposed.dim2(), shape.columns());

        for row in [0, 1, 5, 64, 127] {
            let bits: Vec<bool> = transposed.bits(row).map(|b| b == 1).collect();
            let expected: Vec<bool> = (0..shape.columns())
                .map(|column| table.bit(column, row))
                .collect();
            assert_eq!(bits, expected, "row {row}");
        }
    }
}
