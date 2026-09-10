//! The committed bit table.

use crate::Shape;
use field::F128;

/// Bits in a packed element, and the shift that divides an index into element
/// and offset.
pub(crate) const PACKED_BITS: usize = 128;
const PACKED_SHIFT: u32 = 7;
/// Bits in each half of a packed element.
const HALF_BITS: usize = 64;

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
    packed: &'a [F128],
}

impl<'a> BitTable<'a> {
    /// Wraps `packed` in `shape`, least significant bit first inside `lo`.
    ///
    /// Crate-private: [`crate::BitZParams::table`] is the only way in, so a
    /// table is always shaped by a checked parameter set.
    pub(crate) fn new(shape: Shape, packed: &'a [F128]) -> Result<Self, TableError> {
        // `m >= 22`, so the bit count is always a whole number of elements.
        if packed.len() != (1 << shape.log_bits()) / PACKED_BITS {
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
        let element = self.packed[index >> PACKED_SHIFT];
        let offset = index % PACKED_BITS;
        let half = if offset < HALF_BITS {
            element.lo
        } else {
            element.hi
        };
        (half >> (offset % HALF_BITS)) & 1 == 1
    }

    /// The `2^(t-7)` elements holding one column, in ascending row order.
    ///
    /// A column starts at bit `c * 2^t` and `t >= 7`, so it begins on an
    /// element boundary and spans whole elements. The fold walks these directly
    /// rather than calling [`BitTable::bit`] once per row: at `m = 35` that is
    /// `2^35` calls, each of them a bounds check and two shifts.
    pub fn column(&self, column: usize) -> &'a [F128] {
        let elements = self.shape.rows() / PACKED_BITS;
        &self.packed[column * elements..(column + 1) * elements]
    }

    /// The column's bits, in ascending row order.
    pub fn column_bits(&self, column: usize) -> ColumnBits<'a> {
        ColumnBits {
            elements: self.column(column).iter(),
            word: 0,
            hi_pending: None,
            remaining: 0,
        }
    }
}

/// Iterator over one column's bits, in ascending row order — see
/// [`BitTable::column_bits`].
pub struct ColumnBits<'a> {
    elements: std::slice::Iter<'a, F128>,
    /// Bits not yet emitted; the next one to emit is the LSB.
    word: u64,
    /// `hi` of the current element, once `lo` has been fully shifted out of
    /// `word` but before it takes `word`'s place.
    hi_pending: Option<u64>,
    /// How many low bits of `word` are still valid, i.e. still unemitted.
    remaining: u32,
}

impl Iterator for ColumnBits<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        loop {
            if self.remaining > 0 {
                let bit = self.word & 1 == 1;
                self.word >>= 1;
                self.remaining -= 1;
                return Some(bit);
            }
            if let Some(hi) = self.hi_pending.take() {
                self.word = hi;
                self.remaining = HALF_BITS as u32;
            } else {
                let element = self.elements.next()?;
                self.word = element.lo;
                self.hi_pending = Some(element.hi);
                self.remaining = HALF_BITS as u32;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for ColumnBits<'_> {
    fn len(&self) -> usize {
        self.remaining as usize
            + self.hi_pending.map_or(0, |_| HALF_BITS)
            + self.elements.len() * PACKED_BITS
    }
}

#[cfg(test)]
mod tests {
    use num_traits::ConstZero;

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
        let table = BitTable::new(shape, &packed).unwrap();

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
        let table = BitTable::new(shape, &packed).unwrap();

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
            BitTable::new(shape, &[F128::ZERO; 8]).err(),
            Some(TableError::BitCountMismatch)
        );
    }

    #[test]
    fn a_column_is_element_aligned_and_spans_whole_elements() {
        let shape = small_shape();
        let packed = with_bits(&shape, &[(0, 3), (2, 3), (64, 3)]);
        let table = BitTable::new(shape, &packed).unwrap();

        assert_eq!(table.column(3).len(), shape.rows() / PACKED_BITS);
        assert_eq!(table.column(3), [F128::new(0b101, 1)]);
        assert!(table.column(4).iter().all(|element| *element == F128::ZERO));
    }

    #[test]
    fn column_bits_agrees_with_the_bit_accessor() {
        let shape = small_shape();
        let rows: Vec<(usize, usize)> = (0..shape.rows()).step_by(7).map(|row| (row, 2)).collect();
        let packed = with_bits(&shape, &rows);
        let table = BitTable::new(shape, &packed).unwrap();

        let iter = table.column_bits(2);
        assert_eq!(iter.len(), shape.rows());
        let bits: Vec<bool> = iter.collect();
        assert_eq!(bits.len(), shape.rows());
        for (row, bit) in bits.into_iter().enumerate() {
            assert_eq!(bit, table.bit(2, row), "row {row}");
        }
    }

    #[test]
    fn the_column_elements_agree_with_the_bit_accessor() {
        let shape = small_shape();
        let rows: Vec<(usize, usize)> = (0..shape.rows()).step_by(7).map(|row| (row, 2)).collect();
        let packed = with_bits(&shape, &rows);
        let table = BitTable::new(shape, &packed).unwrap();

        for (index, element) in table.column(2).iter().enumerate() {
            let bits = u128::from(element.lo) | (u128::from(element.hi) << 64);
            for offset in 0..PACKED_BITS {
                let row = index * PACKED_BITS + offset;
                assert_eq!(table.bit(2, row), (bits >> offset) & 1 == 1, "row {row}");
            }
        }
    }
}
