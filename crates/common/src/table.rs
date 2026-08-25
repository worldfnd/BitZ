//! The committed bit table.

use crate::Shape;
use field::F128;

/// Bits in a word, and the shift that divides an index into word and offset.
pub(crate) const WORD_BITS: usize = u64::BITS as usize;
const WORD_SHIFT: u32 = u64::BITS.trailing_zeros();

/// A witness that does not match its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableError {
    /// The witness does not hold `2^m` bits.
    BitCountMismatch,
}

/// The committed bits `B[c][b]`, read through a shape.
///
/// A borrowed view over the prover's witness, packed 64 bits to a word: at
/// `m = 35` a byte per bit would be 32 GiB. The index of a bit is
/// `(c << t) | b` — column major, then row — which is the order every
/// coefficient vector in the protocol is written in. Getting it wrong
/// produces a valid-looking proof that fails only at the final opening.
#[derive(Debug, Clone, Copy)]
pub struct BitTable<'a> {
    shape: Shape,
    words: &'a [u64],
}

impl<'a> BitTable<'a> {
    /// Wraps `words` in `shape`, least significant bit first inside a word.
    pub fn new(shape: Shape, words: &'a [u64]) -> Result<Self, TableError> {
        // `m >= 22`, so the bit count is always a whole number of words.
        if words.len() != (1 << shape.m()) / WORD_BITS {
            return Err(TableError::BitCountMismatch);
        }
        Ok(Self { shape, words })
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
        let index = (column << self.shape.t()) | row;
        (self.words[index >> WORD_SHIFT] >> (index % WORD_BITS)) & 1 == 1
    }

    /// The packed field elements `P`, one per 128 consecutive bits of a
    /// column.
    ///
    /// `P[j * 2^(t-7) + i_hi] = sum_v f[(i_hi << 7) | v, j] * beta_v` for the
    /// monomial basis `beta_v = X^v`. In that basis bit `v` of an `F128`'s
    /// little-endian `lo || hi` *is* the coefficient of `X^v`, so this is a
    /// reinterpretation of the witness words rather than an evaluation.
    ///
    /// `t >= 7` makes a column `2^t >= 128` bits and a multiple of 128, so no
    /// group of 128 straddles a column boundary; the index order `(c << t) | b`
    /// then puts `P` in exactly the order above, column major.
    pub fn pack(&self) -> Vec<F128> {
        self.words
            .chunks_exact(2)
            .map(|pair| F128::new(pair[0], pair[1]))
            .collect()
    }

    /// The `2^t / 64` words holding one column, in ascending row order.
    ///
    /// A column starts at bit `c * 2^t` and `t >= 7`, so it begins on a word
    /// boundary and spans whole words. The fold walks these directly rather
    /// than calling [`BitTable::bit`] once per row: at `m = 35` that is `2^35`
    /// calls, each of them a bounds check and two shifts.
    pub fn column(&self, column: usize) -> &'a [u64] {
        let words = self.shape.rows() / WORD_BITS;
        &self.words[column * words..(column + 1) * words]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn small_shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    /// Sets the rows named by `bits` as `(row, column)` in a zeroed witness.
    fn with_bits(shape: &Shape, bits: &[(usize, usize)]) -> Vec<u64> {
        let mut words = vec![0u64; (1 << shape.m()) / WORD_BITS];
        for &(row, column) in bits {
            let index = (column << shape.t()) | row;
            words[index >> WORD_SHIFT] |= 1u64 << (index % WORD_BITS);
        }
        words
    }

    #[test]
    fn a_bit_lands_where_the_index_order_says_it_does() {
        let shape = small_shape();
        let set = [(0, 0), (2, 0), (3, 0), (65, 0), (1, 9)];
        let words = with_bits(&shape, &set);
        let table = BitTable::new(shape, &words).unwrap();

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
    fn packing_puts_bit_v_of_each_element_at_row_v_of_its_group() {
        let shape = small_shape();
        let set = [(0, 0), (5, 0), (127, 0), (64, 3), (2, 9)];
        let words = with_bits(&shape, &set);
        let table = BitTable::new(shape, &words).unwrap();

        let packed = table.pack();
        assert_eq!(packed.len(), 1 << shape.packed_m());

        // Bit v of the element is the coefficient of X^v, so it must be the
        // row at offset v of the 128 that element covers.
        let groups_per_column = shape.rows() / 128;
        for (index, element) in packed.iter().enumerate() {
            let column = index / groups_per_column;
            let i_hi = index % groups_per_column;
            let bits = u128::from(element.lo) | (u128::from(element.hi) << 64);
            for v in 0..128 {
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
            BitTable::new(shape, &[0u64; 8]).err(),
            Some(TableError::BitCountMismatch)
        );
    }

    #[test]
    fn a_column_is_word_aligned_and_spans_whole_words() {
        let shape = small_shape();
        let words = with_bits(&shape, &[(0, 3), (2, 3), (64, 3)]);
        let table = BitTable::new(shape, &words).unwrap();

        assert_eq!(table.column(3).len(), shape.rows() / WORD_BITS);
        assert_eq!(table.column(3), [0b101, 1]);
        assert!(table.column(4).iter().all(|&word| word == 0));
    }

    #[test]
    fn the_column_words_agree_with_the_bit_accessor() {
        let shape = small_shape();
        let rows: Vec<(usize, usize)> = (0..shape.rows()).step_by(7).map(|row| (row, 2)).collect();
        let words = with_bits(&shape, &rows);
        let table = BitTable::new(shape, &words).unwrap();

        for (index, &word) in table.column(2).iter().enumerate() {
            for offset in 0..WORD_BITS {
                let row = index * WORD_BITS + offset;
                assert_eq!(table.bit(2, row), (word >> offset) & 1 == 1, "row {row}");
            }
        }
    }
}
