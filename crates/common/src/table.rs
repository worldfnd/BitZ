//! The committed bit table.

use std::slice::Iter;

use crate::{Shape, ShapeError};
/// The machine word `packed` is sliced into. [`BitTable::BITS`] is derived
/// from this, so changing it is the only step needed to repack into a
/// different word size.
type Word = u64;

/// A witness that does not match its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableError {
    /// The witness does not hold `2^m` bits.
    BitCountMismatch,
}

/// [`BitTable::transpose`] cannot lay out its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeError {
    /// The source has fewer than `2^7 = 128` columns.
    ///
    /// Transposing swaps rows and columns, so the result's row count is the
    /// source's column count -- and every `BitTable`'s row count must be a
    /// whole number of `128`-bit packed elements (the same rule
    /// `ShapeError::RowIndexTooNarrow` enforces when a shape is built).
    /// `Shape::new` only enforces that floor on `log_rows`, never on
    /// `log_columns`, so a source table can satisfy it and still be too
    /// narrow to transpose.
    ColumnCountTooNarrow,
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

    /// The column's bits, in ascending row order.
    pub fn column_bits(&self, column: usize) -> BitsIter<'a> {
        BitsIter::new(self.column(column).iter())
    }

    /// Swaps rows and columns, returning an owned copy: `result.bit(b, c) ==
    /// self.bit(c, b)` for every row `b` and column `c` of `self`.
    ///
    /// Built for callers like a leaf construction that must walk the table
    /// row by row: the packing is column-major (see the type docs), so a
    /// row-major walk over `self` is a scatter, one `bit()` call and cache
    /// miss per cell. Transposing once up front turns that into the same
    /// sequential access [`BitTable::column_bits`] already gives per column,
    /// just over what were originally rows.
    ///
    /// # Constraints
    ///
    /// - **Column count**: `self.shape().log_columns()` must be at least `7`
    ///   (`PACK_BITS`), i.e. at least `128` columns. See
    ///   [`TransposeError::ColumnCountTooNarrow`] -- this is the only way the
    ///   operation fails.
    /// - **Commitment-size window**: never a separate concern. The total bit
    ///   count `shape().log_bits() = log_rows + log_columns` is a sum, so
    ///   swapping its two terms leaves it unchanged; the result automatically
    ///   sits in `[MIN_LOG_BITS, MAX_LOG_BITS]` whenever `self` did.
    /// - **Allocation**: always a full out-of-place copy -- a fresh buffer
    ///   the same size as `self`'s packed witness (up to 4 GiB at `m = 35`),
    ///   never a view over the original.
    /// - **Involution**: transposing the result returns `self`'s bits
    ///   exactly. The forward direction is the only place the column-count
    ///   constraint can fail: once it holds, the result's row count is the
    ///   source's (old) column count and its column count is the source's
    ///   (old) row count, and every `BitTable` already has `log_rows >= 7`
    ///   unconditionally -- so the second transpose is never the one that
    ///   rejects.
    pub fn transpose(&self) -> Result<TransposedBitTable, TransposeError> {
        let shape =
            Shape::new(self.shape.log_columns(), self.shape.log_rows()).map_err(|error| {
                debug_assert_eq!(
                    error,
                    ShapeError::RowIndexTooNarrow,
                    "log_bits is preserved by swapping row/column axes, so the \
                     commitment-size window cannot be what rejected the swapped shape"
                );
                TransposeError::ColumnCountTooNarrow
            })?;

        let packed = bit_transpose(self.packed, self.shape.columns(), self.shape.rows());

        Ok(TransposedBitTable { shape, packed })
    }
}

// Bits packed into machine words
// out of place variant.
// dim1 and dim2 should be given in bits
// dim2 is the axis over which the data is adjacent, think xs[dim1][dim2].
fn bit_transpose(xs: &[Word], dim1: usize, dim2: usize) -> Vec<Word> {
    assert_eq!(xs.len() * Word::BITS as usize, dim1 * dim2);
    let mut out = bytemuck::zeroed_vec(xs.len());

    // Bit transpose approach is dividing the matrix up in blocks of WORD::BITS x WORD::BITS. Perform transpose on the block and write the block to the transposed position.

    let dim1_groups = dim1 / (Word::BITS as usize); // chunk size in dim1 (same as number of words in line)
    let dim2_groups = dim2 / (Word::BITS as usize); // chunk size in dim2
    for d1 in 0..dim1_groups {
        for d2 in 0..dim2_groups {
            let mut block = [0; Word::BITS as usize];

            for (lane, word) in block.iter_mut().enumerate() {
                *word = xs[(d1 * Word::BITS as usize + lane) * dim2_groups + d2]
            }

            bit_block_transpose(&mut block);

            let dim1t = d2;
            let dim2t = d1;
            let dim2t_groups = dim1_groups;

            for (lane, word) in block.iter().enumerate() {
                out[(dim1t * Word::BITS as usize + lane) * dim2t_groups + dim2t] = *word;
            }
        }
    }

    out
}

fn bit_block_transpose(xs: &mut [Word; Word::BITS as usize]) {
    let mut j = (Word::BITS / 2) as usize;
    let mut mask = (1 << j) - 1; // Lower half high
    while j > 0 {
        let mut k: usize = 0;
        while k < Word::BITS as usize {
            let t = xs[k];
            let b = xs[k | j];
            // Alternative is a delta swap, however this is more readable.
            xs[k] = (t & mask) | ((b << j) & !mask);
            xs[k | j] = ((t >> j) & mask) | (b & !mask);
            // (k|j) count the upper part. +1 advances the upper part. !j converts it into the lower index again except for when it hits the next round. In that case the | j was
            k = ((k | j) + 1) & !j;
        }
        j >>= 1;
        // Generate the alternating bit pattern
        // 00001111
        // 00110011
        // 01010101
        mask ^= mask << j;
    }
}

/// An owned, transposed copy of a [`BitTable`]'s bits -- see
/// [`BitTable::transpose`], which is the only way to build one.
#[derive(Debug)]
pub struct TransposedBitTable {
    shape: Shape,
    packed: Vec<u64>,
}

impl TransposedBitTable {
    /// The transposed shape: rows and columns swapped from the source.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Borrows the transposed bits as an ordinary [`BitTable`].
    ///
    /// Infallible: [`BitTable::transpose`] already sized this buffer to
    /// match `shape`.
    pub fn as_table(&self) -> BitTable<'_> {
        BitTable::new(self.shape, &self.packed)
            .expect("a TransposedBitTable's shape and packed length always agree")
    }

    /// Unwraps the packed, transposed bits.
    pub fn into_packed(self) -> Vec<u64> {
        self.packed
    }
}

pub type BitsIter<'a> = StepIter<'a, 1>;

/// Iterator over one column's bits, in ascending row order — see
/// [`BitTable::column_bits`].
pub struct StepIter<'a, const S: usize> {
    elements: std::slice::Iter<'a, Word>,
    /// Bits not yet emitted; the next one to emit is the LSB.
    word: Word,
    remaining: u8,
}

impl<'a, const S: usize> StepIter<'a, S> {
    const CHECK_SIZE: () = assert!(
        S != 0 && S <= Word::BITS as usize && Word::BITS as usize % S == 0,
        "step size must be nonzero, at most 64, and divide 64 evenly"
    );
    const MASK: u64 = 1u64.unbounded_shl(S as u32).wrapping_sub(1);
    fn new(iter: Iter<'a, Word>) -> Self {
        let () = Self::CHECK_SIZE;
        Self {
            elements: iter,
            word: 0,
            remaining: 0,
        }
    }
}

impl<'a, const S: usize> Iterator for StepIter<'_, S> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            let element = self.elements.next()?;
            self.word = *element;
            self.remaining = Word::BITS as u8;
        }
        let bitset = self.word & Self::MASK;
        self.word = self.word.unbounded_shr(S as u32);
        self.remaining -= S as u8;
        Some(bitset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, const S: usize> ExactSizeIterator for StepIter<'_, S> {
    fn len(&self) -> usize {
        (self.remaining as usize + self.elements.len() * (BitTable::BITS as usize)) / S
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use num_traits::ConstZero;

    // We want to use compare against
    use field::F128;

    /// Bits in a packed element, and the shift that divides an index into element
    /// and offset.
    const PACKED_BITS: usize = 128;
    const PACKED_SHIFT: u32 = 7;
    /// Bits in each half of a packed element.
    const HALF_BITS: usize = 64;

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
        assert_eq!(table.column(3), [0b101u64, 1u64]);
        assert!(table.column(4).iter().all(|&element| element == 0));
    }

    #[test]
    fn column_bits_agrees_with_the_bit_accessor() {
        let shape = small_shape();
        let rows: Vec<(usize, usize)> = (0..shape.rows()).step_by(7).map(|row| (row, 2)).collect();
        let packed = with_bits(&shape, &rows);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        let iter = table.column_bits(2);
        assert_eq!(iter.len(), shape.rows());
        let bits: Vec<bool> = iter.map(|b| b == 1).collect();
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
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        for (index, &element) in table.column(2).iter().enumerate() {
            for offset in 0..BitTable::BITS {
                let row = index * BitTable::BITS + offset;
                assert_eq!(table.bit(2, row), (element >> offset) & 1 == 1, "row {row}");
            }
        }
    }

    /// A cheap, deterministic stand-in for randomness: no `rand` dependency,
    /// but different `seed`s still produce unrelated-looking bit patterns.
    fn pseudo_random_bit(seed: u64) -> bool {
        (seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63) & 1 == 1
    }

    #[test]
    fn bit_block_transpose_matches_a_brute_force_reference() {
        let mut original = [0 as Word; BitTable::BITS];
        for (i, word) in original.iter_mut().enumerate() {
            *word = (i as Word).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xA5A5;
        }

        let mut transposed = original;
        bit_block_transpose(&mut transposed);

        for (i, &transposed_word) in transposed.iter().enumerate() {
            for (j, &original_word) in original.iter().enumerate() {
                assert_eq!(
                    (transposed_word >> j) & 1,
                    (original_word >> i) & 1,
                    "word {i} bit {j}"
                );
            }
        }
    }

    #[test]
    fn transpose_reference_matches_bit_transpose() {
        // 128 segments of 4 words each: two block groups on both axes.
        const GS: usize = 128;
        let segment_n = 4;
        let tt: Vec<Word> = (0..GS * segment_n)
            .map(|i| (i as Word).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xA5A5)
            .collect();

        let reference = transpose_reference::<GS>(&tt);
        assert_eq!(reference.len(), tt.len());
        assert_eq!(
            reference,
            bit_transpose(&tt, GS, segment_n * BitTable::BITS)
        );
    }

    /// Two row groups (`log_rows = 8`), two column groups (`log_columns =
    /// 15`), so the block-tiling loop in `transpose` runs more than once on
    /// both axes.
    fn transpose_shape() -> Shape {
        Shape::new(8, 15).unwrap()
    }

    fn pseudo_random_table(shape: &Shape) -> Vec<F128> {
        let mut packed = vec![F128::ZERO; (1 << shape.log_bits()) / PACKED_BITS];
        for column in 0..shape.columns() {
            for row in 0..shape.rows() {
                let seed = (column as u64).wrapping_mul(2654435761) ^ (row as u64);
                if pseudo_random_bit(seed) {
                    let index = (column << shape.log_rows()) | row;
                    let element = &mut packed[index >> PACKED_SHIFT];
                    let offset = index % PACKED_BITS;
                    if offset < HALF_BITS {
                        element.lo |= 1u64 << offset;
                    } else {
                        element.hi |= 1u64 << (offset - HALF_BITS);
                    }
                }
            }
        }
        packed
    }

    #[test]
    fn transpose_swaps_row_and_column_bits() {
        let shape = transpose_shape();
        let packed = pseudo_random_table(&shape);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        let transposed = table.transpose().unwrap();
        assert_eq!(transposed.shape().log_rows(), shape.log_columns());
        assert_eq!(transposed.shape().log_columns(), shape.log_rows());

        let transposed = transposed.as_table();

        // Full columns at both ends of each axis, including a block
        // boundary (128 rows per group here), rather than every column --
        // this shape alone already has 2^15 of them.
        for column in [0, 1, shape.columns() / 2, shape.columns() - 1] {
            for row in [0, 1, PACKED_BITS - 1, PACKED_BITS, shape.rows() - 1] {
                assert_eq!(
                    transposed.bit(row, column),
                    table.bit(column, row),
                    "B[{column}][{row}] vs its transpose"
                );
            }
        }
    }

    #[test]
    fn transposing_twice_recovers_the_original_bits() {
        let shape = transpose_shape();
        let packed = pseudo_random_table(&shape);
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        let roundtripped = table
            .transpose()
            .unwrap()
            .as_table()
            .transpose()
            .unwrap()
            .into_packed();

        assert_eq!(roundtripped, bytemuck::cast_slice::<F128, Word>(&packed));
    }

    #[test]
    fn transpose_rejects_a_table_with_too_few_columns() {
        // `log_columns = 0`: a single column, well under the 128 a
        // transposed row would need to fill one packed element -- even
        // though this shape is perfectly admissible for `BitTable` itself.
        let shape = Shape::new(22, 0).unwrap();
        let packed = vec![F128::ZERO; (1 << shape.log_bits()) / PACKED_BITS];
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        assert_eq!(
            table.transpose().err(),
            Some(TransposeError::ColumnCountTooNarrow)
        );
    }

    fn transpose_reference<const GS: usize>(tt: &[Word]) -> Vec<Word> {
        assert_eq!(tt.len() % GS, 0);
        let segment_n = tt.len() / GS;
        let mut segments: [_; GS] =
            array::from_fn(|i| BitsIter::new(tt[i * segment_n..(i + 1) * segment_n].iter()));

        let mut out = Vec::with_capacity(tt.len());
        let mut word: Word = 0;
        let mut offset = 0;
        'outer: loop {
            for s in &mut segments {
                let Some(b) = s.next() else { break 'outer };
                // LSB first, like `bit()` and `StepIter`.
                word |= b << offset;
                offset += 1;
                if offset == BitTable::BITS {
                    out.push(word);
                    word = 0;
                    offset = 0;
                }
            }
        }
        debug_assert_eq!(offset, 0);

        out
    }
}
