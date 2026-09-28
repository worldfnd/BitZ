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
//
// A block is always BITS words, built from `d = min(dim1, BITS)` rows of
// `r = BITS / d` consecutive words each: word i = c * d + a is row a, word c.
// Writing a block's index bits as (word | bit), the input is
// ([c | a] | [p_hi | p_lo]) with a and p_hi log2(d) bits wide, and the
// output needs ([c | p_hi] | [p_lo | a]): the bit index rotates by log2(d).
//
// A butterfly stage swaps one word index bit with one bit index bit. Stages
// s < log d swap a with the low bits of p: every d x d tile is transposed and
// each word holds correct d-bit pieces. The later stages move whole pieces
// with the same word index bits, s % log d: stage s puts the bit that stage
// s - log d took out of the bit index back in, log d places higher. That is
// one sweep of 6 stages for any d >= 2, and it has to run upwards. The word
// index ends as [c | p_hi] with p_hi's bits rotated by 6 % log d, which the
// indexed write undoes. For d = BITS this is the plain blocked transpose.
fn bit_transpose(xs: &[Word], dim1: usize, dim2: usize) -> Vec<Word> {
    assert_eq!(xs.len() * BitTable::BITS, dim1 * dim2);
    assert!(dim1.is_power_of_two(), "dim1 {dim1} is not a power of two");

    // One body, compiled once per block height. With `log2(d)` a runtime
    // value the stage loops cannot be unrolled with constant masks and
    // distances, which cost ~30% at d = 64.
    match dim1.min(BitTable::BITS).trailing_zeros() {
        // A single row is its own transpose.
        0 => xs.to_vec(),
        1 => bit_transpose_blocks::<1>(xs, dim1, dim2),
        2 => bit_transpose_blocks::<2>(xs, dim1, dim2),
        3 => bit_transpose_blocks::<3>(xs, dim1, dim2),
        4 => bit_transpose_blocks::<4>(xs, dim1, dim2),
        5 => bit_transpose_blocks::<5>(xs, dim1, dim2),
        _ => bit_transpose_blocks::<6>(xs, dim1, dim2),
    }
}

/// [`bit_transpose`] for blocks of `d = 2^LOG_D` rows, `LOG_D >= 1`.
fn bit_transpose_blocks<const LOG_D: u32>(xs: &[Word], dim1: usize, dim2: usize) -> Vec<Word> {
    const BITS: usize = Word::BITS as usize;
    const LOG_BITS: u32 = Word::BITS.trailing_zeros();

    let d = 1 << LOG_D; // rows per block
    let r = BITS / d; // words per row per block
    let row_words = dim2 / BITS;
    // Output words between consecutive words of a block; 1 when d < BITS.
    let out_stride = dim1 / d;
    assert!(
        dim2.is_multiple_of(BITS) && row_words.is_multiple_of(r),
        "dim2 {dim2} does not fill whole {d}-row blocks"
    );

    // The sweep leaves p_hi in the low LOG_D word index bits rotated left
    // by `rot`; the indexed write rotates them back.
    let rot = LOG_BITS % LOG_D;

    let mut out = bytemuck::zeroed_vec(xs.len());

    for g1 in 0..dim1 / d {
        for g2 in 0..row_words / r {
            // Word i = [c | a] is row a, word c. Two loops rather than one
            // over i: splitting i back into a and c measured 4-12% slower
            // for d < BITS.
            let mut block = [0; BITS];
            for c in 0..r {
                for a in 0..d {
                    block[c * d + a] = xs[(g1 * d + a) * row_words + g2 * r + c];
                }
            }

            bit_block_transpose(LOG_D, &mut block);

            // o is output row g2 * BITS + o, word g1 in [dim2][dim1]; g1 is
            // only nonzero when d = BITS.
            for (i, &word) in block.iter().enumerate() {
                let low = i & (d - 1);
                let o = (i & !(d - 1)) | ((low >> rot) | (low << (LOG_D - rot))) & (d - 1);
                out[(g2 * BITS + o) * out_stride + g1] = word;
            }
        }
    }

    out
}

/// Runs the 6 butterfly stages on a BITS x BITS block, lowest first. Stage
/// `s` swaps bit index bit `s` with word index bit `s % log_d`, so
/// `log_d = 6` is the full transpose. See [`bit_transpose`] for smaller
/// `log_d`.
#[inline(always)]
fn bit_block_transpose(log_d: u32, xs: &mut [Word; Word::BITS as usize]) {
    for s in 0..Word::BITS.trailing_zeros() {
        let j = 1 << s;
        // Alternating runs of j bits, lowest run set:
        // 01010101, 00110011, 00001111
        let mask = Word::MAX / ((1 << j) + 1);
        // Distance between the paired words.
        let jw = 1 << (s % log_d);
        let mut k: usize = 0;
        while k < Word::BITS as usize {
            let t = xs[k];
            let b = xs[k | jw];
            // Alternative is a delta swap, however this is more readable.
            xs[k] = (t & mask) | ((b << j) & !mask);
            xs[k | jw] = ((t >> j) & mask) | (b & !mask);
            // (k|jw) count the upper part. +1 advances the upper part. !jw converts it into the lower index again except for when it hits the next round. In that case the | jw was
            k = ((k | jw) + 1) & !jw;
        }
    }
}

/// Bit-at-a-time transpose of `GS` equal segments of `tt`: output bit `i`
/// is bit `i / GS` of segment `i % GS`. The oracle [`bit_transpose`] is
/// checked (and benchmarked) against.
#[cfg(any(test, feature = "bench"))]
fn transpose_reference<const GS: usize>(tt: &[Word]) -> Vec<Word> {
    assert_eq!(tt.len() % GS, 0);
    let segment_n = tt.len() / GS;
    let mut segments: [_; GS] =
        std::array::from_fn(|i| BitsIter::new(tt[i * segment_n..(i + 1) * segment_n].iter()));

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

/// Benchmark-only access to the private transposes; see `benches/table.rs`.
#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench {
    pub fn bit_transpose(xs: &[u64], dim1: usize, dim2: usize) -> Vec<u64> {
        super::bit_transpose(xs, dim1, dim2)
    }

    pub fn transpose_reference<const GS: usize>(tt: &[u64]) -> Vec<u64> {
        super::transpose_reference::<GS>(tt)
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
        bit_block_transpose(6, &mut transposed);

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

    /// `bit_transpose` of `GS` rows of `words` words each against the
    /// bit-at-a-time reference.
    fn check_bit_transpose<const GS: usize>(words: usize) {
        let tt: Vec<Word> = (0..GS * words)
            .map(|i| (i as Word).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xA5A5)
            .collect();
        assert_eq!(
            bit_transpose(&tt, GS, words * BitTable::BITS),
            transpose_reference::<GS>(&tt),
            "dim1 {GS}, {words} words per row"
        );
    }

    #[test]
    fn bit_transpose_matches_reference_for_every_row_count() {
        // Two blocks along dim2 at every row count, so block placement is
        // exercised as well as the block contents.
        check_bit_transpose::<1>(128);
        check_bit_transpose::<2>(64);
        check_bit_transpose::<4>(32);
        check_bit_transpose::<8>(16);
        check_bit_transpose::<16>(8);
        check_bit_transpose::<32>(4);
        check_bit_transpose::<64>(2);
        check_bit_transpose::<128>(2);
        check_bit_transpose::<256>(2);
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
}
