//! The committed bit table.

use std::{ops::Deref, slice::Iter};

use crate::Shape;
/// The word `packed` is sliced into: one `F128` element. [`BitTable::BITS`]
/// is derived from this, so changing it is the only step needed to repack
/// into a different word size of at least 64 bits.
///
/// 128 rather than 64 bits because `bit_transpose` is bound by its strided
/// loads and stores, and each one then moves 16 bytes instead of 8: the same
/// code with `u64` measured 20-55% slower on `WordColumns::transpose`.
type Word = u128;
/// Bits in a [`Word`]'s bit index.
const LOG_BITS: u32 = Word::BITS.trailing_zeros();
/// Bits in a `u64`'s bit index. Swapping two bit index bits below this never
/// moves a bit from one of a word's `u64` lanes to another.
const LOG_LANE_BITS: u32 = u64::BITS.trailing_zeros();

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
/// The formula is written for `t >= 7`; in general element `k` holds bit
/// indices `128 k` to `128 k + 127`, which below that is several whole columns.
///
/// Generic over the storage `S` of the packed words; use the [`BitTable`]
/// (borrowed) and [`OwnedBitTable`] aliases rather than naming this directly.
#[derive(Debug, Clone, Copy)]
pub struct BitTableBase<S> {
    shape: Shape,
    packed: S,
}

/// A [`BitTableBase`] borrowing the prover's packed witness.
pub type BitTable<'a> = BitTableBase<&'a [Word]>;

/// A [`BitTableBase`] owning its packed words, as [`WordColumns::transpose`]
/// returns.
pub type OwnedBitTable = BitTableBase<Vec<Word>>;

impl<S> BitTableBase<S> {
    pub const BITS: usize = Word::BITS as usize;
}

impl<S: Deref<Target = [Word]>> BitTableBase<S> {
    /// Wraps `packed` in `shape`, least significant bit first inside `lo`.
    ///
    /// Crate-private: [`crate::BitZParams::table`] is the only way in, so a
    /// table is always shaped by a checked parameter set.
    pub(crate) fn new(shape: Shape, packed: S) -> Result<Self, TableError> {
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

    /// The table read column by column as whole elements, or `None` when
    /// `t < 7` and a column is narrower than one.
    pub fn word_columns(&self) -> Option<WordColumns<'_>> {
        let words = self.shape.rows() / BitTable::BITS;
        (words > 0).then(|| WordColumns {
            packed: &self.packed,
            words,
        })
    }

    /// The column's bits, in ascending row order.
    pub fn column_bits(&self, column: usize) -> BitsIter<'_> {
        if let Some(columns) = self.word_columns() {
            return BitsIter::new(columns.column(column).iter());
        }
        // A column narrower than an element shares it with its neighbours:
        // shift it down to bit 0 and stop after its `2^t` rows.
        let start = column << self.shape.log_rows();
        BitsIter::preloaded(
            self.packed[start / BitTable::BITS] >> (start % BitTable::BITS),
            self.shape.rows(),
        )
    }

    /// Unwraps the packed bits.
    pub fn into_packed(self) -> S {
        self.packed
    }
}

/// A table's columns as whole elements, which needs `t >= 7`; from
/// [`BitTableBase::word_columns`]. The fold and [`WordColumns::transpose`]
/// both read the table this way.
///
// Refactoring Seam might be removed later.
#[derive(Debug, Clone, Copy)]
pub struct WordColumns<'a> {
    packed: &'a [Word],
    /// Elements per column, `2^(t-7)`.
    words: usize,
}

impl<'a> WordColumns<'a> {
    /// The `2^(t-7)` elements holding one column, in ascending row order.
    ///
    /// A column starts at bit `c * 2^t` and `t >= 7`, so it begins on an
    /// element boundary and spans whole elements. The fold walks these directly
    /// rather than calling [`BitTable::bit`] once per row: at `m = 35` that is
    /// `2^35` calls, each of them a bounds check and two shifts.
    pub fn column(&self, column: usize) -> &'a [Word] {
        &self.packed[column * self.words..(column + 1) * self.words]
    }

    /// The number of columns.
    pub fn columns(&self) -> usize {
        self.packed.len() / self.words
    }

    /// Swaps rows and columns, returning an owned copy: `result.bit(b, c) ==
    /// table.bit(c, b)` for every row `b` and column `c` of the table.
    ///
    /// Built for callers like a leaf construction that must walk the table
    /// row by row: the packing is column-major (see [`BitTableBase`]), so a
    /// row-major walk over the table is a scatter, one `bit()` call and cache
    /// miss per cell. Transposing once up front turns that into the same
    /// sequential access [`BitTable::column_bits`] already gives per column,
    /// just over what were originally rows.
    ///
    /// # Constraints
    ///
    /// - **Row count**: `bit_transpose` reads each column as whole words,
    ///   which is what this view guarantees. The column count is free: with
    ///   fewer than `128` columns the result's columns share words.
    /// - **Commitment-size window**: never a separate concern. The total bit
    ///   count `shape().log_bits() = log_rows + log_columns` is a sum, so
    ///   swapping its two terms leaves it unchanged; the result automatically
    ///   sits in `[MIN_LOG_BITS, MAX_LOG_BITS]` whenever the table did.
    /// - **Allocation**: always a full out-of-place copy -- a fresh buffer
    ///   the same size as the packed witness (up to 4 GiB at `m = 35`), never
    ///   a view over the original.
    /// - **Involution**: transposing the result returns the table's bits
    ///   exactly, provided it has at least `128` columns. With fewer, the
    ///   result has no word view of its own to transpose.
    pub fn transpose(&self) -> OwnedBitTable {
        let (columns, rows) = (self.columns(), self.words * BitTable::BITS);
        let shape = Shape::new(columns.ilog2() as usize, rows.ilog2() as usize).expect(
            "log_bits is preserved by swapping row/column axes, and the \
             commitment-size window is all a shape is checked against",
        );

        OwnedBitTable {
            shape,
            packed: bit_transpose(self.packed, columns, rows),
        }
    }
}

/// out of place variant.
/// dim1 and dim2 are in bits
/// dim2 is the axis over which the data is adjacent, think xs[dim1][dim2].
/// dim1 < WORD::Bits dim2 >= WORD::Bits
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
        6 => bit_transpose_blocks::<6>(xs, dim1, dim2),
        _ => bit_transpose_blocks::<LOG_BITS>(xs, dim1, dim2),
    }
}

/// [`bit_transpose`] for blocks of `d = 2^LOG_D` rows, `LOG_D >= 1`.
fn bit_transpose_blocks<const LOG_D: u32>(xs: &[Word], dim1: usize, dim2: usize) -> Vec<Word> {
    const BITS: usize = Word::BITS as usize;

    let d = 1 << LOG_D; // rows per block
    let r = BITS / d; // words per row per block
    let row_words = dim2 / BITS;
    // Output words between consecutive words of a block; 1 when d < BITS.
    let out_stride = dim1 / d;
    assert!(
        dim2.is_multiple_of(BITS) && row_words.is_multiple_of(r),
        "dim2 {dim2} does not fill whole {d}-row blocks"
    );

    let mut out = bytemuck::zeroed_vec(xs.len());

    // g1 innermost, so consecutive blocks write neighbouring words of the
    // same output rows while those lines are still cached, and it is the
    // input that is revisited after a sweep; the other way round measured
    // ~10% slower on a 2^13-column table. g1 only takes more than one value
    // when d = BITS.
    for g2 in 0..row_words / r {
        for g1 in 0..dim1 / d {
            // Word i = [c | a] is row a, word c. Two loops rather than one
            // over i: splitting i back into a and c measured 4-12% slower
            // for d < BITS.
            let mut block = [0; BITS];
            for c in 0..r {
                for a in 0..d {
                    block[c * d + a] = xs[(g1 * d + a) * row_words + g2 * r + c];
                }
            }

            rotate_bit_index::<LOG_D>(&mut block);
            bit_block_transpose(LOG_D, &mut block);

            // o is output row g2 * BITS + o, word g1 in [dim2][dim1]; g1 is
            // only nonzero when d = BITS.
            for (o, &word) in block.iter().enumerate() {
                out[(g2 * BITS + o) * out_stride + g1] = word;
            }
        }
    }

    out
}

/// The bits whose index has bit `s` clear: alternating runs of `2^s` bits,
/// lowest run set. 01010101, 00110011, 00001111 for `s` = 0, 1, 2.
const fn index_bit_clear(s: u32) -> Word {
    Word::MAX / ((1 << (1 << s)) + 1)
}

/// Interleaves the `d = 2^LOG_D` chunks of `r = BITS / d` bits in every word
/// of `block`, a `d`-way perfect shuffle: bit `j` of chunk `k` moves to bit
/// `j * d + k`. On bit indices that is a left rotation by `LOG_D`: bit
/// `[p_hi | p_lo] = [k | j]` moves to `[p_lo | p_hi]`, so unlike
/// `rotate_left` each bit moves its own distance. That puts `p_hi` in the low
/// `LOG_D` index bits, where [`bit_block_transpose`] swaps it with the row
/// bits of the word index. One pass over the block per entry of
/// [`rotation_swaps`], none for `LOG_D = LOG_BITS`.
///
/// A pass that stays inside `u64` lanes runs on the lanes rather than on
/// whole words, which vectorises: on `u128` that made the rotation about
/// 40% cheaper. Only a swap with bit index bit 6 or above needs whole words.
#[inline(always)]
fn rotate_bit_index<const LOG_D: u32>(block: &mut [Word; Word::BITS as usize]) {
    // Evaluated at compile time, so every shift and mask is a constant.
    let (swaps, n) = const { rotation_swaps(LOG_D) };
    for &(shift, mask, in_lane) in &swaps[..n] {
        if in_lane {
            let mask = mask as u64;
            for x in bytemuck::cast_slice_mut::<Word, u64>(block) {
                let t = ((*x >> shift) ^ *x) & mask;
                *x ^= t ^ (t << shift);
            }
        } else {
            for x in block.iter_mut() {
                let t = ((*x >> shift) ^ *x) & mask;
                *x ^= t ^ (t << shift);
            }
        }
    }
}

/// The delta swaps that rotate a bit index left by `log_d`, as `(shift,
/// mask, in_lane)`: each bit in `mask` trades places with the bit `shift`
/// above it, and `in_lane` says no bit crosses from one `u64` lane to
/// another. Returns the swaps and how many of them are used.
///
/// Each swap exchanges two bits of the bit index, `low` and `i`: the bits
/// whose index has `low` set and `i` clear move up by `shift`, their partners
/// move down by it, and the other half of the word stays put. The rotation
/// splits the index bits into `gcd(LOG_BITS, log_d)` cycles. Each cycle is
/// walked from its lowest index bit `low`, which is swapped with the cycle's
/// other bits `i` in turn: `LOG_BITS - gcd` swaps in all, the fewest index
/// bit swaps that make the rotation.
const fn rotation_swaps(log_d: u32) -> ([(u32, Word, bool); LOG_BITS as usize], usize) {
    let (mut cycles, mut rest) = (LOG_BITS, log_d);
    while rest != 0 {
        (cycles, rest) = (rest, cycles % rest);
    }

    let mut swaps = [(0, 0, false); LOG_BITS as usize];
    let mut n = 0;
    let mut low = 0;
    while low < cycles {
        let mut i = (low + log_d) % LOG_BITS;
        while i != low {
            // The bits whose index has bit low set and bit i clear.
            swaps[n] = (
                (1 << i) - (1 << low),
                !index_bit_clear(low) & index_bit_clear(i),
                i < LOG_LANE_BITS,
            );
            n += 1;
            i = (i + log_d) % LOG_BITS;
        }
        low += 1;
    }
    (swaps, n)
}

/// Transposes every `2^log_d x 2^log_d` tile of a BITS x BITS block, the
/// tile's words being those that differ only in their low `log_d` index
/// bits. Stage `s` swaps bit index bit `s` with word index bit `s`, coarse
/// to fine; the stages commute, so the order is free. `log_d = LOG_BITS` is
/// the full transpose.
#[inline(always)]
fn bit_block_transpose(log_d: u32, xs: &mut [Word; Word::BITS as usize]) {
    for s in (0..log_d).rev() {
        // Distance between the paired bits, and between the paired words.
        let j: usize = 1 << s;
        let mask = index_bit_clear(s);
        let mut k: usize = 0;
        while k < Word::BITS as usize {
            // A delta swap across the pair: t is where the upper runs of
            // xs[k] and the lower runs of xs[k | j] differ, and flipping those
            // bits in both swaps the runs. Measured 6-10% faster than
            // rebuilding both words from masked halves.
            let t = ((xs[k] >> j) ^ xs[k | j]) & mask;
            xs[k | j] ^= t;
            xs[k] ^= t << j;
            // (k|j) count the upper part. +1 advances the upper part. !j converts it into the lower index again except for when it hits the next round. In that case the | j was
            k = ((k | j) + 1) & !j;
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
    pub fn bit_transpose(xs: &[super::Word], dim1: usize, dim2: usize) -> Vec<super::Word> {
        super::bit_transpose(xs, dim1, dim2)
    }

    pub fn transpose_reference<const GS: usize>(tt: &[super::Word]) -> Vec<super::Word> {
        super::transpose_reference::<GS>(tt)
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
        S != 0 && S <= Word::BITS as usize && (Word::BITS as usize).is_multiple_of(S),
        "step size must be nonzero, at most Word::BITS, and divide Word::BITS evenly"
    );
    const MASK: Word = (1 as Word).unbounded_shl(S as u32).wrapping_sub(1);
    fn new(iter: Iter<'a, Word>) -> Self {
        let () = Self::CHECK_SIZE;
        Self {
            elements: iter,
            word: 0,
            remaining: 0,
        }
    }

    /// Emits the low `bits` bits of `word` and nothing after; `bits` is a
    /// multiple of `S` and at most `Word::BITS`.
    fn preloaded(word: Word, bits: usize) -> Self {
        let () = Self::CHECK_SIZE;
        debug_assert!(bits.is_multiple_of(S) && bits <= Word::BITS as usize);
        Self {
            elements: [].iter(),
            word,
            remaining: bits as u8,
        }
    }
}

impl<const S: usize> Iterator for StepIter<'_, S> {
    type Item = Word;

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

impl<const S: usize> ExactSizeIterator for StepIter<'_, S> {
    fn len(&self) -> usize {
        (self.remaining as usize + self.elements.len() * BitTable::BITS) / S
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
        let columns = table.word_columns().unwrap();

        assert_eq!(columns.columns(), shape.columns());
        assert_eq!(columns.column(3).len(), shape.rows() / BitTable::BITS);
        assert_eq!(columns.column(3), [0b101 | (1 << 64)]);
        assert!(columns.column(4).iter().all(|&element| element == 0));
    }

    /// So neither the fold nor `transpose` can be handed it.
    #[test]
    fn a_column_narrower_than_an_element_has_no_word_view() {
        let shape = Shape::new(6, 16).unwrap();
        let packed = vec![F128::ZERO; (1 << shape.log_bits()) / PACKED_BITS];
        let table = BitTable::new(shape, bytemuck::cast_slice(&packed)).unwrap();

        assert!(table.word_columns().is_none());
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

        for (index, &element) in table.word_columns().unwrap().column(2).iter().enumerate() {
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

    /// A word with bits spread over its whole width, distinct for every `i`.
    fn pseudo_random_word(i: usize) -> Word {
        (i as Word).wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15) ^ 0xA5A5
    }

    #[test]
    fn bit_block_transpose_matches_a_brute_force_reference() {
        let mut original = [0 as Word; BitTable::BITS];
        for (i, word) in original.iter_mut().enumerate() {
            *word = pseudo_random_word(i);
        }

        let mut transposed = original;
        bit_block_transpose(LOG_BITS, &mut transposed);

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

    /// Bit `p` must land at `p` rotated left by `LOG_D` within `LOG_BITS`
    /// bits. Word `p` of the block holds only bit `p`, so one call checks
    /// every bit.
    fn check_rotate_bit_index<const LOG_D: u32>() {
        let mut block: [Word; BitTable::BITS] = std::array::from_fn(|p| 1 << p);
        rotate_bit_index::<LOG_D>(&mut block);
        for (p, &word) in block.iter().enumerate() {
            let rotated = ((p << LOG_D) | (p >> (LOG_BITS - LOG_D))) % BitTable::BITS;
            assert_eq!(word, 1 << rotated, "LOG_D {LOG_D}, bit {p}");
        }
    }

    #[test]
    fn rotate_bit_index_rotates_every_bit_index() {
        check_rotate_bit_index::<1>();
        check_rotate_bit_index::<2>();
        check_rotate_bit_index::<3>();
        check_rotate_bit_index::<4>();
        check_rotate_bit_index::<5>();
        check_rotate_bit_index::<6>();
        check_rotate_bit_index::<7>();
    }

    /// `bit_transpose` of `GS` rows of `words` words each against the
    /// bit-at-a-time reference.
    fn check_bit_transpose<const GS: usize>(words: usize) {
        let tt: Vec<Word> = (0..GS * words).map(pseudo_random_word).collect();
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
        check_bit_transpose::<1>(256);
        check_bit_transpose::<2>(128);
        check_bit_transpose::<4>(64);
        check_bit_transpose::<8>(32);
        check_bit_transpose::<16>(16);
        check_bit_transpose::<32>(8);
        check_bit_transpose::<64>(4);
        check_bit_transpose::<128>(2);
        check_bit_transpose::<256>(2);
    }

    #[test]
    fn transpose_reference_matches_bit_transpose() {
        // 256 segments of 2 words each: two block groups on both axes.
        const GS: usize = 256;
        let segment_n = 2;
        let tt: Vec<Word> = (0..GS * segment_n).map(pseudo_random_word).collect();

        let reference = transpose_reference::<GS>(&tt);
        assert_eq!(reference.len(), tt.len());
        assert_eq!(
            reference,
            bit_transpose(&tt, GS, segment_n * BitTable::BITS)
        );
    }

    /// Every row width below one element, where columns share elements.
    #[test]
    fn column_bits_agrees_with_the_bit_accessor_below_one_element() {
        for log_rows in 0..PACKED_SHIFT as usize {
            let shape = Shape::new(log_rows, 22 - log_rows).unwrap();
            let packed: Vec<Word> = (0..(1 << shape.log_bits()) / BitTable::BITS)
                .map(pseudo_random_word)
                .collect();
            let table = BitTable::new(shape, &packed).unwrap();

            // Both ends of the first, a middle and the last element.
            let per_element = BitTable::BITS / shape.rows();
            let last = shape.columns() - 1;
            for column in [0, 1, per_element - 1, per_element, last / 2, last] {
                let iter = table.column_bits(column);
                assert_eq!(iter.len(), shape.rows());
                let bits: Vec<bool> = iter.map(|b| b == 1).collect();
                let expected: Vec<bool> = (0..shape.rows())
                    .map(|row| table.bit(column, row))
                    .collect();
                assert_eq!(bits, expected, "t {log_rows}, column {column}");
            }
        }
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

        let transposed = table.word_columns().unwrap().transpose();
        assert_eq!(transposed.shape().log_rows(), shape.log_columns());
        assert_eq!(transposed.shape().log_columns(), shape.log_rows());

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
            .word_columns()
            .unwrap()
            .transpose()
            .word_columns()
            .unwrap()
            .transpose()
            .into_packed();

        assert_eq!(roundtripped, bytemuck::cast_slice::<F128, Word>(&packed));
    }

    /// Every column count below one element, down to a single column: the
    /// result's rows share elements, and its columns read back through
    /// `column_bits` as the source's rows.
    #[test]
    fn transpose_handles_fewer_columns_than_an_element() {
        for log_columns in 0..PACKED_SHIFT as usize {
            let shape = Shape::new(22 - log_columns, log_columns).unwrap();
            let packed: Vec<Word> = (0..(1 << shape.log_bits()) / BitTable::BITS)
                .map(pseudo_random_word)
                .collect();
            let table = BitTable::new(shape, &packed).unwrap();

            let transposed = table.word_columns().unwrap().transpose();
            assert_eq!(transposed.shape().log_rows(), log_columns);
            assert_eq!(transposed.shape().log_columns(), shape.log_rows());

            for row in [
                0,
                1,
                PACKED_BITS - 1,
                PACKED_BITS,
                shape.rows() / 2,
                shape.rows() - 1,
            ] {
                let bits: Vec<bool> = transposed.column_bits(row).map(|b| b == 1).collect();
                let expected: Vec<bool> = (0..shape.columns())
                    .map(|column| table.bit(column, row))
                    .collect();
                assert_eq!(bits, expected, "s {log_columns}, row {row}");
            }
        }
    }
}
