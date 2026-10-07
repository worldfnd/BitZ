//! Bit matrices in memory order, and the transpose between them.

use std::slice::Iter;

/// The word `packed` is sliced into: one `F128` element. [`crate::BitTable::BITS`]
/// is derived from this, so changing it is the only step needed to repack
/// into a different word size of at least 64 bits.
///
/// 128 rather than 64 bits because `bit_transpose` is bound by its strided
/// loads and stores, and each one then moves 16 bytes instead of 8: the same
/// code with `u64` measured 20-55% slower on `BitMatrix::transpose`.
pub(crate) type Word = u128;
/// Bits in a [`Word`].
const BITS: usize = Word::BITS as usize;
/// Bits in a [`Word`]'s bit index.
const LOG_BITS: u32 = Word::BITS.trailing_zeros();
/// Bits in a `u64`'s bit index. Swapping two bit index bits below this never
/// moves a bit from one of a word's `u64` lanes to another.
const LOG_LANE_BITS: u32 = u64::BITS.trailing_zeros();

/// Packed bits read as `xs[dim1][dim2]`: bit `(i1, i2)` is bit
/// `(i1 << log2(dim2)) | i2` of the words, least significant bit first. Each
/// `xs[i1]` is contiguous, so that is what reads fast; [`BitMatrix::transpose`]
/// swaps which axis that is.
///
/// Geometry only: unlike [`crate::BitTable`] it carries no [`crate::Shape`],
/// so no protocol gate applies to it, and an `xs[i1]` shorter than a word
/// shares that word with its neighbours. Only `dim2` is stored; `dim1` is
/// whatever the words hold beyond it, so [`BitMatrix::reshape`] changes one
/// number.
#[derive(Debug, Clone)]
pub struct BitMatrix {
    packed: Vec<Word>,
    log_dim2: usize,
}

impl BitMatrix {
    /// Wraps `packed` as `xs[dim1][2^log_dim2]`. The words must hold a power
    /// of two bits, at least `dim2` of them.
    pub(crate) fn new(packed: Vec<Word>, log_dim2: usize) -> Self {
        let bits = packed.len() * BITS;
        assert!(
            bits.is_power_of_two() && log_dim2 <= bits.ilog2() as usize,
            "{bits} bits do not split into rows of 2^{log_dim2}"
        );
        Self { packed, log_dim2 }
    }

    /// Transposes `xs[dim1][2^log_dim2]` into an owned `[dim2][dim1]`, as
    /// [`BitMatrix::transpose`] does, reading the words where they lie.
    pub(crate) fn transposed(xs: &[Word], log_dim2: usize) -> Self {
        let dim2 = 1 << log_dim2;
        let dim1 = xs.len() * BITS / dim2;
        Self::new(bit_transpose(xs, dim1, dim2), dim1.ilog2() as usize)
    }

    /// The outer, strided axis.
    pub fn dim1(&self) -> usize {
        (self.packed.len() * BITS) >> self.log_dim2
    }

    /// The inner, contiguous axis.
    pub fn dim2(&self) -> usize {
        1 << self.log_dim2
    }

    /// The bit `xs[i1][i2]`.
    pub fn bit(&self, i1: usize, i2: usize) -> bool {
        debug_assert!(i1 < self.dim1(), "i1 {i1} is outside dim1");
        debug_assert!(i2 < self.dim2(), "i2 {i2} is outside dim2");

        let index = (i1 << self.log_dim2) | i2;
        (self.packed[index / BITS] >> (index % BITS)) & 1 == 1
    }

    /// The `dim2` bits of `xs[i1]`, in ascending order.
    pub fn bits(&self, i1: usize) -> BitsIter<'_> {
        let start = i1 << self.log_dim2;
        if self.dim2() >= BITS {
            let words = self.dim2() / BITS;
            return BitsIter::new(self.packed[start / BITS..][..words].iter());
        }
        // Shorter than a word, so it shares one with its neighbours: shift it
        // down to bit 0 and stop after its `dim2` bits.
        BitsIter::preloaded(self.packed[start / BITS] >> (start % BITS), self.dim2())
    }

    /// Swaps the axes, returning an owned copy: `result.bit(i2, i1) ==
    /// self.bit(i1, i2)` for every `i1` and `i2`.
    ///
    /// Built for callers that must walk the strided axis: in place that is a
    /// scatter, one `bit()` call and cache miss per bit. Transposing once up
    /// front turns it into the sequential access [`BitMatrix::bits`] gives.
    ///
    /// # Constraints
    ///
    /// - **`dim2`**: at least `128`, since `bit_transpose` reads each
    ///   `xs[i1]` as whole words. `dim1` is free: below `128` the result's
    ///   `xs[i2]` share words.
    /// - **Allocation**: always a full out-of-place copy -- a fresh buffer
    ///   the same size as the packed words (up to 4 GiB at `m = 35`), never
    ///   a view over the original.
    /// - **Involution**: transposing the result returns the original bits,
    ///   provided `dim1` is at least `128` as well.
    pub fn transpose(&self) -> BitMatrix {
        Self::transposed(&self.packed, self.log_dim2)
    }

    /// The same words read as `xs[dim1][2^log_dim2]`. No bit moves: bit
    /// `(i1 << log_dim2) | i2` stays where it is, only the split between
    /// `dim1` and `dim2` changes. `log_dim2` is at most the words' total
    /// number of index bits.
    pub fn reshape(self, log_dim2: usize) -> Self {
        Self::new(self.packed, log_dim2)
    }

    /// Unwraps the packed bits.
    pub fn into_packed(self) -> Vec<Word> {
        self.packed
    }
}

/// out of place variant.
/// dim1 and dim2 are in bits
/// dim2 is the axis over which the data is adjacent, think xs[dim1][dim2].
/// dim1 < WORD::Bits dim2 >= WORD::Bits
fn bit_transpose(xs: &[Word], dim1: usize, dim2: usize) -> Vec<Word> {
    assert_eq!(xs.len() * BITS, dim1 * dim2);
    assert!(dim1.is_power_of_two(), "dim1 {dim1} is not a power of two");

    // One body, compiled once per block height. With `log2(d)` a runtime
    // value the stage loops cannot be unrolled with constant masks and
    // distances, which cost ~30% at d = 64.
    match dim1.min(BITS).trailing_zeros() {
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
            if offset == BITS {
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

/// Iterator over packed bits `S` at a time, least significant first — see
/// [`BitMatrix::bits`].
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
        (self.remaining as usize + self.elements.len() * BITS) / S
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word with bits spread over its whole width, distinct for every `i`.
    fn pseudo_random_word(i: usize) -> Word {
        (i as Word).wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15) ^ 0xA5A5
    }

    /// A pseudo-random `xs[2^log_dim1][2^log_dim2]`.
    fn pseudo_random_matrix(log_dim1: usize, log_dim2: usize) -> BitMatrix {
        let packed = (0..(1 << (log_dim1 + log_dim2)) / BITS)
            .map(pseudo_random_word)
            .collect();
        BitMatrix::new(packed, log_dim2)
    }

    #[test]
    fn a_bit_lands_where_the_memory_order_says_it_does() {
        // xs[1][130] of an xs[2][256]: bit 256 + 130, which is word 3, bit 2.
        let mut packed = vec![0 as Word; 4];
        packed[3] = 1 << 2;
        let matrix = BitMatrix::new(packed, 8);
        assert_eq!((matrix.dim1(), matrix.dim2()), (2, 256));

        for i1 in 0..matrix.dim1() {
            for i2 in 0..matrix.dim2() {
                assert_eq!(matrix.bit(i1, i2), (i1, i2) == (1, 130), "xs[{i1}][{i2}]");
            }
        }
    }

    /// `dim2` of whole words, and shorter than one, where neighbours share it.
    #[test]
    fn bits_agrees_with_the_bit_accessor() {
        for log_dim2 in 0..=LOG_BITS as usize + 1 {
            let matrix = pseudo_random_matrix(12 - log_dim2, log_dim2);

            for i1 in 0..matrix.dim1() {
                let iter = matrix.bits(i1);
                assert_eq!(iter.len(), matrix.dim2());
                let bits: Vec<bool> = iter.map(|b| b == 1).collect();
                let expected: Vec<bool> = (0..matrix.dim2()).map(|i2| matrix.bit(i1, i2)).collect();
                assert_eq!(bits, expected, "log_dim2 {log_dim2}, i1 {i1}");
            }
        }
    }

    #[test]
    fn bit_block_transpose_matches_a_brute_force_reference() {
        let mut original = [0 as Word; BITS];
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
        let mut block: [Word; BITS] = std::array::from_fn(|p| 1 << p);
        rotate_bit_index::<LOG_D>(&mut block);
        for (p, &word) in block.iter().enumerate() {
            let rotated = ((p << LOG_D) | (p >> (LOG_BITS - LOG_D))) % BITS;
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
            bit_transpose(&tt, GS, words * BITS),
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
        assert_eq!(reference, bit_transpose(&tt, GS, segment_n * BITS));
    }

    /// `xs[512][256]`: two word blocks along each axis, so the block-tiling
    /// loop in `bit_transpose` runs more than once both ways.
    const TRANSPOSE_LOG_DIM1: usize = 9;
    const TRANSPOSE_LOG_DIM2: usize = 8;

    #[test]
    fn transpose_swaps_the_axes() {
        let matrix = pseudo_random_matrix(TRANSPOSE_LOG_DIM1, TRANSPOSE_LOG_DIM2);

        let transposed = matrix.transpose();
        assert_eq!(transposed.dim1(), matrix.dim2());
        assert_eq!(transposed.dim2(), matrix.dim1());

        for i1 in 0..matrix.dim1() {
            for i2 in 0..matrix.dim2() {
                assert_eq!(
                    transposed.bit(i2, i1),
                    matrix.bit(i1, i2),
                    "xs[{i1}][{i2}] vs its transpose"
                );
            }
        }
    }

    #[test]
    fn transposing_twice_recovers_the_original_bits() {
        let matrix = pseudo_random_matrix(TRANSPOSE_LOG_DIM1, TRANSPOSE_LOG_DIM2);

        let roundtripped = matrix.transpose().transpose();

        assert_eq!(roundtripped.dim2(), matrix.dim2());
        assert_eq!(roundtripped.into_packed(), matrix.into_packed());
    }

    /// Every `dim1` below one word, down to one: the result's `xs[i2]` are
    /// shorter than a word and share it.
    #[test]
    fn transpose_handles_dim1_below_a_word() {
        for log_dim1 in 0..LOG_BITS as usize {
            let matrix = pseudo_random_matrix(log_dim1, 15 - log_dim1);

            let transposed = matrix.transpose();
            assert_eq!(transposed.dim1(), matrix.dim2());
            assert_eq!(transposed.dim2(), matrix.dim1());

            for i2 in 0..matrix.dim2() {
                let bits: Vec<bool> = transposed.bits(i2).map(|b| b == 1).collect();
                let expected: Vec<bool> = (0..matrix.dim1()).map(|i1| matrix.bit(i1, i2)).collect();
                assert_eq!(bits, expected, "log_dim1 {log_dim1}, i2 {i2}");
            }
        }
    }

    /// Every split of the index, from one long `xs[0]` down to `dim2 = 1`.
    #[test]
    fn reshape_leaves_every_bit_in_place() {
        let matrix = pseudo_random_matrix(TRANSPOSE_LOG_DIM1, TRANSPOSE_LOG_DIM2);
        let log_bits = TRANSPOSE_LOG_DIM1 + TRANSPOSE_LOG_DIM2;

        for log_dim2 in 0..=log_bits {
            let reshaped = matrix.clone().reshape(log_dim2);
            assert_eq!(reshaped.dim1(), 1 << (log_bits - log_dim2));
            assert_eq!(reshaped.dim2(), 1 << log_dim2);

            for index in 0..1 << log_bits {
                assert_eq!(
                    reshaped.bit(index >> log_dim2, index & (reshaped.dim2() - 1)),
                    matrix.bit(index >> TRANSPOSE_LOG_DIM2, index & (matrix.dim2() - 1)),
                    "log_dim2 {log_dim2}, bit {index}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "do not split")]
    fn reshape_rejects_a_dim2_longer_than_the_words() {
        let matrix = pseudo_random_matrix(TRANSPOSE_LOG_DIM1, TRANSPOSE_LOG_DIM2);
        matrix.reshape(TRANSPOSE_LOG_DIM1 + TRANSPOSE_LOG_DIM2 + 1);
    }
}
