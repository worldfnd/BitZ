//! Bit matrices in memory order, and the transpose between them.

use std::{
    borrow::Cow,
    ops::{BitAnd, BitXor, BitXorAssign, Deref, Not, Shl, Shr},
    slice::Iter,
};

/// The word `packed` is sliced into: one `F128` element. [`crate::BitTable::BITS`]
/// is derived from this, so changing it is the only step needed to repack
/// into a different word size of at least 64 bits.
///
/// 128 rather than 64 bits because `bit_transpose` is bound by its strided
/// loads and stores, and each one then moves 16 bytes instead of 8: the same
/// code with `u64` measured 20-55% slower on `BitMatrix::transpose`. It
/// drops to a narrower word only for an axis shorter than this.
pub(crate) type Word = u128;
/// Bits in a [`Word`].
const BITS: usize = Word::BITS as usize;

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
///
/// The words are owned or borrowed: [`crate::BitTable::as_matrix`] reads the
/// table's words where they lie, and [`BitMatrix::transpose`] always returns
/// an owned copy.
#[derive(Debug, Clone)]
pub struct BitMatrix<'a> {
    packed: Cow<'a, [Word]>,
    log_dim2: usize,
}

impl<'a> BitMatrix<'a> {
    /// Wraps `packed` as `xs[dim1][2^log_dim2]`. The words must hold a power
    /// of two bits, at least `dim2` of them.
    pub(crate) fn new(packed: impl Into<Cow<'a, [Word]>>, log_dim2: usize) -> Self {
        let packed = packed.into();
        let bits = packed.len() * BITS;
        // Compared as logs: `1 << log_dim2` overflows from `usize::BITS` on,
        // and a release build masks the shift instead of panicking.
        assert!(
            bits.is_power_of_two() && log_dim2 <= bits.ilog2() as usize,
            "{bits} bits do not split into rows of 2^{log_dim2}"
        );
        Self { packed, log_dim2 }
    }

    /// Swaps the axes, returning an owned copy: bit `(i2, i1)` of the result
    /// is bit `(i1, i2)` of `self` for every `i1` and `i2`.
    ///
    /// # Constraints
    ///
    /// - **Both axes**: at least `8`, since `bit_transpose` moves square
    ///   blocks of the widest word both axes fill, `u128` down to `u8`. Its
    ///   doc comment says where a version for a shorter `dim1` lives.
    /// - **Allocation**: always a full out-of-place copy -- a fresh owned
    ///   buffer the same size as the packed words (up to 4 GiB at `m = 35`),
    ///   whether `self` owned or borrowed them.
    /// - **Involution**: transposing the result returns the original bits.
    pub fn transpose(&self) -> BitMatrix<'static> {
        let (dim1, dim2) = (self.dim1(), self.dim2());
        // `dim1` is a power of two, so its trailing zeros are its log.
        let log_dim1 = dim1.trailing_zeros() as usize;
        BitMatrix::new(bit_transpose(&self.packed, dim1, dim2), log_dim1)
    }

    /// The outer, strided axis.
    pub fn dim1(&self) -> usize {
        (self.packed.len() * BITS) >> self.log_dim2
    }

    /// The inner, contiguous axis.
    pub fn dim2(&self) -> usize {
        1 << self.log_dim2
    }

    pub fn log_dim2(&self) -> usize {
        self.log_dim2
    }

    /// The `dim2` bits of `xs[i1]`, in ascending order.
    ///
    /// # Panics
    ///
    /// If `dim2` is below `128`: `xs[i1]` then shares a word with its
    /// neighbours.
    pub fn bits(&self, i1: usize) -> BitsIter<'_> {
        assert!(
            self.dim2() >= BITS,
            "xs[{i1}] of {} bits shares a word",
            self.dim2()
        );
        let words = self.dim2() / BITS;
        BitsIter::new(self.packed[i1 * words..(i1 + 1) * words].iter())
    }

    /// The same words read as `xs[dim1][2^log_dim2]`. No bit moves: bit
    /// `(i1 << log_dim2) | i2` stays where it is, only the split between
    /// `dim1` and `dim2` changes. `log_dim2` is at most the words' total
    /// number of index bits.
    pub fn reshape(self, log_dim2: usize) -> Self {
        Self::new(self.packed, log_dim2)
    }

    /// Unwraps the packed bits, copying them if they were borrowed.
    pub fn into_packed(self) -> Vec<Word> {
        self.packed.into_owned()
    }
}

/// Transposes `xs[dim1][dim2]`, `dim2` contiguous and both in bits, into an
/// owned `[dim2][dim1]`, one square block at a time.
///
/// The block is the widest word both axes are whole multiples of: a [`Word`]
/// whenever they can be, otherwise the words are read in place as `u64` down
/// to `u8` and the same kernel runs at that width. An axis of 1, 2 or 4 bits
/// fills no word. For a short `dim1`, commit
/// 336e8097225aca4f8146f35594740b0e5054d839 has a version that reaches those
/// too (`bit_transpose_blocks` and `rotate_bit_index` in this file): it
/// gathers `BITS / dim1` words of each row into one block and interleaves
/// them by rotating the bit index.
fn bit_transpose(xs: &[Word], dim1: usize, dim2: usize) -> Vec<Word> {
    assert_eq!(xs.len() * BITS, dim1 * dim2);

    let mut out = bytemuck::zeroed_vec(xs.len());
    // The lowest set bit of `dim1 | dim2` is the largest power of two both
    // are multiples of; or-ing in `BITS` caps it at a `Word`.
    match 1usize << (dim1 | dim2 | BITS).trailing_zeros() {
        8 => transpose_blocks::<u8, 8>(xs, &mut out, dim1, dim2),
        16 => transpose_blocks::<u16, 16>(xs, &mut out, dim1, dim2),
        32 => transpose_blocks::<u32, 32>(xs, &mut out, dim1, dim2),
        64 => transpose_blocks::<u64, 64>(xs, &mut out, dim1, dim2),
        BITS => transpose_blocks::<Word, BITS>(xs, &mut out, dim1, dim2),
        _ => panic!("{dim1} x {dim2} bits is not whole words along both axes"),
    }

    out
}

/// [`bit_transpose`] in `N x N` blocks of the `N`-bit word `W`.
fn transpose_blocks<W: BlockWord, const N: usize>(
    xs: &[Word],
    out: &mut [Word],
    dim1: usize,
    dim2: usize,
) {
    const { assert!(N == 8 * size_of::<W>()) };
    // Read in place: on a little-endian target, bit `i` of the words is bit
    // `i` of the narrower words too.
    const { assert!(cfg!(target_endian = "little")) };
    let xs: &[W] = bytemuck::cast_slice(xs);
    let out: &mut [W] = bytemuck::cast_slice_mut(out);
    let dim1_words = dim1 / N;
    let dim2_words = dim2 / N;

    let mut block = [W::zeroed(); N];
    // g1 innermost, so consecutive blocks write neighbouring words of the
    // same output rows while those lines are still cached, and it is the
    // input that is revisited after a sweep; the other way round measured
    // ~10% slower on a 2^13-column table.
    for g2 in 0..dim2_words {
        for g1 in 0..dim1_words {
            for (a, word) in block.iter_mut().enumerate() {
                *word = xs[(g1 * N + a) * dim2_words + g2];
            }

            transpose_bit_block(&mut block);

            for (o, &word) in block.iter().enumerate() {
                out[(g2 * N + o) * dim1_words + g1] = word;
            }
        }
    }
}

/// An unsigned word [`transpose_bit_block`] can work in.
trait BlockWord:
    bytemuck::Pod
    + BitAnd<Output = Self>
    + BitXor<Output = Self>
    + BitXorAssign
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
{
}

impl BlockWord for u8 {}
impl BlockWord for u16 {}
impl BlockWord for u32 {}
impl BlockWord for u64 {}
impl BlockWord for u128 {}

/// Transposes an `N x N` bit matrix stored as `N` words of `N` bits: after
/// the call, bit `j` of word `i` is what bit `i` of word `j` held before it,
/// for every `i, j`.
///
/// The recursive-doubling bit-matrix transpose (Hacker's Delight, 2nd ed.,
/// §7-3), generalized from its usual 64x64 form to any word width, `u128`
/// being the lane width `F128` packs bits into. `O(n log n)` word operations
/// rather than the `O(n^2)` of moving one bit at a time.
fn transpose_bit_block<W: BlockWord, const N: usize>(a: &mut [W; N]) {
    let mut j = N / 2;
    // The low half of the word.
    let mut mask = !W::zeroed() >> j;
    while j > 0 {
        let mut k = 0;
        while k < N {
            let t = ((a[k] >> j) ^ a[k | j]) & mask;
            a[k | j] ^= t;
            a[k] ^= t << j;
            k = ((k | j) + 1) & !j;
        }
        j >>= 1;
        mask ^= mask << j;
    }
}

/// Bit-at-a-time transpose of `GS` equal segments of `tt`: output bit `i`
/// is bit `i / GS` of segment `i % GS`. The oracle [`bit_transpose`] is
/// checked (and benchmarked) against.
#[cfg(any(test, feature = "bench"))]
fn transpose_reference<const GS: usize>(tt: &[Word]) -> Vec<Word> {
    assert_eq!(tt.len() % GS, 0);
    let segment_n = tt.len() / GS;
    // On the heap: the benchmark's `GS` is `2^20`, too many iterators for the
    // stack.
    let mut segments: Vec<_> = (0..GS)
        .map(|i| BitsIter::new(tt[i * segment_n..(i + 1) * segment_n].iter()))
        .collect();

    let mut out = Vec::with_capacity(tt.len());
    let mut word: Word = 0;
    let mut offset = 0;
    'outer: loop {
        for s in &mut segments {
            let Some(b) = s.next() else { break 'outer };
            // LSB first, like `StepIter`.
            word |= Word::from(b) << offset;
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

/// Benchmark-only access to the private transposes; see `benches/matrix.rs`.
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
///
/// Reads the words as `u64`, since no step needs more. A `u128` word was
/// kept on the stack between steps and made the GKR's layer 3 ~25% slower.
pub struct StepIter<'a, const S: usize> {
    elements: std::slice::Iter<'a, u64>,
    /// Bits not yet emitted; the next one to emit is the LSB.
    word: u64,
    remaining: u8,
}

impl<'a, const S: usize> StepIter<'a, S> {
    const CHECK_SIZE: () = assert!(
        S != 0 && S <= u64::BITS as usize && (u64::BITS as usize).is_multiple_of(S),
        "step size must be nonzero, at most 64, and divide 64 evenly"
    );
    const MASK: u64 = 1u64.unbounded_shl(S as u32).wrapping_sub(1);
    pub fn new(iter: Iter<'a, Word>) -> Self {
        let () = Self::CHECK_SIZE;
        // On a little-endian target, bit `i` of the words is bit `i` of the
        // `u64`s too.
        const { assert!(cfg!(target_endian = "little")) };
        let elements: &[u64] = bytemuck::cast_slice(iter.as_slice());
        Self {
            elements: elements.iter(),
            word: 0,
            remaining: 0,
        }
    }
}

impl<const S: usize> Iterator for StepIter<'_, S> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            let element = self.elements.next()?;
            self.word = *element;
            self.remaining = u64::BITS as u8;
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
        (self.remaining as usize + self.elements.len() * u64::BITS as usize) / S
    }
}

impl<'a> Deref for BitMatrix<'a> {
    type Target = [Word];

    fn deref(&self) -> &Self::Target {
        &self.packed
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
    fn pseudo_random_matrix(log_dim1: usize, log_dim2: usize) -> BitMatrix<'static> {
        let packed: Vec<Word> = (0..(1 << (log_dim1 + log_dim2)) / BITS)
            .map(pseudo_random_word)
            .collect();
        BitMatrix::new(packed, log_dim2)
    }

    /// The bit `xs[i1][i2]`, read on its own: the oracle the word-level
    /// reads and the transpose are checked against.
    fn bit(matrix: &BitMatrix, i1: usize, i2: usize) -> bool {
        assert!(i1 < matrix.dim1(), "i1 {i1} is outside dim1");
        assert!(i2 < matrix.dim2(), "i2 {i2} is outside dim2");

        let index = (i1 << matrix.log_dim2) | i2;
        (matrix.packed[index / BITS] >> (index % BITS)) & 1 == 1
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
                assert_eq!(bit(&matrix, i1, i2), (i1, i2) == (1, 130), "xs[{i1}][{i2}]");
            }
        }
    }

    /// `dim2` of one word and of two.
    #[test]
    fn bits_agrees_with_the_bit_accessor() {
        for log_dim2 in [7, 8] {
            let matrix = pseudo_random_matrix(12 - log_dim2, log_dim2);

            for i1 in 0..matrix.dim1() {
                let iter = matrix.bits(i1);
                assert_eq!(iter.len(), matrix.dim2());
                let bits: Vec<bool> = iter.map(|b| b == 1).collect();
                let expected: Vec<bool> =
                    (0..matrix.dim2()).map(|i2| bit(&matrix, i1, i2)).collect();
                assert_eq!(bits, expected, "log_dim2 {log_dim2}, i1 {i1}");
            }
        }
    }

    #[test]
    fn transpose_bit_block_matches_a_brute_force_reference() {
        let mut original = [0 as Word; BITS];
        for (i, word) in original.iter_mut().enumerate() {
            *word = pseudo_random_word(i);
        }

        let mut transposed = original;
        transpose_bit_block(&mut transposed);

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
        let tt: Vec<Word> = (0..GS * words).map(pseudo_random_word).collect();
        assert_eq!(
            bit_transpose(&tt, GS, words * BITS),
            transpose_reference::<GS>(&tt),
            "dim1 {GS}, {words} words per row"
        );
    }

    #[test]
    fn bit_transpose_matches_the_reference() {
        // One block, then two along each axis and along both, so block
        // placement is exercised as well as the block contents.
        check_bit_transpose::<128>(1);
        check_bit_transpose::<128>(2);
        check_bit_transpose::<256>(1);
        check_bit_transpose::<256>(2);
    }

    /// `xs[512][256]`: two word blocks along each axis, so the block-tiling
    /// loop in `bit_transpose` runs more than once both ways.
    const TRANSPOSE_LOG_DIM1: usize = 9;
    const TRANSPOSE_LOG_DIM2: usize = 8;

    /// The transpose of a pseudo-random `xs[2^log_dim1][2^log_dim2]` against
    /// the bit accessor.
    fn check_transpose(log_dim1: usize, log_dim2: usize) {
        let matrix = pseudo_random_matrix(log_dim1, log_dim2);

        let transposed = matrix.transpose();
        assert_eq!(transposed.dim1(), matrix.dim2());
        assert_eq!(transposed.dim2(), matrix.dim1());

        for i1 in 0..matrix.dim1() {
            for i2 in 0..matrix.dim2() {
                assert_eq!(
                    bit(&transposed, i2, i1),
                    bit(&matrix, i1, i2),
                    "xs[{i1}][{i2}] vs its transpose, {log_dim1} x {log_dim2}"
                );
            }
        }
    }

    #[test]
    fn transpose_swaps_the_axes() {
        check_transpose(TRANSPOSE_LOG_DIM1, TRANSPOSE_LOG_DIM2);
    }

    /// An axis of 8, 16, 32 and 64 bits, so each narrower block word runs,
    /// with the short axis on either side.
    #[test]
    fn transpose_swaps_the_axes_of_a_short_axis() {
        for log_short in 3..7 {
            check_transpose(log_short, 9);
            check_transpose(9, log_short);
        }
    }

    #[test]
    #[should_panic(expected = "whole words")]
    fn transpose_rejects_an_axis_below_a_byte() {
        pseudo_random_matrix(2, 10).transpose();
    }

    #[test]
    fn transposing_twice_recovers_the_original_bits() {
        let matrix = pseudo_random_matrix(TRANSPOSE_LOG_DIM1, TRANSPOSE_LOG_DIM2);

        let roundtripped = matrix.transpose().transpose();

        assert_eq!(roundtripped.dim2(), matrix.dim2());
        assert_eq!(roundtripped.into_packed(), matrix.into_packed());
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
                    bit(&reshaped, index >> log_dim2, index & (reshaped.dim2() - 1)),
                    bit(
                        &matrix,
                        index >> TRANSPOSE_LOG_DIM2,
                        index & (matrix.dim2() - 1)
                    ),
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
