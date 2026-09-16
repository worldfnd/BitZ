//! Bit transpose for the ring-switch claim's 128 polynomial-basis rows.

use flock_core::field::F128;
use flock_core::pcs::LOG_PACKING;

const _: () = assert!(1 << LOG_PACKING == 128);

/// Returns rows with `output[column].bit(row) == input[row].bit(column)`.
///
/// Seven block-swap stages exchange the row and column index bits.
/// The first stage exchanges whole words across the four 64×64 quadrants.
/// The remaining stages transpose each quadrant with masks and shifts on `u64` words.
/// All loop bounds and memory addresses are independent of the input bits.
pub(crate) fn tensor_algebra_transpose(input: &[F128]) -> Vec<F128> {
    assert_eq!(input.len(), 128);
    let mut output = input.to_vec();

    let (top, bottom) = output.split_at_mut(64);
    for (top, bottom) in top.iter_mut().zip(bottom) {
        core::mem::swap(&mut top.hi, &mut bottom.lo);
    }

    swap_blocks::<32>(&mut output, 0x0000_0000_ffff_ffff);
    swap_blocks::<16>(&mut output, 0x0000_ffff_0000_ffff);
    swap_blocks::<8>(&mut output, 0x00ff_00ff_00ff_00ff);
    swap_blocks::<4>(&mut output, 0x0f0f_0f0f_0f0f_0f0f);
    swap_blocks::<2>(&mut output, 0x3333_3333_3333_3333);
    swap_blocks::<1>(&mut output, 0x5555_5555_5555_5555);
    output
}

/// Exchanges off-diagonal blocks of width `SHIFT` in each 64×64 quadrant.
#[inline]
fn swap_blocks<const SHIFT: usize>(rows: &mut [F128], mask: u64) {
    for block in rows.chunks_exact_mut(2 * SHIFT) {
        let (top, bottom) = block.split_at_mut(SHIFT);
        for (top, bottom) in top.iter_mut().zip(bottom) {
            // Move the top row's high block into the bottom row's low block.
            let lo = ((top.lo >> SHIFT) ^ bottom.lo) & mask;
            top.lo ^= lo << SHIFT;
            bottom.lo ^= lo;

            let hi = ((top.hi >> SHIFT) ^ bottom.hi) & mask;
            top.hi ^= hi << SHIFT;
            bottom.hi ^= hi;
        }
    }
}
