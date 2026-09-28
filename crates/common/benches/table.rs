//! Compare the blocked `bit_transpose` with the bit-at-a-time reference.
//!
//! Run with `cargo bench -p common --features bench --bench table`.
//! Both transpose a `128 x DIM2`-bit matrix (`dim2` contiguous), including
//! output allocation. Input generation runs outside the measured loop.

use common::table::bench::{bit_transpose, transpose_reference};
use divan::counter::BytesCount;
use divan::{Bencher, black_box};

const DIM1: usize = 128;
/// `2^20` bits per row: a 16 MiB matrix, well past the last-level cache.
const DIM2: usize = 1 << 20;

fn main() {
    let xs = input();
    assert_eq!(
        bit_transpose(&xs, DIM1, DIM2),
        transpose_reference::<DIM1>(&xs),
        "bit_transpose disagrees with the reference"
    );
    divan::main();
}

/// SplitMix64 for deterministic, well-mixed bits.
fn input() -> Vec<u64> {
    let mut state = 0u64;
    (0..DIM1 * DIM2 / u64::BITS as usize)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut word = state;
            word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            word ^ (word >> 31)
        })
        .collect()
}

#[divan::bench(sample_count = 20)]
fn blocked(bencher: Bencher) {
    let xs = input();
    bencher
        .counter(BytesCount::of_slice(&xs))
        .bench(|| bit_transpose(black_box(&xs), DIM1, DIM2));
}

#[divan::bench(sample_count = 20)]
fn reference(bencher: Bencher) {
    let xs = input();
    bencher
        .counter(BytesCount::of_slice(&xs))
        .bench(|| transpose_reference::<DIM1>(black_box(&xs)));
}
