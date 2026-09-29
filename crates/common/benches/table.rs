//! Compare the blocked `bit_transpose` with the bit-at-a-time reference.
//!
//! Run with `cargo bench -p common --features bench --bench table`.
//! Both transpose a `128 x DIM2`-bit matrix (`dim2` contiguous), including
//! output allocation. Input generation runs outside the measured loop.

use common::table::bench::{bit_transpose, transpose_reference};
use divan::counter::BytesCount;
use divan::{Bencher, black_box};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

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

/// Seeded so every run transposes the same matrix.
fn input() -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..DIM1 * DIM2 / u64::BITS as usize)
        .map(|_| rng.random())
        .collect()
}

fn blocked(bencher: Bencher) {
    let xs = input();
    bencher
        .counter(BytesCount::of_slice(&xs))
        .bench(|| bit_transpose(black_box(&xs), DIM1, DIM2));
}

fn reference(bencher: Bencher) {
    let xs = input();
    bencher
        .counter(BytesCount::of_slice(&xs))
        .bench(|| transpose_reference::<DIM1>(black_box(&xs)));
}
