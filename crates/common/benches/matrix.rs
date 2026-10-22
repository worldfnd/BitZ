//! Compare the blocked `bit_transpose` with the bit-at-a-time reference.
//!
//! Run with `cargo bench -p common --features bench --bench matrix`.
//! Both transpose a `DIM1 x DIM2`-bit matrix (`dim2` contiguous), including
//! output allocation. Input generation runs outside the measured loop.

use common::matrix::bench::{bit_transpose, transpose_reference};
use divan::counter::BytesCount;
use divan::{Bencher, black_box};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// The table as `GrandProductCircuit::new` transposes it, `[columns][rows]`,
/// at the CLI's `2^20` columns of `128` rows: a 16 MiB matrix, well past the
/// last-level cache.
const DIM1: usize = 1 << 20;
const DIM2: usize = 128;

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
fn input() -> Vec<u128> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..DIM1 * DIM2 / u128::BITS as usize)
        .map(|_| rng.random())
        .collect()
}

#[divan::bench]
fn blocked(bencher: Bencher) {
    let xs = input();
    bencher
        .counter(BytesCount::of_slice(&xs))
        .bench(|| bit_transpose(black_box(&xs), DIM1, DIM2));
}

#[divan::bench]
fn reference(bencher: Bencher) {
    let xs = input();
    bencher
        .counter(BytesCount::of_slice(&xs))
        .bench(|| transpose_reference::<DIM1>(black_box(&xs)));
}
