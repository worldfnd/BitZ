//! Compare the pinned Flock transpose with the local transpose.
//!
//! Run with `cargo bench -p pcs --bench tensor_algebra_transpose`.
//! Each item is one 128-by-128-bit transpose, including output allocation and destruction.
//! Divan reports elapsed time per batch and throughput in transposes per second.
//! Divide batch time by its item count to obtain time per transpose.
//!
//! `DenseSingle` repeatedly uses one matrix and can train the reference's branch predictor.
//! `DenseBatch64` uses 64 independent matrices to reduce that effect.
//! These batch sizes control microbenchmark sampling, not the number of calls in one PCS opening.
//! Input generation and correctness checks run outside the measured loop.

use divan::counter::ItemsCount;
use divan::{Bencher, black_box};
use flock_core::field::F128;
use flock_core::pcs::ring_switch::tensor_algebra_transpose as flock_transpose;

#[path = "../src/transpose.rs"]
mod transpose;

const WIDTH: usize = 128;
const BATCH_SIZE: usize = 64;
const CASES: &[Case] = &[
    Case::DenseSingle,
    Case::DenseBatch64,
    Case::Zero,
    Case::AllBitsOne,
    Case::SparseBatch64,
];

#[derive(Clone, Copy, Debug)]
enum Case {
    DenseSingle,
    DenseBatch64,
    Zero,
    AllBitsOne,
    SparseBatch64,
}

fn main() {
    println!("One item = one transpose, including output allocation and destruction.");
    println!("Batch64 cases contain 64 items; other cases contain one item.");
    divan::main();
}

/// SplitMix64 supplies deterministic mixed bits across both field words.
fn next_word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut word = *state;
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

fn matrices(case: Case) -> Vec<Vec<F128>> {
    let count = match case {
        Case::DenseBatch64 | Case::SparseBatch64 => BATCH_SIZE,
        _ => 1,
    };
    let mut state = 0x243f_6a88_85a3_08d3;
    (0..count)
        .map(|_| {
            (0..WIDTH)
                .map(|_| match case {
                    Case::DenseSingle | Case::DenseBatch64 => {
                        F128::new(next_word(&mut state), next_word(&mut state))
                    }
                    Case::Zero => F128::ZERO,
                    Case::AllBitsOne => F128::new(u64::MAX, u64::MAX),
                    // Each row contains one set bit at a deterministic random position.
                    Case::SparseBatch64 => {
                        let bit = next_word(&mut state) as usize % WIDTH;
                        if bit < 64 {
                            F128::new(1u64 << bit, 0)
                        } else {
                            F128::new(0, 1u64 << (bit - 64))
                        }
                    }
                })
                .collect()
        })
        .collect()
}

fn measure(bencher: Bencher, case: Case, implementation: impl Fn(&[F128]) -> Vec<F128>) {
    let inputs = matrices(case);
    for input in &inputs {
        assert_eq!(
            transpose::tensor_algebra_transpose(input),
            flock_transpose(input),
            "implementations disagree on {case:?}"
        );
    }

    bencher
        .counter(ItemsCount::new(inputs.len()))
        .bench_local(|| {
            for input in &inputs {
                drop(black_box(implementation(black_box(input.as_slice()))));
            }
        });
}

#[divan::bench(args = CASES)]
fn flock_reference(bencher: Bencher, case: Case) {
    measure(bencher, case, flock_transpose);
}

#[divan::bench(args = CASES)]
fn local(bencher: Bencher, case: Case) {
    measure(bencher, case, transpose::tensor_algebra_transpose);
}
