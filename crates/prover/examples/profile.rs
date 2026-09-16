//! Profiling target for `samply`.
//!
//! The `profiling` profile enables debug symbols and uses 16 codegen units.
//! It disables cross-crate LTO; use the release profile for benchmark timings.
//! Build and record with:
//! `cargo build --profile profiling -p prover --example profile && samply record ./target/profiling/examples/profile`

use std::hint::black_box;

use common::{BitTable, BitZParams, Fold, Shape};
use field::F128;
use num_traits::ConstOne;
use transcript::ProverState;

const Q114: u128 = (1 << 114) - 11;
const ITERATIONS: usize = 30;

fn random_table(shape: Shape) -> BitTable<'static> {
    let n = 1 << (shape.log_columns() + shape.log_rows() - 7);
    let packed: Box<Vec<_>> =
        Box::new((0..n).map(|_| F128::from(rand::random::<u128>())).collect());
    let packed: &'static _ = packed.leak();
    let params: BitZParams<Q114> =
        BitZParams::new(shape, field::gf128::smallest_generator()).unwrap();

    params.table(packed).unwrap()
}

fn main() {
    let shape = Shape::new(10, 15).unwrap();
    let table = random_table(shape);

    let row_images = (0..1 << shape.log_rows())
        .map(|_| F128::from(rand::random::<u128>()))
        .collect();
    let zeta = (0..shape.log_columns())
        .map(|_| F128::from(rand::random::<u128>()))
        .collect();

    let folds = vec![0u128; shape.columns()];
    let images = vec![F128::ONE; shape.columns()];
    let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();
    let instance = "blabla";

    for _ in 0..ITERATIONS {
        let transcript = transcript::build_prover("gkr", instance);
        gkr_wrapper(transcript, &fold, table);
    }
}

#[inline(never)]
fn gkr_wrapper(mut transcript: ProverState, fold: &Fold, table: BitTable<'_>) {
    black_box(
        prover::gkr_reduce(&mut transcript, black_box(fold), black_box(&table))
            .expect("profiling fold matches the table shape"),
    );
}
