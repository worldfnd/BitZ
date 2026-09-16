//! Prover benchmarks.
//!
//! Run with `cargo bench -p prover --bench prover`.

use std::hint::black_box;

use common::{BitTable, BitZParams, Fold, Shape};
use divan::Bencher;
use field::F128;
use num_traits::ConstOne;

fn main() {
    divan::main();
}
// Taken from one of the tests
const Q114: u128 = (1 << 114) - 11;

fn random_table(shape: Shape) -> BitTable<'static> {
    let n = 1 << (shape.log_columns() + shape.log_rows() - 7);
    let packed: Box<Vec<_>> =
        Box::new((0..n).map(|_| F128::from(rand::random::<u128>())).collect());
    let packed: &'static _ = packed.leak();
    let params: BitZParams<Q114> =
        BitZParams::new(shape, field::gf128::smallest_generator()).unwrap();

    params.table(packed).unwrap()
}

#[divan::bench]
fn gkr(bencher: Bencher) {
    // `Shape::new(log_rows, log_columns)` rejects fewer than `2^22` committed
    // bits total (`MIN_LOG_BITS`) and a row width under 7 (`PACK_BITS`), so
    // the two args must sum to at least 22. That floor isn't a security bound
    // computed here -- it's the range flock-core ships precomputed Ligerito
    // configs for (see the comment on `MIN_LOG_BITS`).
    let shape = Shape::new(10, 15).unwrap();
    let table = random_table(shape);

    // Only required data
    let row_images = (0..1 << shape.log_rows())
        .map(|_| F128::from(rand::random::<u128>()))
        .collect();
    let zeta = (0..shape.log_columns())
        .map(|_| F128::from(rand::random::<u128>()))
        .collect();

    // `gkr_reduce` never reads `fold.folds`/`fold.images`, but `Fold::new`
    // still checks their lengths against the shape, so they must be present.
    let folds = vec![0u128; shape.columns()];
    let images = vec![F128::ONE; shape.columns()];
    let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();
    let instance = "blabla";

    bencher
        .with_inputs(|| transcript::build_prover("gkr", instance))
        .bench_values(|mut transcript| {
            black_box(
                prover::gkr_reduce(&mut transcript, black_box(&fold), black_box(&table))
                    .expect("benchmark fold matches the table shape"),
            )
        });
}
