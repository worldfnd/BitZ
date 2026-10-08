//! The `F128` evaluator against the generic one by table size, on the
//! global pool: run once per thread count, `RAYON_NUM_THREADS=n cargo bench
//! -p poly --bench poly -- f128`. `serial` is what `parallel` has to beat.

use super::common::field_values;
use divan::{Bencher, black_box};
use field::F128;
use poly::{DenseMultilinearExtension, f128};

const LOG_SIZES: &[usize] = &[10, 12, 14, 16, 17, 18, 20, 22, 24];

fn inputs(num_vars: usize) -> (Vec<F128>, Vec<F128>) {
    (field_values(1 << num_vars, 4), field_values(num_vars, 5))
}

#[divan::bench(args = LOG_SIZES)]
fn serial(bencher: Bencher, num_vars: usize) {
    let (table, point) = inputs(num_vars);
    bencher.bench_local(|| black_box(f128::evaluate_serial(black_box(&table), black_box(&point))));
}

#[divan::bench(args = LOG_SIZES)]
fn parallel(bencher: Bencher, num_vars: usize) {
    let (table, point) = inputs(num_vars);
    bencher.bench_local(|| {
        black_box(f128::evaluate_parallel(
            black_box(&table),
            black_box(&point),
        ))
    });
}

#[divan::bench(args = LOG_SIZES)]
fn generic(bencher: Bencher, num_vars: usize) {
    let (table, point) = inputs(num_vars);
    let mle = DenseMultilinearExtension::from_evaluations(num_vars, table).unwrap();
    bencher.bench_local(|| black_box(mle.evaluate(black_box(&point)).unwrap()));
}
