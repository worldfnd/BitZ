use super::common::{LOG_SIZES, field_values};
use divan::{Bencher, black_box};
use field::{F128, FqDefault};
use poly::eq::{eq_eval, eq_table};

#[divan::bench(args = LOG_SIZES)]
fn eval(bencher: Bencher, num_vars: usize) {
    let x = field_values::<F128>(num_vars, 1);
    let y = field_values::<F128>(num_vars, 2);

    bencher.bench_local(|| {
        black_box(eq_eval(black_box(x.as_slice()), black_box(y.as_slice())).unwrap())
    });
}

#[divan::bench(args = LOG_SIZES)]
fn table_f128(bencher: Bencher, num_vars: usize) {
    let point = field_values::<F128>(num_vars, 3);

    bencher.bench_local(|| black_box(eq_table(black_box(point.as_slice()))));
}

#[divan::bench(args = LOG_SIZES)]
fn table_fq(bencher: Bencher, num_vars: usize) {
    let point = field_values::<FqDefault>(num_vars, 3);

    bencher.bench_local(|| black_box(eq_table(black_box(point.as_slice()))));
}
