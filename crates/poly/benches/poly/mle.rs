use super::common::{LOG_SIZES, field_values};
use divan::{Bencher, black_box};
use field::F128;
use poly::mle::DenseMultilinearExtension;

#[divan::bench(args = LOG_SIZES)]
fn evaluate(bencher: Bencher, num_vars: usize) {
    let evaluations = field_values::<F128>(1usize << num_vars, 4);
    let point = field_values::<F128>(num_vars, 5);
    let mle = DenseMultilinearExtension::from_evaluations(num_vars, evaluations).unwrap();

    bencher.bench_local(|| {
        black_box(
            mle.evaluate(black_box(point.as_slice()))
                .expect("the point has the MLE's width"),
        )
    });
}

#[divan::bench(args = LOG_SIZES)]
fn fold_one_round(bencher: Bencher, num_vars: usize) {
    let evaluations = field_values::<F128>(1usize << num_vars, 4);
    let challenge = field_values::<F128>(1, 5);

    bencher
        .with_inputs(|| {
            DenseMultilinearExtension::from_evaluations(num_vars, evaluations.clone()).unwrap()
        })
        .bench_local_refs(|mle| {
            mle.fold(black_box(challenge.as_slice()))
                .expect("one challenge fits a nonempty MLE");
            black_box(mle[0])
        });
}
