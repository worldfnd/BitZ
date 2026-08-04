use super::common::{NATURAL_SIZES, field_values};
use divan::{Bencher, black_box};
use field::F128;
use poly::nat_evaluation::{LagrangeInterpolationDomain, NatEvaluatedPoly};

#[divan::bench(args = NATURAL_SIZES)]
fn fresh_domain(bencher: Bencher, len: usize) {
    let polynomial = NatEvaluatedPoly::new(field_values(len, 6));
    let point = F128::from(len as u128 + 1);

    bencher.bench_local(|| {
        black_box(
            polynomial
                .evaluate_at_point(black_box(point))
                .expect("the polynomial is nonempty"),
        )
    });
}

#[divan::bench(args = NATURAL_SIZES)]
fn reused_domain(bencher: Bencher, len: usize) {
    let polynomial = NatEvaluatedPoly::new(field_values(len, 6));
    let domain = LagrangeInterpolationDomain::new(len);
    let point = F128::from(len as u128 + 1);

    bencher.bench_local(|| {
        black_box(
            polynomial
                .evaluate_at_point_with_domain(black_box(point), black_box(&domain))
                .expect("the domain matches the nonempty polynomial"),
        )
    });
}
