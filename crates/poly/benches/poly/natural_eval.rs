use super::common::{
    GENERAL_NATURAL_SIZES, LOG_SIZES, POWER_OF_TWO_NATURAL_SIZES, PROTOCOL_NATURAL_SIZES,
    field_values,
};
use divan::{Bencher, black_box};
use field::F128;
use poly::nat_evaluation::{LagrangeInterpolationDomain, NatEvaluatedPoly};

fn bench_fresh_domain(bencher: Bencher, len: usize) {
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

#[divan::bench(args = POWER_OF_TWO_NATURAL_SIZES)]
fn fresh_power_of_two_domain(bencher: Bencher, len: usize) {
    bench_fresh_domain(bencher, len);
}

#[divan::bench(args = GENERAL_NATURAL_SIZES)]
fn fresh_general_domain(bencher: Bencher, len: usize) {
    bench_fresh_domain(bencher, len);
}

#[divan::bench(args = PROTOCOL_NATURAL_SIZES)]
fn protocol_sized_reused_domain(bencher: Bencher, len: usize) {
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

/// Synthetic scalability benchmark; BitZ's current round polynomials have
/// lengths 3, 4, or 6 rather than multilinear-table sizes.
#[divan::bench(args = LOG_SIZES)]
fn synthetic_large_reused_domain(bencher: Bencher, num_vars: usize) {
    let len = 1usize << num_vars;
    let polynomial = NatEvaluatedPoly::new(field_values(len, 6));
    // Power-of-two domains use the library's linear additive-subspace path;
    // setup remains outside the timed evaluation kernel.
    let domain = LagrangeInterpolationDomain::new(len);
    let point = F128::from(u128::MAX);

    bencher.bench_local(|| {
        black_box(
            polynomial
                .evaluate_at_point_with_domain(black_box(point), black_box(&domain))
                .expect("the domain matches the nonempty polynomial"),
        )
    });
}
