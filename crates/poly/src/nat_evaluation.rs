// SPDX-License-Identifier: Apache-2.0

use field::F128;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::parallel::workload_size;

/// Reusable interpolation data for evaluations on the natural `F128` domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LagrangeInterpolationDomain {
    /// `points[i] = F128::from(i as u128)`.
    pub points: Vec<F128>,
    /// `w_i = (∏_{j != i} (points[i] - points[j]))⁻¹`
    pub weights: Vec<F128>,
}

impl LagrangeInterpolationDomain {
    /// Precomputes the interpolation nodes and their barycentric weights.
    pub fn new(len: usize) -> Self {
        let points: Vec<_> = (0..len).map(|i| F128::from(i as u128)).collect();

        // Power-of-two natural domains are additive subspaces, so all weights
        // share one denominator. Adapted from Binius64 (MIT OR Apache-2.0):
        // https://github.com/IrreducibleOSS/binius64/blob/49deecec1bf691c57aeadcda2499316d8094fcd8/crates/math/src/univariate.rs#L25-L75
        if len.is_power_of_two() {
            let denominator = points[1..]
                .iter()
                .copied()
                .fold(F128::ONE, |product, node| product * node);
            let weight = denominator
                .inverse()
                .expect("the product of nonzero domain points is nonzero");

            return Self {
                weights: vec![weight; len],
                points,
            };
        }

        #[cfg(feature = "parallel")]
        // Require enough work for at least two cache-sized tasks.
        let parallel_workload = workload_size::<F128>().saturating_mul(2);
        #[cfg(feature = "parallel")]
        let mut weights: Vec<F128> = if len.saturating_mul(len.saturating_sub(1))
            > parallel_workload
            && rayon::current_num_threads() > 1
        {
            let min_rows = (workload_size::<F128>() / len).max(1);
            (0..len)
                .into_par_iter()
                .with_min_len(min_rows)
                .map(|i| lagrange_denominator(&points, i))
                .collect()
        } else {
            (0..len).map(|i| lagrange_denominator(&points, i)).collect()
        };

        #[cfg(not(feature = "parallel"))]
        let mut weights: Vec<F128> = (0..len).map(|i| lagrange_denominator(&points, i)).collect();

        batch_invert_nonzero(&mut weights);

        Self { points, weights }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NatEvaluatedPoly {
    evaluations: Vec<F128>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NatEvaluationError {
    EmptyPolynomial,
    DomainSizeMismatch,
}

impl NatEvaluatedPoly {
    pub const fn new(evaluations: Vec<F128>) -> Self {
        Self { evaluations }
    }

    /// Evaluates using a freshly constructed interpolation domain.
    pub fn evaluate_at_point(&self, point: F128) -> Result<F128, NatEvaluationError> {
        let domain = LagrangeInterpolationDomain::new(self.evaluations.len());
        self.evaluate_at_point_with_domain(point, &domain)
    }

    /// Linear, allocation-free adaptation of Binius64's
    /// `EvaluationDomain::extrapolate` (Apache-2.0):
    /// <https://github.com/IrreducibleOSS/binius64/blob/49deecec1bf691c57aeadcda2499316d8094fcd8/crates/math/src/univariate.rs#L200-L216>
    pub fn evaluate_at_point_with_domain(
        &self,
        point: F128,
        domain: &LagrangeInterpolationDomain,
    ) -> Result<F128, NatEvaluationError> {
        let len = self.evaluations.len();
        if len == 0 {
            return Err(NatEvaluationError::EmptyPolynomial);
        } else if len != domain.points.len() || len != domain.weights.len() {
            return Err(NatEvaluationError::DomainSizeMismatch);
        }

        Ok(evaluate_block(&self.evaluations, &domain.points, &domain.weights, point).0)
    }
}

fn lagrange_denominator(points: &[F128], i: usize) -> F128 {
    let point = points[i];
    points[..i]
        .iter()
        .chain(&points[i + 1..])
        .fold(F128::ONE, |denominator, &node| denominator * (point - node))
}

/// Evaluates one contiguous block and returns its `(result, product)` summary.
/// Adjacent summaries compose as
/// `(r_l p_r + p_l r_r, p_l p_r)`, so independent halves can be evaluated
/// concurrently without division or allocation.
fn evaluate_block(
    evaluations: &[F128],
    nodes: &[F128],
    weights: &[F128],
    point: F128,
) -> (F128, F128) {
    debug_assert_eq!(evaluations.len(), nodes.len());
    debug_assert_eq!(evaluations.len(), weights.len());

    #[cfg(feature = "parallel")]
    // Each term reads one evaluation, node, and weight.
    if evaluations.len().saturating_mul(3) > workload_size::<F128>()
        && rayon::current_num_threads() > 1
    {
        let mid = evaluations.len() / 2;
        let (left_evaluations, right_evaluations) = evaluations.split_at(mid);
        let (left_nodes, right_nodes) = nodes.split_at(mid);
        let (left_weights, right_weights) = weights.split_at(mid);

        let (left, right) = rayon::join(
            || evaluate_block(left_evaluations, left_nodes, left_weights, point),
            || evaluate_block(right_evaluations, right_nodes, right_weights, point),
        );

        return combine_evaluation_blocks(left, right);
    }

    evaluate_block_serial(evaluations, nodes, weights, point)
}

fn evaluate_block_serial(
    evaluations: &[F128],
    nodes: &[F128],
    weights: &[F128],
    point: F128,
) -> (F128, F128) {
    let mut result = F128::ZERO;
    let mut product = F128::ONE;
    for ((&evaluation, &node), &weight) in evaluations.iter().zip(nodes).zip(weights) {
        let difference = point - node;
        result = result * difference + product * evaluation * weight;
        product *= difference;
    }

    (result, product)
}

#[cfg(feature = "parallel")]
#[inline]
fn combine_evaluation_blocks(
    (left_result, left_product): (F128, F128),
    (right_result, right_product): (F128, F128),
) -> (F128, F128) {
    (
        left_result * right_product + left_product * right_result,
        left_product * right_product,
    )
}

/// Inverts a nonzero slice using Montgomery's batch-inversion trick. For
/// `n > 0`, this costs one field inversion and `3(n - 1)` multiplications;
/// an empty slice returns without doing any field arithmetic.
///
/// Serial, in-place adaptation of Flock's chunked batch inverse:
/// <https://github.com/succinctlabs/flock/blob/85fc0e7cc002e7ca4dffdff805ba89976e9a5293/crates/flock-core/src/permutation.rs#L133-L159>
/// Flock's parallel path uses `2^14`-element chunks; F2Z's natural domains are
/// far smaller, so one scan and one inversion avoid unnecessary task overhead.
fn batch_invert_nonzero(values: &mut [F128]) {
    let Some((&first, remaining)) = values.split_first() else {
        return;
    };

    debug_assert!(
        !first.is_zero() && remaining.iter().all(|value| !value.is_zero()),
        "batch_invert_nonzero requires nonzero inputs",
    );

    let mut prefix_products = Vec::with_capacity(values.len());
    let mut product = first;
    prefix_products.push(product);
    for &value in remaining {
        product *= value;
        prefix_products.push(product);
    }

    let mut inverse = product
        .inverse()
        .expect("a product of nonzero field elements is nonzero");

    for i in (1..values.len()).rev() {
        let value = values[i];
        values[i] = inverse * prefix_products[i - 1];
        inverse *= value;
    }
    values[0] = inverse;
}

#[cfg(test)]
mod tests {
    use super::{LagrangeInterpolationDomain, NatEvaluatedPoly, NatEvaluationError};
    use field::F128;
    use proptest::prelude::*;

    fn arbitrary_field_vector_and_point() -> impl Strategy<Value = (Vec<F128>, F128)> {
        (1usize..=8).prop_flat_map(|len| {
            (prop::collection::vec(any::<u128>(), len), any::<u128>()).prop_map(
                |(values, point)| {
                    (
                        values.into_iter().map(F128::from).collect(),
                        F128::from(point),
                    )
                },
            )
        })
    }

    fn evaluate_coefficients(coefficients: &[F128], point: F128) -> F128 {
        coefficients
            .iter()
            .rev()
            .fold(F128::ZERO, |value, &coefficient| {
                value * point + coefficient
            })
    }

    fn polynomial_from_coefficients(
        coefficients: &[F128],
        domain: &LagrangeInterpolationDomain,
    ) -> NatEvaluatedPoly {
        NatEvaluatedPoly::new(
            domain
                .points
                .iter()
                .map(|&point| evaluate_coefficients(coefficients, point))
                .collect(),
        )
    }

    fn evaluate_lagrange_naively(
        evaluations: &[F128],
        domain: &LagrangeInterpolationDomain,
        point: F128,
    ) -> F128 {
        evaluations
            .iter()
            .enumerate()
            .fold(F128::ZERO, |sum, (i, &evaluation)| {
                let mut numerator = F128::ONE;
                let mut denominator = F128::ONE;

                for (j, &node) in domain.points.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    numerator *= point - node;
                    denominator *= domain.points[i] - node;
                }

                sum + evaluation
                    * numerator
                    * denominator
                        .inverse()
                        .expect("interpolation nodes are distinct")
            })
    }

    #[test]
    fn zero_length_domain_is_empty() {
        let aux = LagrangeInterpolationDomain::new(0);

        assert!(aux.points.is_empty());
        assert!(aux.weights.is_empty());
    }

    #[test]
    fn interpolation_nodes_use_f128_bit_pattern_embedding() {
        let aux = LagrangeInterpolationDomain::new(5);
        let expected: Vec<_> = (0..5).map(|i| F128::from(i as u128)).collect();

        assert_eq!(aux.points, expected);
    }

    #[test]
    fn singleton_has_inverse_of_the_empty_product() {
        let aux = LagrangeInterpolationDomain::new(1);

        assert_eq!(aux.points, vec![F128::ZERO]);
        assert_eq!(aux.weights, vec![F128::ONE]);
    }

    #[test]
    fn power_of_two_domains_share_one_barycentric_weight() {
        for len in [1, 2, 4, 8, 16] {
            let domain = LagrangeInterpolationDomain::new(len);

            assert!(
                domain
                    .weights
                    .iter()
                    .all(|&weight| weight == domain.weights[0])
            );
        }
    }

    #[test]
    fn denominator_inverses_match_direct_field_products() {
        for len in 2..=8 {
            let aux = LagrangeInterpolationDomain::new(len);

            assert_eq!(aux.points.len(), len);
            assert_eq!(aux.weights.len(), len);

            for i in 0..len {
                let denominator = aux
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .fold(F128::ONE, |product, (_, &point)| {
                        product * (aux.points[i] - point)
                    });

                assert_eq!(
                    denominator * aux.weights[i],
                    F128::ONE,
                    "incorrect inverse denominator for len={len}, i={i}",
                );
            }
        }
    }

    #[test]
    fn batch_inversion_matches_individual_inverses() {
        for len in 0..=16 {
            let mut values: Vec<_> = (1..=len).map(|value| F128::from(value as u128)).collect();
            let expected: Vec<_> = values
                .iter()
                .map(|value| value.inverse().expect("test inputs are nonzero"))
                .collect();

            super::batch_invert_nonzero(&mut values);

            assert_eq!(values, expected);
        }
    }

    #[test]
    fn evaluating_an_empty_polynomial_is_rejected() {
        let polynomial = NatEvaluatedPoly::new(vec![]);
        let domain = LagrangeInterpolationDomain::new(0);
        let point = F128::from(7u128);

        assert_eq!(
            polynomial.evaluate_at_point(point),
            Err(NatEvaluationError::EmptyPolynomial),
        );
        assert_eq!(
            polynomial.evaluate_at_point_with_domain(point, &domain),
            Err(NatEvaluationError::EmptyPolynomial),
        );
    }

    #[test]
    fn evaluation_rejects_a_domain_with_the_wrong_number_of_nodes() {
        let polynomial = NatEvaluatedPoly::new(vec![F128::ONE, F128::from(2u128)]);
        let domain = LagrangeInterpolationDomain::new(3);

        assert_eq!(
            polynomial.evaluate_at_point_with_domain(F128::from(7u128), &domain),
            Err(NatEvaluationError::DomainSizeMismatch),
        );
    }

    #[test]
    fn evaluation_rejects_a_domain_with_the_wrong_number_of_weights() {
        let polynomial = NatEvaluatedPoly::new(vec![F128::ONE, F128::from(2u128)]);
        let mut domain = LagrangeInterpolationDomain::new(2);
        domain.weights.pop();

        assert_eq!(
            polynomial.evaluate_at_point_with_domain(F128::from(7u128), &domain),
            Err(NatEvaluationError::DomainSizeMismatch),
        );
    }

    #[test]
    fn singleton_polynomial_is_constant() {
        let value = F128::from(42u128);
        let polynomial = NatEvaluatedPoly::new(vec![value]);
        let domain = LagrangeInterpolationDomain::new(1);

        assert_eq!(
            polynomial.evaluate_at_point_with_domain(F128::from(123u128), &domain),
            Ok(value),
        );
    }

    #[test]
    fn evaluation_at_interpolation_nodes_recovers_stored_values() {
        let evaluations = vec![
            F128::from(3u128),
            F128::from(5u128),
            F128::from(11u128),
            F128::from(17u128),
            F128::from(29u128),
        ];
        let polynomial = NatEvaluatedPoly::new(evaluations.clone());
        let domain = LagrangeInterpolationDomain::new(evaluations.len());

        for (&point, &expected) in domain.points.iter().zip(&evaluations) {
            assert_eq!(
                polynomial.evaluate_at_point_with_domain(point, &domain),
                Ok(expected),
            );
        }
    }

    #[test]
    fn evaluation_matches_the_direct_lagrange_definition() {
        let evaluations = vec![
            F128::from(2u128),
            F128::from(7u128),
            F128::from(13u128),
            F128::from(29u128),
            F128::from(43u128),
        ];
        let polynomial = NatEvaluatedPoly::new(evaluations.clone());
        let domain = LagrangeInterpolationDomain::new(evaluations.len());
        let point = F128::from(31u128);

        assert_eq!(
            polynomial.evaluate_at_point_with_domain(point, &domain),
            Ok(evaluate_lagrange_naively(&evaluations, &domain, point)),
        );
    }

    #[test]
    fn evaluation_matches_known_cubic_and_reuses_the_domain() {
        let domain = LagrangeInterpolationDomain::new(4);
        let coefficient_sets = [
            [
                F128::from(2u128),
                F128::from(3u128),
                F128::from(5u128),
                F128::from(7u128),
            ],
            [
                F128::from(11u128),
                F128::from(13u128),
                F128::from(17u128),
                F128::from(19u128),
            ],
        ];
        let evaluation_points = [F128::from(5u128), F128::from(23u128)];

        for coefficients in coefficient_sets {
            let polynomial = polynomial_from_coefficients(&coefficients, &domain);

            for point in evaluation_points {
                assert_eq!(
                    polynomial.evaluate_at_point_with_domain(point, &domain),
                    Ok(evaluate_coefficients(&coefficients, point)),
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn domain_construction_matches_serial_around_parallel_threshold() {
        let workload = crate::parallel::workload_size::<F128>().saturating_mul(2);
        let parallel_len = (1usize..)
            .find(|&len| len.saturating_mul(len.saturating_sub(1)) > workload)
            .unwrap();
        let serial_len = (1..parallel_len)
            .rev()
            .find(|len| !len.is_power_of_two())
            .unwrap();
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        for len in [serial_len, parallel_len] {
            let expected = serial_pool.install(|| LagrangeInterpolationDomain::new(len));
            let actual = parallel_pool.install(|| LagrangeInterpolationDomain::new(len));

            assert_eq!(actual, expected);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn evaluation_matches_serial_across_parallel_threshold() {
        let parallel_len = (crate::parallel::workload_size::<F128>() / 3 + 1).next_power_of_two();
        let serial_len = parallel_len / 2;

        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                for len in [serial_len, parallel_len] {
                    let evaluations: Vec<_> =
                        (0..len).map(|i| F128::from((3 * i + 1) as u128)).collect();
                    let domain = LagrangeInterpolationDomain::new(len);
                    let polynomial = NatEvaluatedPoly::new(evaluations.clone());

                    for point in [F128::from(u128::MAX), domain.points[len / 2]] {
                        let expected = super::evaluate_block_serial(
                            &evaluations,
                            &domain.points,
                            &domain.weights,
                            point,
                        )
                        .0;

                        assert_eq!(
                            polynomial.evaluate_at_point_with_domain(point, &domain),
                            Ok(expected),
                        );
                    }
                }
            });
    }

    proptest! {
        #[test]
        fn evaluation_matches_direct_lagrange_for_arbitrary_tables(
            (evaluations, point) in arbitrary_field_vector_and_point(),
        ) {
            let domain = LagrangeInterpolationDomain::new(evaluations.len());
            let expected = evaluate_lagrange_naively(&evaluations, &domain, point);
            let polynomial = NatEvaluatedPoly::new(evaluations);

            prop_assert_eq!(
                polynomial.evaluate_at_point(point),
                polynomial.evaluate_at_point_with_domain(point, &domain),
            );
            prop_assert_eq!(
                polynomial.evaluate_at_point_with_domain(point, &domain),
                Ok(expected),
            );
        }

        #[test]
        fn evaluation_matches_arbitrary_coefficient_polynomials(
            (coefficients, point) in arbitrary_field_vector_and_point(),
        ) {
            let domain = LagrangeInterpolationDomain::new(coefficients.len());
            let polynomial = polynomial_from_coefficients(&coefficients, &domain);

            prop_assert_eq!(
                polynomial.evaluate_at_point_with_domain(point, &domain),
                Ok(evaluate_coefficients(&coefficients, point)),
            );
        }
    }
}
