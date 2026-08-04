// SPDX-License-Identifier: Apache-2.0

use field::F128;

/// Reusable interpolation data for evaluations on the natural `F128` domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LagrangeInterpolationDomain {
    /// `domain[i] = F128::from(i as u128)`.
    pub domain: Vec<F128>,
    /// `w_i = (∏_{j != i} (domain[i] - domain[j]))⁻¹`
    pub weights: Vec<F128>,
}

impl LagrangeInterpolationDomain {
    /// Precomputes the interpolation nodes and their barycentric weights.
    pub fn new(len: usize) -> Self {
        let domain: Vec<_> = (0..len).map(|i| F128::from(i as u128)).collect();
        let mut weights = Vec::with_capacity(len);

        for i in 0..len {
            let mut denominator = F128::ONE;
            for j in 0..len {
                if j == i {
                    continue;
                }
                denominator *= domain[i] - domain[j];
            }
            weights.push(denominator);
        }

        batch_invert_nonzero(&mut weights);

        Self { domain, weights }
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

    /// Linear, allocation-free adaptation of Binius64's
    /// `EvaluationDomain::extrapolate` (Apache-2.0):
    /// <https://github.com/binius-zk/binius64/blob/e0ddeb91d3826457322e3b7434a8ca0625f2f56e/crates/math/src/univariate.rs#L208-L223>
    pub fn evaluate_at_point_with_domain(
        &self,
        point: F128,
        domain: &LagrangeInterpolationDomain,
    ) -> Result<F128, NatEvaluationError> {
        let len = self.evaluations.len();
        if len == 0 {
            return Err(NatEvaluationError::EmptyPolynomial);
        } else if len != domain.domain.len() || len != domain.weights.len() {
            return Err(NatEvaluationError::DomainSizeMismatch);
        }

        let mut result = F128::ZERO;
        let mut product = F128::ONE;
        for ((&evaluation, &node), &weight) in self
            .evaluations
            .iter()
            .zip(&domain.domain)
            .zip(&domain.weights)
        {
            let difference = point - node;
            result = result * difference + product * evaluation * weight;
            product *= difference;
        }

        Ok(result)
    }
}

/// Inverts a nonzero slice using Montgomery's batch-inversion trick. For
/// `n > 0`, this costs one field inversion and `3(n - 1)` multiplications;
/// an empty slice returns without doing any field arithmetic.
///
/// Serial, in-place adaptation of Flock's chunked batch inverse:
/// <https://github.com/succinctlabs/flock/blob/85fc0e7cc002e7ca4dffdff805ba89976e9a5293/crates/flock-core/src/permutation.rs#L133-L159>
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
                .domain
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

                for (j, &node) in domain.domain.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    numerator *= point - node;
                    denominator *= domain.domain[i] - node;
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

        assert!(aux.domain.is_empty());
        assert!(aux.weights.is_empty());
    }

    #[test]
    fn interpolation_nodes_use_f128_bit_pattern_embedding() {
        let aux = LagrangeInterpolationDomain::new(5);
        let expected: Vec<_> = (0..5).map(|i| F128::from(i as u128)).collect();

        assert_eq!(aux.domain, expected);
    }

    #[test]
    fn singleton_has_inverse_of_the_empty_product() {
        let aux = LagrangeInterpolationDomain::new(1);

        assert_eq!(aux.domain, vec![F128::ZERO]);
        assert_eq!(aux.weights, vec![F128::ONE]);
    }

    #[test]
    fn denominator_inverses_match_direct_field_products() {
        for len in 2..=8 {
            let aux = LagrangeInterpolationDomain::new(len);

            assert_eq!(aux.domain.len(), len);
            assert_eq!(aux.weights.len(), len);

            for i in 0..len {
                let denominator = aux
                    .domain
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .fold(F128::ONE, |product, (_, &point)| {
                        product * (aux.domain[i] - point)
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

        assert_eq!(
            polynomial.evaluate_at_point_with_domain(F128::from(7u128), &domain),
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

        for (&point, &expected) in domain.domain.iter().zip(&evaluations) {
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
}
