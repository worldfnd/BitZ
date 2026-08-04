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
    use super::LagrangeInterpolationDomain;
    use field::F128;

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
}
