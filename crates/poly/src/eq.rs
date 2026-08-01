use field::F128;

/// Evaluates the multilinear equality polynomial over F128
///
/// eq(x, y) = ∏_{i=0}^{n-1} [x_i y_i + (1 - x_i)(1 - y_i)].
///
/// On Boolean vectors, eq(x, y) = 1 exactly when x = y, and 0 otherwise.
///
/// Over GF(2^128), characteristic two gives
///
/// eq(x, y) = ∏_{i=0}^{n-1} (1 + x_i + y_i).
pub fn eq_eval(x: &[F128], y: &[F128]) -> F128 {
    assert_eq!(x.len(), y.len());
    let mut result = F128::ONE;
    for (&a, &b) in x.into_iter().zip(y.into_iter()) {
        result *= F128::ONE + a + b;
    }
    result
}

#[cfg(test)]
pub mod tests {
    use super::eq_eval;
    use field::F128;
    use proptest::prelude::*;

    #[test]
    fn empty_vectors_give_one() {
        assert_eq!(eq_eval(&[], &[]), F128::ONE);
    }

    #[test]
    fn one_coordinate_boolean_truth_table() {
        let zero = F128::ZERO;
        let one = F128::ONE;

        assert_eq!(eq_eval(&[zero], &[zero]), one);
        assert_eq!(eq_eval(&[zero], &[one]), zero);
        assert_eq!(eq_eval(&[one], &[zero]), zero);
        assert_eq!(eq_eval(&[one], &[one]), one);
    }

    #[test]
    fn boolean_vectors_are_equality_indicators() {
        let zero = F128::ZERO;
        let one = F128::ONE;

        let x = [zero, one, one, zero];
        let equal = [zero, one, one, zero];
        let different = [zero, one, zero, zero];

        assert_eq!(eq_eval(&x, &equal), one);
        assert_eq!(eq_eval(&x, &different), zero);
    }

    #[test]
    fn is_symmetric() {
        let x = [F128::from(3u128), F128::from(17u128)];
        let y = [F128::from(9u128), F128::from(41u128)];

        assert_eq!(eq_eval(&x, &y), eq_eval(&y, &x));
    }

    #[test]
    #[should_panic]
    fn rejects_different_widths() {
        eq_eval(&[F128::ZERO], &[F128::ZERO, F128::ONE]);
    }

    #[test]
    fn boolean_weights_sum_to_one() {
        let r = [
            F128::from(2u128),
            F128::from(7u128),
            F128::from(19u128),
            F128::from(31u128),
        ];
        let mut sum = F128::ZERO;

        for index in 0..1usize << r.len() {
            let point: Vec<_> = (0..r.len())
                .map(|bit| F128::from((index >> bit) & 1 == 1))
                .collect();
            sum += eq_eval(&point, &r);
        }

        assert_eq!(sum, F128::ONE);
    }

    proptest! {
        #[test]
        fn eq_matches_definition(
            pairs in prop::collection::vec(
                (any::<u128>(), any::<u128>()),
                0..64,
            )
        ) {
            let x: Vec<_> = pairs.iter()
                .map(|&(x, _)| F128::from(x))
                .collect();
            let y: Vec<_> = pairs.iter()
                .map(|&(_, y)| F128::from(y))
                .collect();

            let expected = x.iter().zip(&y).fold(
                F128::ONE,
                |acc, (&x_i, &y_i)| {
                    acc * (
                        x_i * y_i
                        + (F128::ONE - x_i)
                            * (F128::ONE - y_i)
                    )
                },
            );

            prop_assert_eq!(eq_eval(&x, &y), expected);
            prop_assert_eq!(eq_eval(&x, &y), eq_eval(&y, &x));
        }
    }
}
