use std::ops::{Add, Mul, Sub};

use field::{F128, Fq};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::parallel::workload_size;

pub trait Field:
    Copy + From<u128> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Default
{
    const ZERO: Self;
    const ONE: Self;
}

impl Field for F128 {
    const ZERO: Self = F128::ZERO;
    const ONE: Self = F128::ONE;
}

impl<const Q: u128> Field for Fq<Q> {
    const ZERO: Self = Fq::ZERO;
    const ONE: Self = Fq::ONE;
}

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
    for (&a, &b) in x.iter().zip(y.iter()) {
        result *= F128::ONE + a + b;
    }
    result
}

/// Builds the evaluation table of `eq(b, r)` over all Boolean vectors
/// `b ∈ {0,1}^n`, where `n = r.len()`.
///
/// For each Boolean vector `b`,
///
/// eq(b, r) = ∏_{i=0}^{n-1}
///     [b_i r_i + (1 - b_i)(1 - r_i)].
///
/// Since each `b_i` is Boolean, this is equivalently
///
/// eq(b, r) = ∏_{i=0}^{n-1} {
///     r_i       if b_i = 1,
///     1 - r_i   if b_i = 0.
/// }
///
/// The result has length `2^n`. Entry `j` contains `eq(b, r)` for
/// `b_i = (j >> i) & 1`, so variable `i` corresponds to bit `i` of
/// the index (little-endian order).
///
/// For `n = 0`, the table is `[F128::ONE]`, corresponding to the
/// empty product.
pub fn eq_table<F: Field + Send + Sync>(r: &[F]) -> Vec<F> {
    let n = 1 << r.len();
    // Allocate the final output once.
    let mut table = vec![F::ZERO; n];
    table[0] = F::ONE;

    for (i, &r_i) in r.iter().enumerate() {
        let half = 1usize << i;

        // The lower half contains the existing parent values.
        // The upper half receives their one-children.
        let (zero_children, one_children) = table[..2 * half].split_at_mut(half);

        let update = |(zero_child, one_child): (&mut F, &mut F)| {
            let parent = *zero_child;
            let one_value = parent * r_i;

            *zero_child = parent - one_value;
            *one_child = one_value;
        };

        // Per-level parallel doubling adapted from Flock:
        // https://github.com/succinctlabs/flock/blob/85fc0e7cc002e7ca4dffdff805ba89976e9a5293/crates/flock-core/src/pcs/ring_switch.rs#L276-L320
        #[cfg(feature = "parallel")]
        if half >= workload_size::<F>() && rayon::current_num_threads() > 1 {
            zero_children
                .par_iter_mut()
                .zip(one_children.par_iter_mut())
                .for_each(update);
            continue;
        }

        zero_children.iter_mut().zip(one_children).for_each(update);
    }

    table
}

#[cfg(test)]
pub mod tests {
    use super::{Field, eq_eval, eq_table};
    use field::{F128, FqDefault};
    use proptest::prelude::*;

    fn direct_table_entry<F: Field>(r: &[F], index: usize) -> F {
        r.iter().enumerate().fold(F::ONE, |acc, (bit, &r_i)| {
            let factor = if (index >> bit) & 1 == 0 {
                F::ONE - r_i
            } else {
                r_i
            };
            acc * factor
        })
    }

    #[test]
    fn empty_eq_tables_contain_one() {
        assert_eq!(eq_table::<F128>(&[]), vec![F128::ONE]);
        assert_eq!(eq_table::<FqDefault>(&[]), vec![FqDefault::ONE]);
    }

    #[test]
    fn eq_tables_use_little_endian_index_order() {
        let r_0 = F128::from(2u128);
        let r_1 = F128::from(7u128);
        assert_eq!(
            eq_table(&[r_0, r_1]),
            vec![
                (F128::ONE - r_0) * (F128::ONE - r_1),
                r_0 * (F128::ONE - r_1),
                (F128::ONE - r_0) * r_1,
                r_0 * r_1,
            ]
        );

        let r_0 = FqDefault::from(2u128);
        let r_1 = FqDefault::from(7u128);
        assert_eq!(
            eq_table(&[r_0, r_1]),
            vec![
                (FqDefault::ONE - r_0) * (FqDefault::ONE - r_1),
                r_0 * (FqDefault::ONE - r_1),
                (FqDefault::ONE - r_0) * r_1,
                r_0 * r_1,
            ]
        );
    }

    #[test]
    fn boolean_eq_tables_are_one_hot() {
        let selected = 0b1010usize;
        let r_f128: Vec<_> = (0..4)
            .map(|bit| F128::from((selected >> bit) & 1 == 1))
            .collect();
        let r_fq: Vec<_> = (0..4)
            .map(|bit| FqDefault::from(((selected >> bit) & 1) as u128))
            .collect();

        for (index, &weight) in eq_table(&r_f128).iter().enumerate() {
            assert_eq!(
                weight,
                if index == selected {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            );
        }
        for (index, &weight) in eq_table(&r_fq).iter().enumerate() {
            assert_eq!(
                weight,
                if index == selected {
                    FqDefault::ONE
                } else {
                    FqDefault::ZERO
                }
            );
        }
    }

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

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_eq_table_matches_serial_around_threshold() {
        let threshold = crate::parallel::workload_size::<F128>();
        assert!(threshold.is_power_of_two());

        let threshold_log = threshold.trailing_zeros() as usize;
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        // Final-round workloads immediately below and exactly at the cutoff.
        for num_vars in [threshold_log, threshold_log + 1] {
            let point: Vec<_> = (0..num_vars).map(|i| F128::from(i as u128 + 2)).collect();

            let expected = serial_pool.install(|| eq_table(&point));
            let actual = parallel_pool.install(|| eq_table(&point));

            assert_eq!(actual, expected);
        }
    }

    proptest! {
        #[test]
        fn eq_tables_match_the_direct_definition(
            raw in prop::collection::vec(any::<u128>(), 0..10)
        ) {
            let r_f128: Vec<_> = raw.iter().copied().map(F128::from).collect();
            let table_f128 = eq_table(&r_f128);
            prop_assert_eq!(table_f128.len(), 1usize << raw.len());
            for (index, &weight) in table_f128.iter().enumerate() {
                prop_assert_eq!(weight, direct_table_entry(&r_f128, index));
            }

            let r_fq: Vec<_> = raw.iter().copied().map(FqDefault::from).collect();
            let table_fq = eq_table(&r_fq);
            prop_assert_eq!(table_fq.len(), 1usize << raw.len());
            for (index, &weight) in table_fq.iter().enumerate() {
                prop_assert_eq!(weight, direct_table_entry(&r_fq, index));
            }
        }

        #[test]
        fn eq_table_weights_sum_to_one(
            raw in prop::collection::vec(any::<u128>(), 0..10)
        ) {
            let r_f128: Vec<_> = raw.iter().copied().map(F128::from).collect();
            let sum_f128 = eq_table(&r_f128)
                .into_iter()
                .fold(F128::ZERO, |sum, weight| sum + weight);
            prop_assert_eq!(sum_f128, F128::ONE);

            let r_fq: Vec<_> = raw.iter().copied().map(FqDefault::from).collect();
            let sum_fq = eq_table(&r_fq)
                .into_iter()
                .fold(FqDefault::ZERO, |sum, weight| sum + weight);
            prop_assert_eq!(sum_fq, FqDefault::ONE);
        }

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
