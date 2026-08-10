// SPDX-License-Identifier: Apache-2.0

use std::ops::Deref;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::Field;

#[cfg(feature = "parallel")]
use crate::parallel::workload_size;

/// Number of outputs at which we dispatch a fold round to Rayon.
/// This is an initial value picked from Flock; benchmarks should calibrate it for our field.
#[cfg(feature = "parallel")]
const PARALLEL_FOLD_THRESHOLD: usize = 1 << 12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseMleError {
    SizeMismatch,
    InvalidNumVarsRange,
    TooManyChallenges,
}

/// A multilinear polynomial represented by its evaluations on a Boolean cube.
/// Adapted from Zinc+ `DenseMultilinearExtension` at: https://github.com/NethermindEth/zinc-plus/blob/8dbd6007008b2d10e95e73149ca2fd5b7d8e00f9/poly/src/mle/dense.rs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<T: Default> {
    /// Evaluations on `{0,1}^num_vars` in little-endian index order.
    evaluations: Vec<T>,
    /// Number of (unfixed) variables.
    num_vars: usize,
}

impl<T: Default> DenseMultilinearExtension<T> {
    /// Constructs the unique zero-variable MLE with the supplied evaluation.
    pub fn zero_vars(evaluation: T) -> Self {
        Self {
            evaluations: vec![evaluation],
            num_vars: 0,
        }
    }

    /// @dev: question to reviewer: should we use assert instead of error?
    pub fn from_evaluations(num_vars: usize, evaluations: Vec<T>) -> Result<Self, DenseMleError> {
        if num_vars >= usize::BITS as usize {
            return Err(DenseMleError::InvalidNumVarsRange);
        }

        if evaluations.len() != 1 << num_vars {
            return Err(DenseMleError::SizeMismatch);
        }

        Ok(Self {
            evaluations,
            num_vars,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field + Copy> DenseMultilinearExtension<F> {
    /// Fixes the lowest-index remaining variables at `r`, in order.
    ///
    /// Each round replaces adjacent evaluations with
    /// `f(0, b) + r_i * (f(1, b) - f(0, b))`. Large rounds use a parallel
    /// destination buffer; shrinking rounds are folded in place.
    ///
    /// Hybrid parallel scheduling adapted from Flock:
    /// <https://github.com/succinctlabs/flock/blob/85fc0e7cc002e7ca4dffdff805ba89976e9a5293/crates/flock-core/src/permutation.rs#L236-L263>
    pub fn fold(&mut self, r: &[F]) -> Result<(), DenseMleError>
    where
        F: Send + Sync,
    {
        self.check_fold_width(r)?;

        #[cfg(feature = "parallel")]
        let mut scratch = Vec::new();

        for &challenge in r {
            #[cfg(feature = "parallel")]
            if self.evaluations.len() / 2 >= PARALLEL_FOLD_THRESHOLD {
                self.fold_round_parallel(challenge, &mut scratch);
                self.num_vars -= 1;
                continue;
            }

            self.fold_round_in_place(challenge);
            self.num_vars -= 1;
        }

        Ok(())
    }

    /// Evaluates this multilinear extension at `r` without cloning or mutating
    /// the table.
    pub fn evaluate(&self, r: &[F]) -> Result<F, DenseMleError>
    where
        F: Send + Sync,
    {
        let len = self.evaluations.len();
        if r.len() >= usize::BITS as usize {
            return Err(DenseMleError::InvalidNumVarsRange);
        }

        if len != 1 << r.len() {
            return Err(DenseMleError::SizeMismatch);
        }

        Ok(Self::evaluate_exact(&self.evaluations, r))
    }

    #[inline]
    /// Unrolled base cases adapted from WHIR's `eval_exact` (Apache-2.0):
    /// <https://github.com/worldfnd/whir/blob/e0aec15225fd5e63594bdc49566e080a6cab2f24/src/algebra/multilinear.rs#L31-L64>
    fn evaluate_exact(evaluations: &[F], r: &[F]) -> F
    where
        F: Send + Sync,
    {
        debug_assert_eq!(evaluations.len(), 1 << r.len());

        let interpolate = |zero: F, one: F, challenge: F| zero + challenge * (one - zero);

        match r {
            [] => evaluations[0],
            [r0] => interpolate(evaluations[0], evaluations[1], *r0),
            [r0, r1] => {
                let a0 = interpolate(evaluations[0], evaluations[1], *r0);
                let a1 = interpolate(evaluations[2], evaluations[3], *r0);
                interpolate(a0, a1, *r1)
            }
            [r0, r1, r2] => {
                let a00 = interpolate(evaluations[0], evaluations[1], *r0);
                let a01 = interpolate(evaluations[2], evaluations[3], *r0);
                let a10 = interpolate(evaluations[4], evaluations[5], *r0);
                let a11 = interpolate(evaluations[6], evaluations[7], *r0);
                let a0 = interpolate(a00, a01, *r1);
                let a1 = interpolate(a10, a11, *r1);
                interpolate(a0, a1, *r2)
            }
            [r0, r1, r2, r3] => {
                let a000 = interpolate(evaluations[0], evaluations[1], *r0);
                let a001 = interpolate(evaluations[2], evaluations[3], *r0);
                let a010 = interpolate(evaluations[4], evaluations[5], *r0);
                let a011 = interpolate(evaluations[6], evaluations[7], *r0);
                let a100 = interpolate(evaluations[8], evaluations[9], *r0);
                let a101 = interpolate(evaluations[10], evaluations[11], *r0);
                let a110 = interpolate(evaluations[12], evaluations[13], *r0);
                let a111 = interpolate(evaluations[14], evaluations[15], *r0);
                let a00 = interpolate(a000, a001, *r1);
                let a01 = interpolate(a010, a011, *r1);
                let a10 = interpolate(a100, a101, *r1);
                let a11 = interpolate(a110, a111, *r1);
                let a0 = interpolate(a00, a01, *r2);
                let a1 = interpolate(a10, a11, *r2);
                interpolate(a0, a1, *r3)
            }
            [remaining @ .., last_r] => {
                let (zero, one) = evaluations.split_at(evaluations.len() / 2);

                #[cfg(feature = "parallel")]
                let (zero, one) = if evaluations.len() > workload_size::<F>() {
                    rayon::join(
                        || Self::evaluate_exact(zero, remaining),
                        || Self::evaluate_exact(one, remaining),
                    )
                } else {
                    (
                        Self::evaluate_exact(zero, remaining),
                        Self::evaluate_exact(one, remaining),
                    )
                };

                #[cfg(not(feature = "parallel"))]
                let (zero, one) = (
                    Self::evaluate_exact(zero, remaining),
                    Self::evaluate_exact(one, remaining),
                );

                interpolate(zero, one, *last_r)
            }
        }
    }

    fn check_fold_width(&self, r: &[F]) -> Result<(), DenseMleError> {
        if r.len() > self.num_vars {
            return Err(DenseMleError::TooManyChallenges);
        }
        Ok(())
    }

    fn fold_round_in_place(&mut self, challenge: F) {
        let half = self.evaluations.len() / 2;
        for i in 0..half {
            let left = self.evaluations[2 * i];
            let right = self.evaluations[2 * i + 1];
            self.evaluations[i] = left + challenge * (right - left);
        }
        self.evaluations.truncate(half);
    }

    #[cfg(feature = "parallel")]
    fn fold_round_parallel(&mut self, challenge: F, scratch: &mut Vec<F>)
    where
        F: Send + Sync,
    {
        self.evaluations
            .par_chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect_into_vec(scratch);

        std::mem::swap(&mut self.evaluations, scratch);
    }
}

impl<T: Default> Deref for DenseMultilinearExtension<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.evaluations
    }
}

impl<T: Default> IntoIterator for DenseMultilinearExtension<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.evaluations.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseMleError, DenseMultilinearExtension};
    use crate::eq::eq_table;
    use field::F128;
    use num_traits::{ConstOne, ConstZero};
    use proptest::prelude::*;

    fn fold_layer(evaluations: &[F128], challenge: F128) -> Vec<F128> {
        evaluations
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect()
    }

    fn arbitrary_evaluation_case() -> impl Strategy<Value = (Vec<F128>, Vec<F128>)> {
        (0usize..=8).prop_flat_map(|num_vars| {
            (
                prop::collection::vec(any::<u128>(), 1usize << num_vars),
                prop::collection::vec(any::<u128>(), num_vars),
            )
                .prop_map(|(evaluations, point)| {
                    (
                        evaluations.into_iter().map(F128::from).collect(),
                        point.into_iter().map(F128::from).collect(),
                    )
                })
        })
    }

    #[test]
    fn zero_vars_contains_one_evaluation() {
        let mle = DenseMultilinearExtension::zero_vars(7u32);

        assert_eq!(mle.num_vars, 0);
        assert_eq!(mle.evaluations, vec![7]);
    }

    #[test]
    fn from_evaluations_accepts_exact_shapes_and_preserves_order() {
        for num_vars in 0..=8 {
            let evaluations: Vec<_> = (0..1usize << num_vars).collect();
            let expected = evaluations.clone();

            let mle = DenseMultilinearExtension::from_evaluations(num_vars, evaluations)
                .expect("an exact power-of-two table should be accepted");

            assert_eq!(mle.num_vars(), num_vars);
            assert_eq!(&*mle, expected.as_slice());
        }
    }

    #[test]
    fn consuming_iteration_preserves_evaluation_order() {
        let mle = DenseMultilinearExtension::from_evaluations(2, vec![1u32, 2, 3, 4]).unwrap();

        assert_eq!(mle.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn zero_vars_matches_the_exact_shape_constructor() {
        assert_eq!(
            DenseMultilinearExtension::zero_vars(7u32),
            DenseMultilinearExtension::from_evaluations(0, vec![7]).unwrap()
        );
    }

    #[test]
    fn from_evaluations_rejects_empty_table() {
        assert_eq!(
            DenseMultilinearExtension::<u32>::from_evaluations(0, vec![]),
            Err(DenseMleError::SizeMismatch)
        );
    }

    #[test]
    fn from_evaluations_rejects_short_table() {
        assert_eq!(
            DenseMultilinearExtension::from_evaluations(2, vec![1u32, 2, 3]),
            Err(DenseMleError::SizeMismatch)
        );
    }

    #[test]
    fn from_evaluations_rejects_long_table() {
        assert_eq!(
            DenseMultilinearExtension::from_evaluations(2, vec![1u32, 2, 3, 4, 5]),
            Err(DenseMleError::SizeMismatch)
        );
    }

    #[test]
    fn from_evaluations_rejects_unrepresentable_num_vars_without_shifting() {
        assert_eq!(
            DenseMultilinearExtension::<u32>::from_evaluations(usize::BITS as usize, vec![]),
            Err(DenseMleError::InvalidNumVarsRange)
        );
        assert_eq!(
            DenseMultilinearExtension::<u32>::from_evaluations(usize::MAX, vec![]),
            Err(DenseMleError::InvalidNumVarsRange)
        );
    }

    #[test]
    fn fold_with_no_challenges_is_a_noop() {
        let mut zero_vars = DenseMultilinearExtension::zero_vars(F128::from(7u128));
        let expected_zero_vars = zero_vars.clone();
        zero_vars.fold(&[]).unwrap();
        assert_eq!(zero_vars, expected_zero_vars);

        let mut two_vars = DenseMultilinearExtension::from_evaluations(
            2,
            vec![
                F128::from(10u128),
                F128::from(11u128),
                F128::from(20u128),
                F128::from(21u128),
            ],
        )
        .unwrap();
        let expected_two_vars = two_vars.clone();
        two_vars.fold(&[]).unwrap();
        assert_eq!(two_vars, expected_two_vars);
    }

    #[test]
    fn fold_at_zero_selects_adjacent_zero_children() {
        let evaluations: Vec<_> = (0u128..8).map(F128::from).collect();
        let mut mle = DenseMultilinearExtension::from_evaluations(3, evaluations.clone()).unwrap();

        mle.fold(&[F128::ZERO]).unwrap();

        assert_eq!(mle.num_vars(), 2);
        assert_eq!(
            &*mle,
            &[
                evaluations[0],
                evaluations[2],
                evaluations[4],
                evaluations[6]
            ]
        );
    }

    #[test]
    fn fold_at_one_selects_adjacent_one_children() {
        let evaluations: Vec<_> = (0u128..8).map(F128::from).collect();
        let mut mle = DenseMultilinearExtension::from_evaluations(3, evaluations.clone()).unwrap();

        mle.fold(&[F128::ONE]).unwrap();

        assert_eq!(mle.num_vars(), 2);
        assert_eq!(
            &*mle,
            &[
                evaluations[1],
                evaluations[3],
                evaluations[5],
                evaluations[7]
            ]
        );
    }

    #[test]
    fn fold_at_non_boolean_challenge_interpolates_adjacent_pairs() {
        let evaluations = vec![
            F128::from(3u128),
            F128::from(5u128),
            F128::from(11u128),
            F128::from(19u128),
        ];
        let challenge = F128::from(7u128);
        let expected = fold_layer(&evaluations, challenge);
        let mut mle = DenseMultilinearExtension::from_evaluations(2, evaluations).unwrap();

        mle.fold(&[challenge]).unwrap();

        assert_eq!(mle.num_vars(), 1);
        assert_eq!(&*mle, expected.as_slice());
    }

    #[test]
    fn fold_uses_each_challenge_in_little_endian_variable_order() {
        let evaluations: Vec<_> = (10u128..18).map(F128::from).collect();
        let challenges = [F128::from(3u128), F128::from(9u128)];
        let after_first = fold_layer(&evaluations, challenges[0]);
        let expected = fold_layer(&after_first, challenges[1]);
        let mut mle = DenseMultilinearExtension::from_evaluations(3, evaluations).unwrap();

        mle.fold(&challenges).unwrap();

        assert_eq!(mle.num_vars(), 1);
        assert_eq!(&*mle, expected.as_slice());
    }

    #[test]
    fn folding_all_variables_leaves_one_evaluation() {
        let evaluations = vec![
            F128::from(2u128),
            F128::from(3u128),
            F128::from(5u128),
            F128::from(7u128),
        ];
        let challenges = [F128::from(11u128), F128::from(13u128)];
        let after_first = fold_layer(&evaluations, challenges[0]);
        let expected = fold_layer(&after_first, challenges[1]);
        let mut mle = DenseMultilinearExtension::from_evaluations(2, evaluations).unwrap();

        mle.fold(&challenges).unwrap();

        assert_eq!(mle.num_vars(), 0);
        assert_eq!(&*mle, expected.as_slice());
    }

    #[test]
    fn evaluate_zero_variable_table_returns_its_only_evaluation() {
        let evaluation = F128::from(7u128);
        let mle = DenseMultilinearExtension::zero_vars(evaluation);

        assert_eq!(mle.evaluate(&[]), Ok(evaluation));
    }

    #[test]
    fn evaluate_rejects_points_with_the_wrong_width() {
        let mle = DenseMultilinearExtension::from_evaluations(2, vec![F128::ZERO; 4]).unwrap();

        assert_eq!(
            mle.evaluate(&[F128::from(3u128)]),
            Err(DenseMleError::SizeMismatch)
        );
        assert_eq!(
            mle.evaluate(&[F128::from(3u128), F128::from(5u128), F128::from(7u128),]),
            Err(DenseMleError::SizeMismatch)
        );
    }

    #[test]
    fn evaluate_rejects_unrepresentable_point_width_without_shifting() {
        let point = vec![F128::ZERO; usize::BITS as usize];
        let mle = DenseMultilinearExtension::zero_vars(F128::ZERO);

        assert_eq!(
            mle.evaluate(&point),
            Err(DenseMleError::InvalidNumVarsRange)
        );
    }

    #[test]
    fn evaluate_at_boolean_points_selects_little_endian_entries() {
        for num_vars in 0..=5 {
            let evaluations: Vec<_> = (0..1usize << num_vars)
                .map(|index| F128::from((index + 10) as u128))
                .collect();
            let mle =
                DenseMultilinearExtension::from_evaluations(num_vars, evaluations.clone()).unwrap();

            for (index, &expected) in evaluations.iter().enumerate() {
                let point: Vec<_> = (0..num_vars)
                    .map(|bit| F128::from(((index >> bit) & 1) as u128))
                    .collect();

                assert_eq!(
                    mle.evaluate(&point),
                    Ok(expected),
                    "failed for {num_vars} variables at index {index}"
                );
            }
        }
    }

    #[test]
    fn evaluate_at_non_boolean_point_matches_layer_folding() {
        let evaluations = vec![
            F128::from(2u128),
            F128::from(3u128),
            F128::from(5u128),
            F128::from(7u128),
        ];
        let point = [F128::from(11u128), F128::from(13u128)];
        let after_first = fold_layer(&evaluations, point[0]);
        let expected = fold_layer(&after_first, point[1])[0];
        let mle = DenseMultilinearExtension::from_evaluations(2, evaluations).unwrap();

        assert_eq!(mle.evaluate(&point), Ok(expected));
    }

    #[test]
    fn evaluate_unrolled_cases_and_recursive_boundary_match_folding() {
        for num_vars in 0..=5 {
            let evaluations: Vec<_> = (0..1usize << num_vars)
                .map(|index| F128::from((3 * index + 1) as u128))
                .collect();
            let point: Vec<_> = (0..num_vars)
                .map(|index| F128::from((5 * index + 2) as u128))
                .collect();
            let expected = point.iter().fold(evaluations.clone(), |layer, &challenge| {
                fold_layer(&layer, challenge)
            })[0];
            let mle = DenseMultilinearExtension::from_evaluations(num_vars, evaluations).unwrap();

            assert_eq!(
                mle.evaluate(&point),
                Ok(expected),
                "failed for {num_vars} variables"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn evaluate_matches_folding_around_parallel_threshold() {
        let threshold = crate::parallel::workload_size::<F128>();
        assert!(threshold.is_power_of_two());

        let threshold_log = threshold.trailing_zeros() as usize;
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                // Exactly at the cutoff is serial; the next power of two
                // performs one parallel split at the root.
                for num_vars in [threshold_log, threshold_log + 1] {
                    let evaluations: Vec<_> = (0..1usize << num_vars)
                        .map(|i| F128::from((3 * i + 1) as u128))
                        .collect();
                    let point: Vec<_> = (0..num_vars)
                        .map(|i| F128::from((5 * i + 2) as u128))
                        .collect();
                    let expected = point.iter().fold(evaluations.clone(), |layer, &challenge| {
                        fold_layer(&layer, challenge)
                    })[0];
                    let mle =
                        DenseMultilinearExtension::from_evaluations(num_vars, evaluations).unwrap();

                    assert_eq!(mle.evaluate(&point), Ok(expected));
                }
            });
    }

    proptest! {
        #[test]
        fn evaluate_matches_fold_and_eq_table(
            (evaluations, point) in arbitrary_evaluation_case()
        ) {
            let mle = DenseMultilinearExtension::from_evaluations(
                point.len(),
                evaluations.clone(),
            )
            .unwrap();
            let actual = mle.evaluate(&point).unwrap();

            let weights = eq_table(&point);
            let expected_from_eq = evaluations
                .iter()
                .zip(weights)
                .fold(F128::ZERO, |sum, (&evaluation, weight)| {
                    sum + evaluation * weight
                });

            let mut folded = DenseMultilinearExtension::from_evaluations(
                point.len(),
                evaluations,
            )
            .unwrap();
            folded.fold(&point).unwrap();

            prop_assert_eq!(actual, expected_from_eq);
            prop_assert_eq!(actual, folded[0]);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_fold_matches_sequential_across_the_threshold() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                let num_vars = 14;
                let evaluations: Vec<_> = (0..1usize << num_vars)
                    .map(|i| F128::from(i as u128))
                    .collect();
                let challenges = [F128::from(3u128), F128::from(9u128), F128::from(27u128)];
                let expected = challenges
                    .iter()
                    .fold(evaluations.clone(), |layer, &challenge| {
                        fold_layer(&layer, challenge)
                    });
                let mut mle =
                    DenseMultilinearExtension::from_evaluations(num_vars, evaluations).unwrap();

                mle.fold(&challenges).unwrap();

                assert_eq!(mle.num_vars(), num_vars - challenges.len());
                assert_eq!(&*mle, expected.as_slice());
            });
    }

    #[test]
    fn fold_rejects_too_many_challenges_before_mutating() {
        let mut mle = DenseMultilinearExtension::from_evaluations(
            1,
            vec![F128::from(3u128), F128::from(5u128)],
        )
        .unwrap();
        let expected = mle.clone();

        assert_eq!(
            mle.fold(&[F128::from(7u128), F128::from(11u128)]),
            Err(DenseMleError::TooManyChallenges)
        );
        assert_eq!(mle, expected);
    }
}
