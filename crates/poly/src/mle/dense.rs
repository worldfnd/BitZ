// SPDX-License-Identifier: Apache-2.0
//
// Adapted from Zinc+ `DenseMultilinearExtension` at:
// https://github.com/NethermindEth/zinc-plus/blob/8dbd6007008b2d10e95e73149ca2fd5b7d8e00f9/poly/src/mle/dense.rs

use std::{
    ops::{Deref, DerefMut, Index, IndexMut},
    slice::SliceIndex,
};

use crate::eq::Field;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseMleError {
    SizeMismatch,
    InvalidNumVarsRange,
    TooManyChallenges,
}

/// A multilinear polynomial represented by its evaluations on a Boolean cube.
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

impl<F: Field> DenseMultilinearExtension<F> {
    pub fn fold(&mut self, r: &[F]) -> Result<(), DenseMleError> {
        let vars_to_fold = r.len();
        if vars_to_fold == 0 {
            return Ok(());
        }
        if vars_to_fold > self.num_vars {
            return Err(DenseMleError::TooManyChallenges);
        }

        for folding_r in r {
            let half = 1 << (self.num_vars - 1);
            for i in 0..half {
                self.evaluations[i] = self.evaluations[2 * i]
                    + *folding_r * (self.evaluations[2 * i + 1] - self.evaluations[2 * i]);
            }
            self.num_vars -= 1;
        }

        self.evaluations.resize(1 << self.num_vars, F::default());
        Ok(())
    }
}

impl<T: Default> Deref for DenseMultilinearExtension<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.evaluations
    }
}

impl<T: Default> DerefMut for DenseMultilinearExtension<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evaluations
    }
}

impl<T: Default> IntoIterator for DenseMultilinearExtension<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.evaluations.into_iter()
    }
}

impl<T: Default, I: SliceIndex<[T]>> Index<I> for DenseMultilinearExtension<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<T: Default, I: SliceIndex<[T]>> IndexMut<I> for DenseMultilinearExtension<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.evaluations[index]
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseMleError, DenseMultilinearExtension};
    use field::F128;

    fn fold_layer(evaluations: &[F128], challenge: F128) -> Vec<F128> {
        evaluations
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect()
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
    fn forwards_slice_indexing_and_consuming_iteration() {
        let mut mle = DenseMultilinearExtension {
            evaluations: vec![1u32, 2, 3, 4],
            num_vars: 2,
        };

        assert_eq!(&*mle, &[1, 2, 3, 4]);
        mle[1] = 5;
        assert_eq!(&mle[1..3], &[5, 3]);
        assert_eq!(mle.into_iter().collect::<Vec<_>>(), vec![1, 5, 3, 4]);
    }

    #[test]
    fn fold_with_no_challenges_is_a_noop() {
        let mut zero_vars = DenseMultilinearExtension::zero_vars(F128::from(7));
        let expected_zero_vars = zero_vars.clone();
        zero_vars.fold(&[]).unwrap();
        assert_eq!(zero_vars, expected_zero_vars);

        let mut two_vars = DenseMultilinearExtension::from_evaluations(
            2,
            vec![
                F128::from(10),
                F128::from(11),
                F128::from(20),
                F128::from(21),
            ],
        )
        .unwrap();
        let expected_two_vars = two_vars.clone();
        two_vars.fold(&[]).unwrap();
        assert_eq!(two_vars, expected_two_vars);
    }

    #[test]
    fn fold_at_zero_selects_adjacent_zero_children() {
        let evaluations: Vec<_> = (0..8).map(F128::from).collect();
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
        let evaluations: Vec<_> = (0..8).map(F128::from).collect();
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
        let evaluations = vec![F128::from(3), F128::from(5), F128::from(11), F128::from(19)];
        let challenge = F128::from(7);
        let expected = fold_layer(&evaluations, challenge);
        let mut mle = DenseMultilinearExtension::from_evaluations(2, evaluations).unwrap();

        mle.fold(&[challenge]).unwrap();

        assert_eq!(mle.num_vars(), 1);
        assert_eq!(&*mle, expected.as_slice());
    }

    #[test]
    fn fold_uses_each_challenge_in_little_endian_variable_order() {
        let evaluations: Vec<_> = (10..18).map(F128::from).collect();
        let challenges = [F128::from(3), F128::from(9)];
        let after_first = fold_layer(&evaluations, challenges[0]);
        let expected = fold_layer(&after_first, challenges[1]);
        let mut mle = DenseMultilinearExtension::from_evaluations(3, evaluations).unwrap();

        mle.fold(&challenges).unwrap();

        assert_eq!(mle.num_vars(), 1);
        assert_eq!(&*mle, expected.as_slice());
    }

    #[test]
    fn folding_all_variables_leaves_one_evaluation() {
        let evaluations = vec![F128::from(2), F128::from(3), F128::from(5), F128::from(7)];
        let challenges = [F128::from(11), F128::from(13)];
        let after_first = fold_layer(&evaluations, challenges[0]);
        let expected = fold_layer(&after_first, challenges[1]);
        let mut mle = DenseMultilinearExtension::from_evaluations(2, evaluations).unwrap();

        mle.fold(&challenges).unwrap();

        assert_eq!(mle.num_vars(), 0);
        assert_eq!(&*mle, expected.as_slice());
    }

    #[test]
    fn fold_rejects_too_many_challenges_before_mutating() {
        let mut mle =
            DenseMultilinearExtension::from_evaluations(1, vec![F128::from(3), F128::from(5)])
                .unwrap();
        let expected = mle.clone();

        assert_eq!(
            mle.fold(&[F128::from(7), F128::from(11)]),
            Err(DenseMleError::TooManyChallenges)
        );
        assert_eq!(mle, expected);
    }
}
