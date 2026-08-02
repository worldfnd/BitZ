// SPDX-License-Identifier: Apache-2.0
//
// Adapted from Zinc+ `DenseMultilinearExtension` at:
// https://github.com/NethermindEth/zinc-plus/blob/8dbd6007008b2d10e95e73149ca2fd5b7d8e00f9/poly/src/mle/dense.rs

use std::{
    ops::{Deref, DerefMut, Index, IndexMut},
    slice::SliceIndex,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseMleError {
    SizeMismatch,
    InvalidNumVarsRange,
}

/// A multilinear polynomial represented by its evaluations on a Boolean cube.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<T> {
    /// Evaluations on `{0,1}^num_vars` in little-endian index order.
    evaluations: Vec<T>,
    /// Number of variables.
    num_vars: usize,
}

impl<T> DenseMultilinearExtension<T> {
    /// Constructs the unique zero-variable MLE with the supplied evaluation.
    pub fn zero_vars(evaluation: T) -> Self {
        Self {
            evaluations: vec![evaluation],
            num_vars: 0,
        }
    }

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

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<T> Deref for DenseMultilinearExtension<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.evaluations
    }
}

impl<T> DerefMut for DenseMultilinearExtension<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.evaluations
    }
}

impl<T> IntoIterator for DenseMultilinearExtension<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.evaluations.into_iter()
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for DenseMultilinearExtension<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for DenseMultilinearExtension<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.evaluations[index]
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseMleError, DenseMultilinearExtension};

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
}
