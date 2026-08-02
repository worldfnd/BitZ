// SPDX-License-Identifier: Apache-2.0
//
// Adapted from Zinc+ `DenseMultilinearExtension` at:
// https://github.com/NethermindEth/zinc-plus/blob/8dbd6007008b2d10e95e73149ca2fd5b7d8e00f9/poly/src/mle/dense.rs

use std::{
    ops::{Deref, DerefMut, Index, IndexMut},
    slice::SliceIndex,
};

/// A multilinear polynomial represented by its evaluations on a Boolean cube.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMultilinearExtension<T> {
    /// Evaluations on `{0,1}^num_vars` in little-endian index order.
    pub evaluations: Vec<T>,
    /// Number of variables.
    pub num_vars: usize,
}

impl<T> DenseMultilinearExtension<T> {
    /// Constructs the unique zero-variable MLE with the supplied evaluation.
    pub fn zero_vars(evaluation: T) -> Self {
        Self {
            evaluations: vec![evaluation],
            num_vars: 0,
        }
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
    use super::DenseMultilinearExtension;

    #[test]
    fn zero_vars_contains_one_evaluation() {
        let mle = DenseMultilinearExtension::zero_vars(7u32);

        assert_eq!(mle.num_vars, 0);
        assert_eq!(mle.evaluations, vec![7]);
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
