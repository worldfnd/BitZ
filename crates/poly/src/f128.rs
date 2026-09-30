//! `F128` kernels: an inner product with a single reduction and the MLE
//! evaluation built on it. This evaluation us about twice as fast as generic
//! [`DenseMultilinearExtension::evaluate`](crate::DenseMultilinearExtension::evaluate).

use field::{F128, Wide256};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::eq_table;

/// Variables of the low equality factor: the table is read in chunks of
/// `2^LOW_BITS` entries, the width of a packed bit row, so the high factor
/// has one entry per chunk.
const LOW_BITS: usize = 7;

/// Entries from which [`evaluate`] runs on the thread pool: the two-thread
/// crossover measured by the bench above.
#[cfg(feature = "parallel")]
pub const PARALLEL_EVALUATE_MIN: usize = 1 << 18;

/// `sum_i a_i b_i`, reduced once.
pub fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len(), "inner product operands differ in length");
    a.iter()
        .zip(b)
        .fold(Wide256::zero(), |sum, (x, y)| sum + Wide256::mul(*x, *y))
        .reduce()
}

/// `MLE[evaluations](point)`, one multiplication per entry: on the thread
/// pool from [`PARALLEL_EVALUATE_MIN`] entries, on the calling thread below.
pub fn evaluate(evaluations: &[F128], point: &[F128]) -> F128 {
    #[cfg(feature = "parallel")]
    if evaluations.len() >= PARALLEL_EVALUATE_MIN {
        return evaluate_parallel(evaluations, point);
    }
    evaluate_serial(evaluations, point)
}

/// [`evaluate`] on the calling thread.
///
/// `eq(., point)` is factored at [`LOW_BITS`]: each chunk of the table is an
/// inner product with the low factor, and the chunk sums are weighted by the
/// high factor.
pub fn evaluate_serial(evaluations: &[F128], point: &[F128]) -> F128 {
    let (eq_low, eq_high) = factors(evaluations, point);
    evaluations
        .chunks_exact(eq_low.len())
        .zip(&eq_high)
        .map(|(chunk, weight)| *weight * inner_product(chunk, &eq_low))
        .sum()
}

/// [`evaluate`] across the thread pool, one chunk per task.
#[cfg(feature = "parallel")]
pub fn evaluate_parallel(evaluations: &[F128], point: &[F128]) -> F128 {
    let (eq_low, eq_high) = factors(evaluations, point);
    evaluations
        .par_chunks_exact(eq_low.len())
        .zip(&eq_high)
        .map(|(chunk, weight)| *weight * inner_product(chunk, &eq_low))
        .sum()
}

/// The equality table's low and high factors, split at [`LOW_BITS`].
fn factors(evaluations: &[F128], point: &[F128]) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(
        evaluations.len(),
        1 << point.len(),
        "the point must have the table's width"
    );
    let (low, high) = point.split_at(point.len().min(LOW_BITS));
    (eq_table(low), eq_table(high))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseMultilinearExtension;

    /// Deterministic, non-sequential values.
    fn values(len: usize, seed: u128) -> Vec<F128> {
        (0..len)
            .map(|i| F128::from((i as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ seed))
            .collect()
    }

    #[test]
    fn inner_product_matches_the_reduced_sum() {
        let a = values(1000, 1);
        let b = values(1000, 2);
        let expected = a.iter().zip(&b).map(|(x, y)| *x * *y).sum::<F128>();
        assert_eq!(inner_product(&a, &b), expected);
    }

    /// Below, at and above the chunk width, and into the parallel range.
    #[test]
    fn both_paths_match_the_generic_evaluator() {
        for num_vars in (0..=9).chain([12, 18]) {
            let table = values(1 << num_vars, 3);
            let point = values(num_vars, 4);
            let generic = DenseMultilinearExtension::from_evaluations(num_vars, table.clone())
                .unwrap()
                .evaluate(&point)
                .unwrap();
            assert_eq!(evaluate(&table, &point), generic, "{num_vars} variables");
            assert_eq!(evaluate_serial(&table, &point), generic);
            #[cfg(feature = "parallel")]
            assert_eq!(evaluate_parallel(&table, &point), generic);
        }
    }
}
