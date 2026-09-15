use flock_core::{field::F128, pcs::ring_switch::tensor_algebra_transpose as reference_transpose};
use proptest::prelude::*;

use crate::transpose::tensor_algebra_transpose;

fn matrix_strategy() -> impl Strategy<Value = Vec<F128>> {
    prop::collection::vec((any::<u64>(), any::<u64>()), 128).prop_map(|words| {
        words
            .into_iter()
            .map(|(lo, hi)| F128::new(lo, hi))
            .collect()
    })
}

#[test]
fn every_single_bit_moves_to_its_transposed_coordinate() {
    for input_row in 0..128 {
        for input_bit in 0..128 {
            let mut input = [F128::ZERO; 128];
            let bit = 1u128 << input_bit;
            input[input_row] = F128::new(bit as u64, (bit >> 64) as u64);

            // Direct coordinates provide an oracle independent of the transpose algorithm.
            let mut expected = [F128::ZERO; 128];
            if input_row < 64 {
                expected[input_bit].lo = 1u64 << input_row;
            } else {
                expected[input_bit].hi = 1u64 << (input_row - 64);
            }

            assert_eq!(
                tensor_algebra_transpose(&input),
                expected,
                "input row {input_row}, input bit {input_bit}"
            );
        }
    }
}

#[test]
fn zero_and_all_one_matrices_remain_unchanged() {
    for value in [F128::ZERO, F128::new(u64::MAX, u64::MAX)] {
        let input = [value; 128];
        assert_eq!(tensor_algebra_transpose(&input), input);
    }
}

#[test]
fn identity_matrix_remains_unchanged() {
    let input: Vec<_> = (0..128)
        .map(|row| {
            let bit = 1u128 << row;
            F128::new(bit as u64, (bit >> 64) as u64)
        })
        .collect();
    assert_eq!(tensor_algebra_transpose(&input), input);
}

#[test]
fn asymmetric_bits_cross_row_and_limb_boundaries() {
    let mut input = [F128::ZERO; 128];
    input[0] = F128::new(0, 1u64 << 63);
    input[1] = F128::new(1, 0);
    input[63] = F128::new(0, 1);
    input[64] = F128::new(1u64 << 62, 0);
    input[127] = F128::new(2, 0);

    let mut expected = [F128::ZERO; 128];
    expected[0] = F128::new(2, 0);
    expected[1] = F128::new(0, 1u64 << 63);
    expected[62] = F128::new(0, 1);
    expected[64] = F128::new(1u64 << 63, 0);
    expected[127] = F128::new(1, 0);

    assert_eq!(tensor_algebra_transpose(&input), expected);
}

#[test]
fn rejects_invalid_matrix_lengths() {
    for length in [0, 1, 127, 129, 256] {
        let input = vec![F128::ZERO; length];
        assert!(
            std::panic::catch_unwind(|| tensor_algebra_transpose(&input)).is_err(),
            "accepted invalid length {length}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn arbitrary_matrices_match_flock_reference(input in matrix_strategy()) {
        prop_assert_eq!(tensor_algebra_transpose(&input), reference_transpose(&input));
    }

    #[test]
    fn arbitrary_matrices_obey_involution_and_linearity(
        left in matrix_strategy(),
        right in matrix_strategy(),
    ) {
        let left_transposed = tensor_algebra_transpose(&left);
        let right_transposed = tensor_algebra_transpose(&right);
        prop_assert_eq!(tensor_algebra_transpose(&left_transposed), &left[..]);

        let sum: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
        let expected: Vec<_> = left_transposed
            .iter()
            .zip(&right_transposed)
            .map(|(&a, &b)| a + b)
            .collect();
        prop_assert_eq!(tensor_algebra_transpose(&sum), expected);
    }
}
