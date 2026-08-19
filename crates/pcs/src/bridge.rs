//! Zero-copy conversion between the local field and Flock's field.

use core::mem::{align_of, offset_of, size_of};

use field::F128 as LocalF128;
use flock_core::field::F128 as FlockF128;

// Both types contain two `u64` words in the same C layout.
// These checks fail during compilation if either dependency changes its layout.
const _: () = {
    let _ = LocalF128 { lo: 0u64, hi: 0u64 };
    let _ = FlockF128 { lo: 0u64, hi: 0u64 };
    assert!(size_of::<LocalF128>() == 16);
    assert!(size_of::<FlockF128>() == 16);
    assert!(align_of::<LocalF128>() == 16);
    assert!(align_of::<FlockF128>() == 16);
    assert!(offset_of!(LocalF128, lo) == 0);
    assert!(offset_of!(FlockF128, lo) == 0);
    assert!(offset_of!(LocalF128, hi) == 8);
    assert!(offset_of!(FlockF128, hi) == 8);
};

/// Views local field elements as Flock field elements without copying.
///
/// The field differential tests verify the common polynomial basis.
#[inline(always)]
pub(crate) fn as_flock_f128s(values: &[LocalF128]) -> &[FlockF128] {
    // SAFETY: The compile-time checks prove equal size, alignment, and offsets.
    // Both types contain only `u64` fields, so every bit pattern is valid.
    // The returned shared slice cannot mutate the source slice.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn cast_preserves_pointer_length_and_words() {
        let values = [
            LocalF128::new(0, 0),
            LocalF128::new(1, 2),
            LocalF128::new(u64::MAX, 1 << 63),
        ];
        let cast = as_flock_f128s(&values);

        assert_eq!(cast.as_ptr().cast::<LocalF128>(), values.as_ptr());
        assert_eq!(cast.len(), values.len());
        for (local, flock) in values.iter().zip(cast) {
            assert_eq!((flock.lo, flock.hi), (local.lo, local.hi));
        }
    }

    #[test]
    fn cast_accepts_an_empty_slice() {
        assert!(as_flock_f128s(&[]).is_empty());
    }

    proptest! {
        #[test]
        fn cast_preserves_arbitrary_words(
            words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..64)
        ) {
            let values = words
                .iter()
                .map(|&(lo, hi)| LocalF128::new(lo, hi))
                .collect::<Vec<_>>();
            let cast = as_flock_f128s(&values);

            prop_assert_eq!(cast.as_ptr().cast::<LocalF128>(), values.as_ptr());
            prop_assert_eq!(cast.len(), values.len());
            for (local, flock) in values.iter().zip(cast) {
                prop_assert_eq!((flock.lo, flock.hi), (local.lo, local.hi));
            }
        }
    }
}
