//! Deferred-reduction accumulator.
//!
//! Reduction is a fixed pattern of shifts and XORs of the input words — no step
//! depends on the data — so it is `F_2`-linear:
//! `reduce(x + y) = reduce(x) + reduce(y)`. A sum of products can therefore be
//! accumulated unreduced and reduced once, bit-identically to reducing each
//! product. That trades one reduction per term for one per accumulator, the
//! shape of the sumcheck round bodies.

use std::ops::AddAssign;

use super::{F128, kernel};

/// An unreduced 256-bit carryless product, or a sum of several.
#[derive(Clone, Copy)]
pub struct Wide256(kernel::Wide);

impl Wide256 {
    #[inline]
    pub fn zero() -> Self {
        Self(kernel::wide_zero())
    }

    /// A reduced element seen as a wide value — the same polynomial, with
    /// nothing above `X^128`.
    #[inline]
    pub fn of(a: F128) -> Self {
        Self(kernel::wide_of(a.words()))
    }

    /// The 256-bit product of two elements, left unreduced.
    #[inline]
    pub fn mul(a: F128, b: F128) -> Self {
        Self(kernel::wide_mul(a.words(), b.words()))
    }

    #[inline]
    pub fn reduce(self) -> F128 {
        kernel::wide_reduce(self.0).into()
    }
}

impl AddAssign for Wide256 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = kernel::wide_add(self.0, rhs.0);
    }
}

#[cfg(test)]
mod tests {
    use super::super::portable;
    use super::*;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    fn f128(rng: &mut Pcg64) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    fn words4(rng: &mut Pcg64) -> [u64; 4] {
        [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ]
    }

    /// The claim the whole type rests on. Tested against `portable::reduce`
    /// over arbitrary 256-bit values, not only over ones reachable as products,
    /// since linearity is a property of the map and not of its inputs.
    #[test]
    fn reduction_is_linear() {
        let mut rng = Pcg64::seed_from_u64(201);
        for _ in 0..512 {
            let (x, y) = (words4(&mut rng), words4(&mut rng));
            let sum = portable::wide_add(x, y);
            let [lo, hi] = portable::reduce(sum);
            let ([xlo, xhi], [ylo, yhi]) = (portable::reduce(x), portable::reduce(y));
            assert_eq!([lo, hi], [xlo ^ ylo, xhi ^ yhi], "{x:?} + {y:?}");
        }
    }

    #[test]
    fn constructors_agree_with_the_field() {
        let mut rng = Pcg64::seed_from_u64(202);
        assert_eq!(Wide256::zero().reduce(), F128::ZERO);
        for _ in 0..256 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            assert_eq!(Wide256::of(a).reduce(), a);
            assert_eq!(Wide256::mul(a, b).reduce(), a * b);
        }
    }

    /// Accumulate 64 products unreduced and reduce once, against reducing each
    /// product and summing in the field.
    #[test]
    fn deferred_sum_matches_eager_sum() {
        let mut rng = Pcg64::seed_from_u64(203);
        for _ in 0..64 {
            let mut wide = Wide256::zero();
            let mut eager = F128::ZERO;
            for _ in 0..64 {
                let (a, b) = (f128(&mut rng), f128(&mut rng));
                wide += Wide256::mul(a, b);
                eager += a * b;
            }
            assert_eq!(wide.reduce(), eager);
        }
    }

    /// As with the multiply, the vector accumulator is pinned to the portable
    /// one — here before reduction as well as after, so an error in
    /// `clmul_256` that the fold happens to cancel still shows up.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_wide_matches_portable() {
        use super::super::aarch64;

        let mut rng = Pcg64::seed_from_u64(204);
        let mut neon = aarch64::wide_zero();
        let mut port = portable::wide_zero();
        for _ in 0..512 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            let (aw, bw) = (a.words(), b.words());

            let (n, p) = (aarch64::wide_mul(aw, bw), portable::wide_mul(aw, bw));
            assert_eq!(aarch64::wide_words(n), p, "product of {a:?} and {b:?}");

            neon = aarch64::wide_add(neon, n);
            port = portable::wide_add(port, p);
            assert_eq!(aarch64::wide_words(neon), port);
        }
        assert_eq!(aarch64::wide_reduce(neon), portable::wide_reduce(port));

        let a = f128(&mut rng);
        assert_eq!(
            aarch64::wide_words(aarch64::wide_of(a.words())),
            portable::wide_of(a.words())
        );
    }
}
