//! `GF(2^128) = F_2[X] / <X^128 + X^7 + X^2 + X + 1>` — the GHASH field.
//!
//! Bit `i` of the little-endian value `hi:lo` is the coefficient of `X^i`:
//! `lo` carries `X^0..X^63`, `hi` carries `X^64..X^127`. This is *not* the
//! reflected byte order NIST SP 800-38D uses, so published GHASH vectors apply
//! only through a bit-reversal.
//!
//! Layout matches flock's `F128`, so converting between the two is a field
//! copy.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// Always compiled: the active kernel where no carryless-multiply instruction
// exists, and the oracle the SIMD kernels are tested against. On aarch64 only
// tests call it, hence the allow.
#[cfg_attr(all(target_arch = "aarch64", target_feature = "aes"), allow(dead_code))]
mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

mod pow;
mod wide;

pub use pow::{FixedBasePow, MULT_ORDER, ORDER_PRIME_FACTORS, is_generator, smallest_generator};
pub use wide::Wide256;

// The multiply and square in use on this target. The gate is `aes`, not `neon`:
// `pmull` is a crypto extension, on by default for `aarch64-apple-darwin` but
// not for `aarch64-unknown-linux-gnu`, which silently gets `portable`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use aarch64 as kernel;
#[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
use portable as kernel;

/// Which kernel this build selected. A timing means nothing without it.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub const KERNEL: &str = "neon";
#[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
pub const KERNEL: &str = "portable";

/// Low bits of the reduction polynomial: `X^128 = X^7 + X^2 + X + 1`.
pub const REDUCTION: u64 = 0x87;

/// An element of `GF(2^128)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct F128 {
    pub lo: u64,
    pub hi: u64,
}

impl F128 {
    pub const ZERO: Self = Self::new(0, 0);
    pub const ONE: Self = Self::new(1, 0);
    /// `X`, whose multiplicative order is the full `2^128 - 1` — checked by
    /// [`is_generator`], not assumed. `X` in AES's `GF(2^8)` has order 51 of
    /// 255.
    pub const GENERATOR: Self = Self::new(2, 0);

    pub const fn new(lo: u64, hi: u64) -> Self {
        Self { lo, hi }
    }

    pub const fn is_zero(self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    #[inline]
    pub fn square(self) -> Self {
        kernel::square(self.words()).into()
    }

    /// Multiply by `X`: a shift and a conditional fold, cheaper than the
    /// general multiply.
    pub const fn mul_x(self) -> Self {
        let [lo, hi] = portable::mul_x(self.words());
        Self { lo, hi }
    }

    /// Canonical encoding: 16 bytes little-endian, `lo` first — the encoding
    /// flock's transcript absorbs.
    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.lo.to_le_bytes());
        out[8..].copy_from_slice(&self.hi.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        let mut lo = [0u8; 8];
        let mut hi = [0u8; 8];
        lo.copy_from_slice(&bytes[..8]);
        hi.copy_from_slice(&bytes[8..]);
        Self::new(u64::from_le_bytes(lo), u64::from_le_bytes(hi))
    }

    const fn words(self) -> [u64; 2] {
        [self.lo, self.hi]
    }
}

impl From<[u64; 2]> for F128 {
    #[inline]
    fn from(words: [u64; 2]) -> Self {
        Self::new(words[0], words[1])
    }
}

impl From<bool> for F128 {
    fn from(bit: bool) -> Self {
        Self::new(bit as u64, 0)
    }
}

/// Bit `i` of `value` becomes the coefficient of `X^i` — an injection of bit
/// patterns, not a ring homomorphism.
impl From<u128> for F128 {
    fn from(value: u128) -> Self {
        Self::new(value as u64, (value >> 64) as u64)
    }
}

impl Add for F128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.lo ^ rhs.lo, self.hi ^ rhs.hi)
    }
}

impl AddAssign for F128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

/// Characteristic 2: subtraction is addition.
impl Sub for F128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.lo ^ rhs.lo, self.hi ^ rhs.hi)
    }
}

impl SubAssign for F128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

impl Neg for F128 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl Mul for F128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        kernel::mul(self.words(), rhs.words()).into()
    }
}

impl MulAssign for F128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    /// A random field element; the tests only need uniform 128-bit words.
    fn f128(rng: &mut Pcg64) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    #[test]
    fn layout_matches_flock() {
        assert_eq!(size_of::<F128>(), 16);
        assert_eq!(align_of::<F128>(), 16);
    }

    #[test]
    fn zero_and_one_are_identities() {
        let mut rng = Pcg64::seed_from_u64(1);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(a + F128::ZERO, a);
            assert_eq!(a * F128::ONE, a);
            assert_eq!(a * F128::ZERO, F128::ZERO);
        }
    }

    #[test]
    fn addition_is_xor_and_self_inverse() {
        let mut rng = Pcg64::seed_from_u64(2);
        for _ in 0..256 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            assert_eq!(a + b, F128::new(a.lo ^ b.lo, a.hi ^ b.hi));
            assert_eq!(a + a, F128::ZERO);
            assert_eq!(a - b, a + b);
            assert_eq!(-a, a);
        }
    }

    #[test]
    fn multiplication_is_commutative_and_associative() {
        let mut rng = Pcg64::seed_from_u64(3);
        for _ in 0..256 {
            let (a, b, c) = (f128(&mut rng), f128(&mut rng), f128(&mut rng));
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
        }
    }

    #[test]
    fn multiplication_distributes_over_addition() {
        let mut rng = Pcg64::seed_from_u64(4);
        for _ in 0..256 {
            let (a, b, c) = (f128(&mut rng), f128(&mut rng), f128(&mut rng));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn square_matches_self_multiply() {
        let mut rng = Pcg64::seed_from_u64(5);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn mul_x_matches_multiply_by_generator() {
        let mut rng = Pcg64::seed_from_u64(6);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(a.mul_x(), a * F128::GENERATOR);
        }
    }

    /// These pin down the reduction polynomial *and* the word order at once:
    /// get either wrong and at least one fails.
    #[test]
    fn reduction_boundary_products() {
        let x = F128::GENERATOR;
        let x_63 = F128::new(1 << 63, 0);
        let x_64 = F128::new(0, 1);
        let x_127 = F128::new(0, 1 << 63);

        assert_eq!(x * x_63, x_64); // crosses the word boundary
        assert_eq!(x * x_127, F128::new(REDUCTION, 0)); // crosses X^128
        assert_eq!(x_64 * x_64, F128::new(REDUCTION, 0)); // the same, from both halves
    }

    /// Squaring is the Frobenius endomorphism, hence `F_2`-linear.
    #[test]
    fn squaring_is_additive() {
        let mut rng = Pcg64::seed_from_u64(7);
        for _ in 0..256 {
            let (a, b) = (f128(&mut rng), f128(&mut rng));
            assert_eq!((a + b).square(), a.square() + b.square());
        }
    }

    #[test]
    fn byte_encoding_round_trips() {
        let mut rng = Pcg64::seed_from_u64(8);
        for _ in 0..256 {
            let a = f128(&mut rng);
            assert_eq!(F128::from_bytes(a.to_bytes()), a);
        }
        assert_eq!(F128::ONE.to_bytes()[0], 1);
        assert_eq!(F128::new(0, 1).to_bytes()[8], 1);
    }

    /// The SIMD kernels must agree with the portable pipeline bit for bit, on
    /// the boundary cases as well as on random input.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_matches_portable() {
        let edges = [
            F128::ZERO,
            F128::ONE,
            F128::GENERATOR,
            F128::new(REDUCTION, 0),
            F128::new(1 << 63, 0),
            F128::new(0, 1),
            F128::new(0, 1 << 63),
            F128::new(u64::MAX, u64::MAX),
        ];

        let mut rng = Pcg64::seed_from_u64(9);
        let mut cases: Vec<(F128, F128)> = Vec::new();
        for &a in &edges {
            for &b in &edges {
                cases.push((a, b));
            }
            for _ in 0..64 {
                cases.push((a, f128(&mut rng)));
                cases.push((f128(&mut rng), a));
            }
        }
        for _ in 0..2048 {
            cases.push((f128(&mut rng), f128(&mut rng)));
        }

        for (a, b) in cases {
            assert_eq!(
                aarch64::mul(a.words(), b.words()),
                portable::mul(a.words(), b.words()),
                "multiply disagrees on {a:?} * {b:?}"
            );
            assert_eq!(
                aarch64::square(a.words()),
                portable::square(a.words()),
                "square disagrees on {a:?}"
            );
            // The squaring runs are separate loops on each side, so they are
            // checked against each other rather than only against `square`.
            for k in [0, 1, 6, 24, 48, 127] {
                assert_eq!(
                    aarch64::square_n(a.words(), k),
                    portable::square_n(a.words(), k),
                    "square_n({k}) disagrees on {a:?}"
                );
            }
        }
    }

    #[test]
    fn u128_conversion_is_the_bit_pattern() {
        assert_eq!(F128::from(1u128), F128::ONE);
        assert_eq!(F128::from(1u128 << 64), F128::new(0, 1));
        assert_eq!(F128::from(true), F128::ONE);
        assert_eq!(F128::from(false), F128::ZERO);
    }
}
