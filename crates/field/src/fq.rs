//! Arithmetic modulo a prime fixed at compile time.

use crate::helpers::{self, FieldMetadata};
use crypto_primitives::{ConstBaseField, LiftElement, WithAssociatedInteger};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{
    Bounded, CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, Inv, One, Pow,
    Zero,
};
#[cfg(feature = "rand")]
use rand::{
    Rng, RngExt,
    distr::{Distribution, StandardUniform},
};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// `2^100 - 15`, prime.
pub const Q100: u128 = (1u128 << 100) - 15;

/// The modulus used unless a caller picks another.
pub type FqDefault = Fq<Q100>;

/// An element of `Z/QZ`, held reduced.
///
/// `Q` must be an odd prime below `2^126`. The upper bound is what lets the
/// reduction finish in two conditional subtractions of a 128-bit remainder:
/// Barrett leaves a value below `3Q`, which has to fit a `u128`. That admits
/// anything up to `floor((2^128 - 1)/3)`, just over `2^126.41`; the bound is
/// rounded down to a power of two.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, InfallibleCheckedOp,
)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
pub struct Fq<const Q: u128>(u128);

impl<const Q: u128> Fq<Q> {
    pub const META: FieldMetadata = {
        assert!(Q >= 3, "modulus must be at least 3");
        // `Fq` claims to be a prime field, so it enforces that itself
        assert!(helpers::is_prime(Q), "modulus must be prime");
        FieldMetadata::new(Q)
    };

    /// Constructs an element from two little-endian `u64` limbs and reduces it
    /// modulo `Q`.
    pub fn from_limbs(low: u64, high: u64) -> Self {
        Self::from(u128::from(low) | (u128::from(high) << 64))
    }
}

impl<const Q: u128> Display for Fq<Q> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{} (mod {})", self.0, Q)
    }
}

#[cfg(feature = "rand")]
impl<const Q: u128> Distribution<Fq<Q>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Fq<Q> {
        // Force validation of the const-generic modulus before using it below.
        let _ = Fq::<Q>::META;

        // A u128 range is not generally an exact multiple of Q. Reject its
        // incomplete final interval before reducing so every residue has the
        // same number of preimages.
        let rejection_remainder = (u128::MAX % Q + 1) % Q;
        let max_accepted = u128::MAX - rejection_remainder;

        loop {
            let candidate = rng.random::<u128>();
            if candidate <= max_accepted {
                return Fq::from(candidate);
            }
        }
    }
}

impl<const Q: u128> Zero for Fq<Q> {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<const Q: u128> One for Fq<Q> {
    fn one() -> Self {
        Self::ONE
    }
}

impl<const Q: u128> ConstZero for Fq<Q> {
    const ZERO: Self = Self(0);
}

impl<const Q: u128> ConstOne for Fq<Q> {
    const ONE: Self = Self(1);
}

/// Reduces its input, so any `u128` is accepted.
impl<const Q: u128> From<u128> for Fq<Q> {
    fn from(value: u128) -> Self {
        let _ = Self::META;
        Self(value % Q)
    }
}

impl<const Q: u128> From<&u128> for Fq<Q> {
    fn from(value: &u128) -> Self {
        Self::from(*value)
    }
}

/// Reduces its input, so any `u64` is accepted.
impl<const Q: u128> From<u64> for Fq<Q> {
    fn from(value: u64) -> Self {
        Self::from(u128::from(value))
    }
}

impl<const Q: u128> From<bool> for Fq<Q> {
    fn from(value: bool) -> Self {
        if value { Self::ONE } else { Self::ZERO }
    }
}

impl<const Q: u128> Neg for Fq<Q> {
    type Output = Self;
    fn neg(self) -> Self {
        let _ = Self::META;
        Self(if self.0 == 0 { 0 } else { Q - self.0 })
    }
}

impl<const Q: u128> Add for Fq<Q> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let _ = Self::META;
        // Both operands are below `Q < 2^126`, so the sum cannot wrap.
        let s = self.0 + rhs.0;
        Self(if s >= Q { s - Q } else { s })
    }
}

impl<const Q: u128> Sub for Fq<Q> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let _ = Self::META;
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0 + Q - rhs.0
        })
    }
}

impl<const Q: u128> Mul for Fq<Q> {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self {
        self.0 = Self::META.mul(self.0, rhs.0);
        self
    }
}

impl<const Q: u128> Add<&Fq<Q>> for Fq<Q> {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        self.add(*rhs)
    }
}

impl<const Q: u128> Sub<&Fq<Q>> for Fq<Q> {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        self.sub(*rhs)
    }
}

impl<const Q: u128> Mul<&Fq<Q>> for Fq<Q> {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        self.mul(*rhs)
    }
}

impl<const Q: u128> AddAssign for Fq<Q> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const Q: u128> SubAssign for Fq<Q> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const Q: u128> MulAssign for Fq<Q> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const Q: u128> AddAssign<&Fq<Q>> for Fq<Q> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<const Q: u128> SubAssign<&Fq<Q>> for Fq<Q> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
}

impl<const Q: u128> MulAssign<&Fq<Q>> for Fq<Q> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl<const Q: u128> Sum for Fq<Q> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<'a, const Q: u128> Sum<&'a Fq<Q>> for Fq<Q> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<const Q: u128> Product for Fq<Q> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}

impl<'a, const Q: u128> Product<&'a Fq<Q>> for Fq<Q> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}

// Required by `crypto_primitives::Field`, not implemented yet.

impl<const Q: u128> Pow<u32> for Fq<Q> {
    type Output = Self;
    fn pow(self, _rhs: u32) -> Self {
        unimplemented!("exponentiation is not implemented yet")
    }
}

impl<const Q: u128> Pow<u128> for Fq<Q> {
    type Output = Self;
    fn pow(self, _rhs: u128) -> Self {
        unimplemented!("exponentiation is not implemented yet")
    }
}

impl<const Q: u128> Pow<&u128> for Fq<Q> {
    type Output = Self;
    fn pow(self, _rhs: &u128) -> Self {
        unimplemented!("exponentiation is not implemented yet")
    }
}

impl<const Q: u128> Inv for Fq<Q> {
    type Output = Option<Self>;
    fn inv(self) -> Option<Self> {
        unimplemented!("inversion is not implemented yet")
    }
}

impl<const Q: u128> Div for Fq<Q> {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self {
        unimplemented!("division is not implemented yet")
    }
}

impl<const Q: u128> Div<&Fq<Q>> for Fq<Q> {
    type Output = Self;
    fn div(self, _rhs: &Self) -> Self {
        unimplemented!("division is not implemented yet")
    }
}

impl<const Q: u128> DivAssign for Fq<Q> {
    fn div_assign(&mut self, _rhs: Self) {
        unimplemented!("division is not implemented yet")
    }
}

impl<const Q: u128> DivAssign<&Fq<Q>> for Fq<Q> {
    fn div_assign(&mut self, _rhs: &Self) {
        unimplemented!("division is not implemented yet")
    }
}

/// Exponents live in `[0, Q)`, and `Q` fits a `u128`.
impl<const Q: u128> WithAssociatedInteger for Fq<Q> {
    type Integer = u128;
}

/// The least non-negative representative, in `[0, Q)`.
///
/// Which representative a lift picks is a convention — centred ones are the
/// usual alternative — and it shows wherever a field element becomes an
/// integer again, so it is fixed here.
impl<const Q: u128> LiftElement<u128> for Fq<Q> {
    fn lift(&self) -> u128 {
        self.0
    }
}

impl<const Q: u128> Bounded for Fq<Q> {
    fn min_value() -> Self {
        Self::ZERO
    }

    /// The largest residue, `Q - 1`.
    fn max_value() -> Self {
        let _ = Self::META;
        Self(Q - 1)
    }
}

impl<const Q: u128> ConstBaseField for Fq<Q> {
    const MODULUS: Self::Integer = {
        let _ = Self::META;
        Q
    };
    const MODULUS_MINUS_ONE_DIV_TWO: Self::Integer = (Q - 1) / 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::tests::mulmod_reference;
    use crypto_primitives::{BaseField, WithExtensionDegree};
    use rand_core::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    /// A prime small enough to check every pair of operands.
    const SMALL: u128 = 251;

    #[test]
    fn ensure_traits() {
        fn assert_impl<T: ConstBaseField>() {}
        assert_impl::<FqDefault>();
    }

    #[test]
    fn base_field_metadata() {
        assert_eq!(FqDefault::META.modulus, Q100);
        assert_eq!(FqDefault::MODULUS_MINUS_ONE_DIV_TWO, (Q100 - 1) / 2);
        assert_eq!(FqDefault::modulus(), Q100);
        assert_eq!(FqDefault::min_value(), FqDefault::ZERO);
        assert_eq!(FqDefault::max_value().lift(), Q100 - 1);
        assert_eq!(FqDefault::extension_degree(), 1);
    }

    fn u128_of(rng: &mut Pcg64) -> u128 {
        (rng.next_u64() as u128) << 64 | rng.next_u64() as u128
    }

    #[test]
    fn barrett_matches_long_division() {
        let mut rng = Pcg64::seed_from_u64(402);
        for _ in 0..4096 {
            let (a, b) = (u128_of(&mut rng) % Q100, u128_of(&mut rng) % Q100);
            let got = (Fq::<Q100>::from(a) * Fq::<Q100>::from(b)).lift();
            assert_eq!(got, mulmod_reference(a, b, Q100), "{a} * {b}");
        }
        // The extremes of the input range, where a quotient estimate that is
        // one too small or one too large would show up.
        for &a in &[0, 1, Q100 - 1, Q100 / 2] {
            for &b in &[0, 1, Q100 - 1, Q100 / 2] {
                let got = (Fq::<Q100>::from(a) * Fq::<Q100>::from(b)).lift();
                assert_eq!(got, mulmod_reference(a, b, Q100), "{a} * {b}");
            }
        }
    }

    #[test]
    fn exhaustive_over_a_small_modulus() {
        for a in 0..SMALL {
            for b in 0..SMALL {
                let (x, y) = (Fq::<SMALL>::from(a), Fq::<SMALL>::from(b));
                assert_eq!((x * y).lift(), a * b % SMALL, "{a} * {b}");
                assert_eq!((x + y).lift(), (a + b) % SMALL, "{a} + {b}");
                assert_eq!((x - y).lift(), (a + SMALL - b) % SMALL, "{a} - {b}");
            }
        }
    }

    #[test]
    fn from_u128_reduces() {
        assert_eq!(Fq::<Q100>::from(Q100).lift(), 0);
        assert_eq!(Fq::<Q100>::from(Q100 + 1).lift(), 1);
        assert_eq!(Fq::<Q100>::from(u128::MAX).lift(), u128::MAX % Q100);
        assert_eq!(Fq::<Q100>::ONE.lift(), 1);
        assert!(Fq::<Q100>::ZERO.is_zero());
    }

    #[test]
    fn from_limbs_packs_little_endian_and_reduces() {
        assert_eq!(Fq::<Q100>::from_limbs(1, 0).lift(), 1);
        assert_eq!(Fq::<Q100>::from_limbs(0, 1).lift(), (1u128 << 64) % Q100);
        assert_eq!(
            Fq::<Q100>::from_limbs(u64::MAX, u64::MAX).lift(),
            u128::MAX % Q100
        );
    }

    #[cfg(feature = "rand")]
    #[test]
    fn standard_uniform_samples_canonical_field_elements() {
        let mut rng = Pcg64::seed_from_u64(404);

        for _ in 0..4096 {
            let sample: FqDefault = rng.random();
            assert!(sample.lift() < Q100);
        }
    }

    #[test]
    fn bits_matches_the_modulus() {
        assert_eq!(Fq::<Q100>::META.bits, 100);
        assert_eq!(Fq::<SMALL>::META.bits, 8);
        // The defining bracket, `2^(BITS-1) <= Q < 2^BITS`.
        const { assert!(1u128 << (Fq::<Q100>::META.bits - 1) <= Q100) };
        const { assert!(Q100 < 1u128 << Fq::<Q100>::META.bits) };
    }

    #[test]
    fn ring_axioms() {
        let mut rng = Pcg64::seed_from_u64(403);
        for _ in 0..512 {
            let a = Fq::<Q100>::from(u128_of(&mut rng));
            let b = Fq::<Q100>::from(u128_of(&mut rng));
            let c = Fq::<Q100>::from(u128_of(&mut rng));
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!(a - a, Fq::ZERO);
            assert_eq!(a + b - b, a);
            assert_eq!(a * Fq::ONE, a);
            assert_eq!(a * Fq::ZERO, Fq::ZERO);
        }
    }

    /// `Q100` has to be prime for this to be a field at all. Miller–Rabin over
    /// the reference multiply rather than over `Fq`, so the check does not
    /// depend on the code it is vouching for.
    #[test]
    fn q100_is_prime() {
        fn powmod(mut base: u128, mut exp: u128, q: u128) -> u128 {
            let mut acc = 1u128;
            while exp != 0 {
                if exp & 1 == 1 {
                    acc = mulmod_reference(acc, base, q);
                }
                base = mulmod_reference(base, base, q);
                exp >>= 1;
            }
            acc
        }

        let q = Q100;
        let mut d = q - 1;
        let mut s = 0;
        while d.is_multiple_of(2) {
            d /= 2;
            s += 1;
        }
        // Deterministic for any modulus below 3.3e24; `Q100` is larger, so this
        // is a strong probable-prime test rather than a proof.
        for a in [2u128, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            let mut x = powmod(a, d, q);
            if x == 1 || x == q - 1 {
                continue;
            }
            let mut witnessed = false;
            for _ in 0..s - 1 {
                x = mulmod_reference(x, x, q);
                if x == q - 1 {
                    witnessed = true;
                    break;
                }
            }
            assert!(witnessed, "{a} witnesses that Q100 is composite");
        }
    }
}
