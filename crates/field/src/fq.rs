//! Arithmetic modulo a prime fixed at compile time.

use crypto_primitives::{ConstBaseField, LiftElement, WithAssociatedInteger};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{
    Bounded, CheckedAdd, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, Inv, One, Pow,
    Zero,
};
#[cfg(feature = "rand")]
use rand::{
    Rng,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
pub struct Fq<const Q: u128>(u128);

/// Full 128x128 -> 256-bit product as `(low, high)`.
const fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    let (a_lo, a_hi) = (a as u64 as u128, a >> 64);
    let (b_lo, b_hi) = (b as u64 as u128, b >> 64);

    let ll = a_lo * b_lo;
    let hh = a_hi * b_hi;
    let (mid, mid_carry) = (a_lo * b_hi).overflowing_add(a_hi * b_lo);

    let (lo, lo_carry) = ll.overflowing_add(mid << 64);
    let hi = hh + (mid >> 64) + ((mid_carry as u128) << 64) + lo_carry as u128;
    (lo, hi)
}

/// `(lo, hi) >> n` for `n < 128`, keeping the low 128 bits. Every use here
/// shifts far enough that nothing above them survives.
const fn shr_wide(lo: u128, hi: u128, n: u32) -> u128 {
    if n == 0 {
        lo
    } else {
        (lo >> n) | (hi << (128 - n))
    }
}

/// `floor(2^(2k) / q)` by binary long division, `k` being `q`'s bit length.
///
/// Barrett's precomputed reciprocal. The numerator has a single set bit, so
/// each step shifts the remainder up and subtracts `q` when it fits.
const fn barrett_mu(q: u128, k: u32) -> u128 {
    let mut rem = 0u128;
    let mut quo = 0u128;
    let mut i = 2 * k;
    loop {
        rem = (rem << 1) | (i == 2 * k) as u128;
        let fits = rem >= q;
        if fits {
            rem -= q;
        }
        quo = (quo << 1) | fits as u128;
        if i == 0 {
            return quo;
        }
        i -= 1;
    }
}

/// The first thirteen primes: the standard deterministic Miller-Rabin base set
/// below `2^81.4`.
const PRIMALITY_BASES: [u128; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// `(a + b) mod q`, for `a, b < q < 2^126`. The sum stays below `2^127`.
const fn add_mod(a: u128, b: u128, q: u128) -> u128 {
    let sum = a + b;
    if sum >= q { sum - q } else { sum }
}

/// `(a * b) mod q` by doubling, avoiding the 256-bit product a u128 cannot
/// hold. Barrett is not an option here: `MU` depends on `BITS`, which is the
/// constant this feeds.
const fn mul_mod(a: u128, b: u128, q: u128) -> u128 {
    let mut result = 0u128;
    let mut addend = a % q;
    let mut remaining = b;
    while remaining != 0 {
        if remaining & 1 == 1 {
            result = add_mod(result, addend, q);
        }
        addend = add_mod(addend, addend, q);
        remaining >>= 1;
    }
    result
}

/// `(base ^ exponent) mod q`, by square-and-multiply.
const fn pow_mod(base: u128, exponent: u128, q: u128) -> u128 {
    let mut result = 1u128 % q;
    let mut square = base % q;
    let mut remaining = exponent;
    while remaining != 0 {
        if remaining & 1 == 1 {
            result = mul_mod(result, square, q);
        }
        square = mul_mod(square, square, q);
        remaining >>= 1;
    }
    result
}

/// Whether `candidate` is prime.
///
/// `const` because [`Fq`] asserts it on its own modulus, which makes a
/// composite `Q` a build failure rather than a type that quietly is not a
/// field. The loops are written out for the same reason: iterator combinators
/// are not available in a const context.
///
/// # What this does not decide
///
/// Miller-Rabin against a fixed base set is **proven** only for
/// `n < 3_317_044_064_679_887_385_961_981`, about `2^81.4`. Moduli above that
/// are not decided by any theorem here, and because the bases are public,
/// someone choosing `Q` could in principle construct a composite that passes
/// all of them. Since `Q` is a compile-time constant, doing so means editing
/// the source rather than forging a proof.
const fn is_prime(candidate: u128) -> bool {
    if candidate < 2 {
        return false;
    }

    let mut index = 0;
    while index < PRIMALITY_BASES.len() {
        let base = PRIMALITY_BASES[index];
        if candidate == base {
            return true;
        }
        if candidate.is_multiple_of(base) {
            return false;
        }
        index += 1;
    }

    // `candidate - 1 = odd * 2^shift`.
    let shift = (candidate - 1).trailing_zeros();
    let odd = (candidate - 1) >> shift;

    let mut index = 0;
    while index < PRIMALITY_BASES.len() {
        let mut witness = pow_mod(PRIMALITY_BASES[index], odd, candidate);
        if witness != 1 && witness != candidate - 1 {
            let mut round = 1;
            loop {
                if round >= shift {
                    return false;
                }
                witness = mul_mod(witness, witness, candidate);
                if witness == candidate - 1 {
                    break;
                }
                round += 1;
            }
        }
        index += 1;
    }
    true
}

impl<const Q: u128> Fq<Q> {
    /// Bit length of the modulus, derived so it cannot disagree with `Q`.
    ///
    /// Its asserts are the only check on `Q`, and an associated constant is
    /// evaluated where it is used, so an operation that does not need the
    /// value reads it anyway rather than accept a modulus out of range.
    pub const BITS: u32 = {
        assert!(Q >= 3, "modulus must be at least 3");
        assert!(Q % 2 == 1, "modulus must be odd");
        assert!(Q < 1u128 << 126, "modulus must be below 2^126");
        // `Fq` claims to be a prime field, so it enforces that itself rather
        // than trusting whoever names the constant.
        assert!(is_prime(Q), "modulus must be prime");
        128 - Q.leading_zeros()
    };

    const MU: u128 = barrett_mu(Q, Self::BITS);

    /// Barrett reduction, Handbook of Applied Cryptography Algorithm 14.42,
    /// after Barrett, CRYPTO '86, LNCS 263:311-323: estimate the quotient
    /// from the top bits of `x` and the precomputed reciprocal, subtract,
    /// then correct. The estimate is never more than two too small, so two
    /// conditional subtractions suffice.
    ///
    /// That algorithm is stated for a radix above 3 and takes the difference
    /// modulo `b^(k+1)`, which is wide enough to hold `3Q`. Here the radix is
    /// 2, where it is not, so the difference is taken modulo `2^128` instead.
    /// Hence the bound on `Q`.
    const fn reduce_wide(lo: u128, hi: u128) -> Self {
        let k = Self::BITS;
        let q1 = shr_wide(lo, hi, k - 1);
        let (q2_lo, q2_hi) = mul_wide(q1, Self::MU);
        let q3 = shr_wide(q2_lo, q2_hi, k + 1);

        let mut r = lo.wrapping_sub(mul_wide(q3, Q).0);
        if r >= Q {
            r -= Q;
        }
        if r >= Q {
            r -= Q;
        }
        Self(r)
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
        let _ = Fq::<Q>::BITS;

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
        let _ = Self::BITS;
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
        let _ = Self::BITS;
        Self(if self.0 == 0 { 0 } else { Q - self.0 })
    }
}

impl<const Q: u128> Add for Fq<Q> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let _ = Self::BITS;
        // Both operands are below `Q < 2^126`, so the sum cannot wrap.
        let s = self.0 + rhs.0;
        Self(if s >= Q { s - Q } else { s })
    }
}

impl<const Q: u128> Sub for Fq<Q> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let _ = Self::BITS;
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0 + Q - rhs.0
        })
    }
}

impl<const Q: u128> Mul for Fq<Q> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (lo, hi) = mul_wide(self.0, rhs.0);
        Self::reduce_wide(lo, hi)
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
        let _ = Self::BITS;
        Self(Q - 1)
    }
}

impl<const Q: u128> ConstBaseField for Fq<Q> {
    const MODULUS: Self::Integer = {
        let _ = Self::BITS;
        Q
    };
    const MODULUS_MINUS_ONE_DIV_TWO: Self::Integer = (Q - 1) / 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{BaseField, WithExtensionDegree};
    #[cfg(feature = "rand")]
    use rand::Rng;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    /// A prime small enough to check every pair of operands.
    const SMALL: u128 = 251;

    #[test]
    fn primality_agrees_with_trial_division() {
        for candidate in 0u128..2_000 {
            let expected = candidate >= 2
                && (2..candidate)
                    .take_while(|d| d * d <= candidate)
                    .all(|d| !candidate.is_multiple_of(d));
            assert_eq!(is_prime(candidate), expected, "{candidate}");
        }
    }

    #[test]
    fn primality_rejects_what_a_fermat_test_would_admit() {
        // Carmichael numbers pass every Fermat test; Miller-Rabin does not.
        for carmichael in [561u128, 41_041, 825_265] {
            assert!(!is_prime(carmichael), "{carmichael}");
        }
    }

    #[test]
    fn primality_holds_at_the_widths_the_moduli_use() {
        assert!(is_prime((1 << 100) - 15));
        assert!(is_prime((1 << 108) - 59));
        assert!(is_prime((1 << 114) - 11));
        // A semiprime with no factor small enough for the trial-division pass.
        assert!(!is_prime(((1u128 << 54) - 33) * ((1u128 << 53) - 111)));
        // Every odd value between the largest prime below 2^114 and 2^114.
        for offset in (1..11).step_by(2) {
            assert!(!is_prime((1 << 114) - offset), "2^114 - {offset}");
        }
    }

    #[test]
    fn ensure_traits() {
        fn assert_impl<T: ConstBaseField>() {}
        assert_impl::<FqDefault>();
    }

    #[test]
    fn base_field_metadata() {
        assert_eq!(FqDefault::MODULUS, Q100);
        assert_eq!(FqDefault::MODULUS_MINUS_ONE_DIV_TWO, (Q100 - 1) / 2);
        assert_eq!(FqDefault::modulus(), Q100);
        assert_eq!(FqDefault::min_value(), FqDefault::ZERO);
        assert_eq!(FqDefault::max_value().lift(), Q100 - 1);
        assert_eq!(FqDefault::extension_degree(), 1);
    }

    fn u128_of(rng: &mut Pcg64) -> u128 {
        (rng.next_u64() as u128) << 64 | rng.next_u64() as u128
    }

    /// `(hi:lo) mod q` by binary long division — the independent reference
    /// Barrett is checked against. Structurally unlike Barrett, which
    /// estimates a quotient and corrects.
    fn mod_reference(lo: u128, hi: u128, q: u128) -> u128 {
        let mut rem = 0u128;
        for i in (0..256).rev() {
            let bit = if i >= 128 {
                (hi >> (i - 128)) & 1
            } else {
                (lo >> i) & 1
            };
            rem = (rem << 1) | bit;
            if rem >= q {
                rem -= q;
            }
        }
        rem
    }

    fn mulmod_reference(a: u128, b: u128, q: u128) -> u128 {
        let (lo, hi) = mul_wide(a, b);
        mod_reference(lo, hi, q)
    }

    #[test]
    fn mul_wide_matches_u128_on_small_inputs() {
        let mut rng = Pcg64::seed_from_u64(401);
        for _ in 0..512 {
            let (a, b) = (rng.next_u64() as u128, rng.next_u64() as u128);
            assert_eq!(mul_wide(a, b), (a * b, 0));
        }
        assert_eq!(mul_wide(u128::MAX, u128::MAX), (1, u128::MAX - 1));
        assert_eq!(mul_wide(1u128 << 127, 2), (0, 1));
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
        assert_eq!(Fq::<Q100>::BITS, 100);
        assert_eq!(Fq::<SMALL>::BITS, 8);
        // The defining bracket, `2^(BITS-1) <= Q < 2^BITS`.
        const { assert!(1u128 << (Fq::<Q100>::BITS - 1) <= Q100) };
        const { assert!(Q100 < 1u128 << Fq::<Q100>::BITS) };
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
