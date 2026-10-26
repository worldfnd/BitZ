//! Various helpers shared across multiple field implementations.

/// The first thirteen primes: the standard deterministic Miller-Rabin base set
/// below `2^81.4`.
const PRIMALITY_BASES: [u128; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Every modulus is below `2^MAX_MODULUS_BITS`: the bound of
/// [`barrett_reduce`], which every prime field here reduces by.
pub const MAX_MODULUS_BITS: u32 = 126;

/// `Some` when trial division by [`PRIMALITY_BASES`] decides
/// `candidate`.
pub const fn sieve(candidate: u128) -> Option<bool> {
    if candidate < 2 {
        return Some(false);
    }
    let mut index = 0;
    while index < PRIMALITY_BASES.len() {
        let prime = PRIMALITY_BASES[index];
        if candidate == prime {
            return Some(true);
        }
        if candidate.is_multiple_of(prime) {
            return Some(false);
        }
        index += 1;
    }
    None
}

/// Whether `candidate`, below `2^MAX_MODULUS_BITS`, is prime.
///
/// Note that **this is not supposed to be used for testing adversarial primes**!
/// (See [`transcript::challenge::prime::sample`])
pub const fn is_prime(candidate: u128) -> bool {
    if let Some(decided) = sieve(candidate) {
        return decided;
    }
    let modulus = FieldMetadata::new(candidate);
    let mut index = 0;
    while index < PRIMALITY_BASES.len() {
        if !modulus.strong_round(PRIMALITY_BASES[index]) {
            return false;
        }
        index += 1;
    }
    true
}

/// `(a + b) mod q`, for `a, b < q < 2^127`. The sum cannot wrap.
const fn add_mod(a: u128, b: u128, q: u128) -> u128 {
    let sum = a + b;
    if sum >= q { sum - q } else { sum }
}

/// `(a * b) mod q` by doubling, for any `q` below `2^127`: slow, but
/// independent of Barrett, so it is the reference [`FieldMetadata`] is checked
/// against.
pub const fn mul_mod(a: u128, b: u128, q: u128) -> u128 {
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

/// Full 128x128 -> 256-bit product as `(low, high)`.
pub const fn mul_wide(a: u128, b: u128) -> (u128, u128) {
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
pub const fn shr_wide(lo: u128, hi: u128, n: u32) -> u128 {
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
pub const fn barrett_mu(q: u128, k: u32) -> u128 {
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

/// `(lo, hi) mod q` for `q` of `k` bits and `mu` its reciprocal from
/// [`barrett_mu`]: Barrett reduction, Handbook of Applied Cryptography
/// Algorithm 14.42, after Barrett, CRYPTO '86, LNCS 263:311-323. Estimate the
/// quotient from the top bits of the input and the precomputed reciprocal,
/// subtract, then correct. The estimate is never more than two too small, so
/// two conditional subtractions suffice.
///
/// That algorithm is stated for a radix above 3 and takes the difference
/// modulo `b^(k+1)`, which is wide enough to hold `3q`. Here the radix is 2,
/// where it is not, so the difference is taken modulo `2^128` instead. Hence
/// the bound `q < 2^126` on every modulus that comes through here.
#[inline]
pub const fn barrett_reduce(lo: u128, hi: u128, q: u128, mu: u128, k: u32) -> u128 {
    let q1 = shr_wide(lo, hi, k - 1);
    let (q2_lo, q2_hi) = mul_wide(q1, mu);
    let q3 = shr_wide(q2_lo, q2_hi, k + 1);

    let mut r = lo.wrapping_sub(mul_wide(q3, q).0);
    if r >= q {
        r -= q;
    }
    if r >= q {
        r -= q;
    }
    r
}

/// A modulus in `[2, 2^MAX_MODULUS_BITS)` with its Barrett reciprocal: the
/// arithmetic of a modulus that is a value rather than a type, in `const` and
/// at runtime alike. What [`DynField`](crate::dynamic::DynField) installs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldMetadata {
    pub modulus: u128,
    /// Bit length of the modulus.
    pub bits: u32,
    /// `floor(2^(2 * bits) / modulus)`.
    pub mu: u128,
}

impl FieldMetadata {
    pub const fn new(modulus: u128) -> Self {
        assert!(
            modulus >= 2 && modulus < 1 << MAX_MODULUS_BITS,
            "modulus out of range"
        );
        assert!(modulus == 2 || modulus % 2 == 1, "modulus > 2 must be odd");
        let bits = u128::BITS - modulus.leading_zeros();
        Self {
            modulus,
            bits,
            mu: barrett_mu(modulus, bits),
        }
    }

    /// `(a * b) mod q` for `a, b < q`.
    #[inline]
    pub const fn mul(&self, a: u128, b: u128) -> u128 {
        let (lo, hi) = mul_wide(a, b);
        barrett_reduce(lo, hi, self.modulus, self.mu, self.bits)
    }

    /// `(base ^ exponent) mod q` for `base < q`, by square-and-multiply.
    pub const fn pow(&self, base: u128, exponent: u128) -> u128 {
        let mut result = 1 % self.modulus;
        let mut bit = u128::BITS - exponent.leading_zeros();
        while bit > 0 {
            bit -= 1;
            result = self.mul(result, result);
            if (exponent >> bit) & 1 == 1 {
                result = self.mul(result, base);
            }
        }
        result
    }

    /// One strong probable-prime round of `q`, odd and at least 3, at `base`
    /// in `[1, q - 1]`: with `q - 1 = odd * 2^shift`, `base^odd` is `±1` or
    /// one of its next `shift - 1` squarings is `-1`. A prime passes every
    /// base; a composite passes at most a quarter of them.
    pub const fn strong_round(&self, base: u128) -> bool {
        let shift = (self.modulus - 1).trailing_zeros();
        let odd = (self.modulus - 1) >> shift;
        let mut witness = self.pow(base, odd);
        if witness == 1 || witness == self.modulus - 1 {
            return true;
        }
        let mut round = 1;
        while round < shift {
            witness = self.mul(witness, witness);
            if witness == self.modulus - 1 {
                return true;
            }
            round += 1;
        }
        false
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use rand_core::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

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
    fn the_sieve_decides_only_what_trial_division_can() {
        assert_eq!(sieve(0), Some(false));
        assert_eq!(sieve(1), Some(false));
        assert_eq!(sieve(2), Some(true));
        assert_eq!(sieve(41), Some(true));
        assert_eq!(sieve(43), None);
        assert_eq!(sieve(1 << 100), Some(false));
        assert_eq!(sieve((1 << 100) - 15), None);
    }

    #[test]
    fn a_strong_round_catches_what_its_base_witnesses() {
        // `2047 = 23 * 89` is a strong pseudoprime to base 2 and nothing else
        // small; `1_373_653` is one to bases 2 and 3.
        assert!(FieldMetadata::new(2047).strong_round(2));
        assert!(!FieldMetadata::new(2047).strong_round(3));
        assert!(FieldMetadata::new(1_373_653).strong_round(2));
        assert!(FieldMetadata::new(1_373_653).strong_round(3));
        assert!(!FieldMetadata::new(1_373_653).strong_round(5));
        // A prime passes every base.
        let prime = FieldMetadata::new((1 << 61) - 1);
        assert!((1..64).all(|base| prime.strong_round(base)));
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

    pub fn mulmod_reference(a: u128, b: u128, q: u128) -> u128 {
        let (lo, hi) = mul_wide(a, b);
        mod_reference(lo, hi, q)
    }

    fn u128_of(rng: &mut Pcg64) -> u128 {
        (rng.next_u64() as u128) << 64 | rng.next_u64() as u128
    }

    #[test]
    fn the_modulus_multiplies_and_exponentiates_like_the_references() {
        let mut rng = Pcg64::seed_from_u64(405);
        for q in [
            2u128,
            3,
            59,
            251,
            (1 << 100) - 15,
            (1 << 114) - 11,
            (1 << 126) - 1,
        ] {
            let modulus = FieldMetadata::new(q);
            for _ in 0..64 {
                let (a, b) = (u128_of(&mut rng) % q, u128_of(&mut rng) % q);
                assert_eq!(
                    modulus.mul(a, b),
                    mulmod_reference(a, b, q),
                    "{a} * {b} mod {q}"
                );
                assert_eq!(modulus.mul(a, b), mul_mod(a, b, q), "{a} * {b} mod {q}");
            }
            for a in [0, 1, q - 1] {
                assert_eq!(modulus.mul(a, a), mul_mod(a, a, q), "{a}^2 mod {q}");
                let mut power = 1 % q;
                for exponent in 0..20u128 {
                    assert_eq!(modulus.pow(a, exponent), power, "{a}^{exponent} mod {q}");
                    power = mul_mod(power, a, q);
                }
            }
        }
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
}
