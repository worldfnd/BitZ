//! Public prime sampling from the challenge stream.
//!
//! Mirrors `f2z-pcs`: `ext_proj::sample_prime_context` fixes a whole-search
//! budget from the interval, then `field::sample_prime_public` draws uniform
//! odd candidates, sieves them by the primes up to 53 and runs Miller–Rabin
//! with bases drawn from the same stream. Everything here is public, so the
//! arithmetic is variable-time.

use std::ops::RangeInclusive;
use thiserror::Error;

/// The whole search accepts a composite with probability below
/// `2^-SECURITY_BITS`: `f2z-pcs` targets 128 bits and adds 16 of slack.
const SECURITY_BITS: u32 = 144;

/// Trial-division sieve.
const SMALL_PRIMES_TABLE: [u128; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

/// Why a draw produced no prime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PrimeSamplingError {
    #[error("prime interval is empty: min {min} exceeds max {max}")]
    InvalidInterval { min: u128, max: u128 },
    #[error("prime interval [{min}, {max}] contains no odd candidate")]
    NoOddCandidate { min: u128, max: u128 },
    #[error("no prime found in [{min}, {max}] after {attempts} transcript candidates")]
    Exhausted { min: u128, max: u128, attempts: u64 },
}

/// The whole-search budget of one draw, a function of the interval alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchPolicy {
    /// Candidates drawn before the search gives up.
    pub max_candidates: u64,
    /// Miller–Rabin bases per candidate that passes the sieve.
    pub rounds: u32,
}

impl SearchPolicy {
    /// 64 candidates per bit of `max`, one if `first` is the only odd candidate;
    /// and enough rounds that the union bound over every candidate keeps the composite
    /// acceptance of the search below `2^-SECURITY_BITS`.
    pub fn for_interval(first: u128, max: u128) -> Self {
        let last_odd = max - u128::from(max & 1 == 0);
        let max_candidates = if first == last_odd {
            1
        } else {
            64 * u64::from(u128::BITS - max.leading_zeros())
        };
        let candidate_bits = u64::BITS - (max_candidates - 1).leading_zeros();
        Self {
            max_candidates,
            rounds: (SECURITY_BITS + candidate_bits).div_ceil(2),
        }
    }
}

/// The primes of exactly `bits` bits, `[2^(bits-1), 2^bits)`.
pub fn interval(bits: u32) -> RangeInclusive<u128> {
    assert!((1..=u128::BITS).contains(&bits), "prime width out of range");
    let top = 1u128 << (bits - 1);
    top..=top | (top - 1)
}

/// A prime from `interval` by successive `u128` draws, the same on both sides:
/// the fingerprint prime of 5. "An end-to-end BitZ-based SNARK over any
/// finitely generated ring", Step 2, drawn once per proof after the commitment
/// and the statement are absorbed and before the PIOP.
///
/// Odd candidates are uniform in the interval. Each passes the sieve, then
/// [`SearchPolicy::rounds`] Miller–Rabin bases uniform in `[2, candidate - 2]`,
/// every base a fresh draw. Candidates and bases are drawn by masking to the
/// bit length of their range and rejecting above it, so at most half of the
/// draws are wasted. The bounded search is not exactly uniform over primes.
pub fn sample(
    mut next_u128: impl FnMut() -> u128,
    interval: &RangeInclusive<u128>,
) -> Result<u128, PrimeSamplingError> {
    let (min, max) = (*interval.start(), *interval.end());
    if min > max {
        return Err(PrimeSamplingError::InvalidInterval { min, max });
    }
    let first = min.max(3) | 1;
    if first > max {
        return Err(PrimeSamplingError::NoOddCandidate { min, max });
    }
    let policy = SearchPolicy::for_interval(first, max);
    // The odd candidates are `first + 2 * index` for `index < count`.
    let count = (max - first) / 2 + 1;
    for _ in 0..policy.max_candidates {
        let candidate = first + 2 * sample_below(&mut next_u128, count);
        if is_probable_prime(candidate, policy.rounds, &mut next_u128) {
            return Ok(candidate);
        }
    }
    Err(PrimeSamplingError::Exhausted {
        min,
        max,
        attempts: policy.max_candidates,
    })
}

/// Uniform in `[0, bound)`: draws masked to the bit length of `bound - 1`,
/// accepted below `bound`. Nothing is drawn when the answer is forced.
fn sample_below(next_u128: &mut impl FnMut() -> u128, bound: u128) -> u128 {
    debug_assert!(bound > 0, "empty sampling range");
    let bits = u128::BITS - (bound - 1).leading_zeros();
    if bits == 0 {
        return 0;
    }
    let mask = u128::MAX >> (u128::BITS - bits);
    loop {
        let candidate = next_u128() & mask;
        if candidate < bound {
            return candidate;
        }
    }
}

/// Trial division, then `rounds` strong tests with bases drawn uniformly from
/// `[2, candidate - 2]`. A composite survives one round with probability at
/// most `1/4`.
fn is_probable_prime(candidate: u128, rounds: u32, next_u128: &mut impl FnMut() -> u128) -> bool {
    if let Some(decided) = small_primes_sieve(candidate) {
        return decided;
    }
    let field = Montgomery::new(candidate);
    // `candidate - 1 = odd * 2^twos`.
    let twos = (candidate - 1).trailing_zeros();
    let odd = (candidate - 1) >> twos;
    (0..rounds).all(|_| {
        let base = 2 + sample_below(next_u128, candidate - 3);
        field.strong_round(base, odd, twos)
    })
}

/// `Some` when the sieve decides `candidate`.
fn small_primes_sieve(candidate: u128) -> Option<bool> {
    if candidate < 2 {
        return Some(false);
    }
    for prime in SMALL_PRIMES_TABLE {
        if candidate == prime {
            return Some(true);
        }
        if candidate.is_multiple_of(prime) {
            return Some(false);
        }
    }
    None
}

/// Montgomery arithmetic modulo an odd `q` with `R = 2^128`, the two-limb
/// schedule of `f2z-pcs`'s `field::prime::montgomery128`. Values are residues
/// times `R`, held below `q`.
#[derive(Clone, Debug)]
struct Montgomery {
    modulus: u128,
    /// `-q^-1 mod 2^64`.
    neg_inv: u64,
    /// `R mod q`, the form of one.
    one: u128,
    /// `R^2 mod q`, which takes an integer into the form.
    r2: u128,
}

impl Montgomery {
    fn new(modulus: u128) -> Self {
        debug_assert!(
            modulus & 1 == 1 && modulus >= 3,
            "Montgomery modulus must be odd"
        );
        // Newton iteration doubles the correct low bits: 1, 2, ..., 64.
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub((modulus as u64).wrapping_mul(inv)));
        }
        let one = (u128::MAX % modulus + 1) % modulus;
        let mut r2 = one;
        for _ in 0..u128::BITS {
            let (doubled, carry) = r2.overflowing_add(r2);
            r2 = if carry || doubled >= modulus {
                doubled.wrapping_sub(modulus)
            } else {
                doubled
            };
        }
        Self {
            modulus,
            neg_inv: inv.wrapping_neg(),
            one,
            r2,
        }
    }

    /// `a * R mod q` for `a < q`.
    fn to_form(&self, a: u128) -> u128 {
        self.mul(a, self.r2)
    }

    /// `a * b / R mod q` for `a, b < q`: schoolbook product, then REDC one
    /// 64-bit limb at a time, then one conditional subtraction.
    fn mul(&self, a: u128, b: u128) -> u128 {
        let lo = |x: u128| u128::from(x as u64);
        let q = self.modulus;
        let (q0, q1) = (lo(q), q >> 64);
        let (a0, a1) = (lo(a), a >> 64);
        let (b0, b1) = (lo(b), b >> 64);

        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> 64) + lo(p01) + lo(p10);
        let mid2 = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + lo(p11);
        let (t0, t1, t2, t3) = (lo(p00), lo(mid), lo(mid2), (mid2 >> 64) + (p11 >> 64));

        // `m0 * q` clears limb 0.
        let m0 = u128::from((t0 as u64).wrapping_mul(self.neg_inv));
        let (mq0, mq1) = (m0 * q0, m0 * q1);
        let c0 = t0 + lo(mq0);
        debug_assert_eq!(lo(c0), 0);
        let c1 = t1 + (mq0 >> 64) + lo(mq1) + (c0 >> 64);
        let c2 = t2 + (mq1 >> 64) + (c1 >> 64);
        let c3 = t3 + (c2 >> 64);
        // `m1 * q` clears limb 1.
        let m1 = u128::from((c1 as u64).wrapping_mul(self.neg_inv));
        let (n0, n1) = (m1 * q0, m1 * q1);
        let d1 = lo(c1) + lo(n0);
        debug_assert_eq!(lo(d1), 0);
        let d2 = lo(c2) + (n0 >> 64) + lo(n1) + (d1 >> 64);
        let d3 = lo(c3) + (n1 >> 64) + (d2 >> 64);

        // The result is below `2q`; the two carries cannot both be set.
        let overflow = (c3 >> 64) | (d3 >> 64) != 0;
        let result = lo(d2) | (lo(d3) << 64);
        if overflow || result >= q {
            result.wrapping_sub(q)
        } else {
            result
        }
    }

    /// `base^exponent` in the form, by left-to-right square-and-multiply.
    fn pow(&self, base: u128, exponent: u128) -> u128 {
        let mut result = self.one;
        for i in (0..u128::BITS - exponent.leading_zeros()).rev() {
            result = self.mul(result, result);
            if (exponent >> i) & 1 == 1 {
                result = self.mul(result, base);
            }
        }
        result
    }

    /// One Miller–Rabin round for `q - 1 = odd * 2^twos`: `base^odd` is `±1`,
    /// or one of its next `twos - 1` squarings is `-1`.
    fn strong_round(&self, base: u128, odd: u128, twos: u32) -> bool {
        let minus_one = self.modulus - self.one;
        let mut value = self.pow(self.to_form(base), odd);
        if value == self.one || value == minus_one {
            return true;
        }
        (1..twos).any(|_| {
            value = self.mul(value, value);
            value == minus_one
        })
    }
}

#[cfg(test)]
#[allow(clippy::reversed_empty_ranges)]
mod tests {
    use field::Q100;

    use super::*;

    /// SplitMix64, two outputs per draw.
    fn stream(mut state: u64) -> impl FnMut() -> u128 {
        let mut next_u64 = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        move || u128::from(next_u64()) | (u128::from(next_u64()) << 64)
    }

    /// `a * b mod q` by doubling, for any `q >= 2`.
    fn mul_mod_reference(a: u128, b: u128, q: u128) -> u128 {
        let add_mod = |a: u128, b: u128| {
            let (sum, carry) = a.overflowing_add(b);
            if carry || sum >= q {
                sum.wrapping_sub(q)
            } else {
                sum
            }
        };
        let mut result = 0;
        let mut addend = a % q;
        for i in 0..u128::BITS - b.leading_zeros() {
            if (b >> i) & 1 == 1 {
                result = add_mod(result, addend);
            }
            addend = add_mod(addend, addend);
        }
        result
    }

    /// Exact primality below `2^40`.
    fn is_prime_by_trial_division(n: u128) -> bool {
        assert!(n < 1 << 40);
        if n < 2 || (n > 2 && n & 1 == 0) {
            return false;
        }
        let mut divisor = 3;
        while divisor * divisor <= n {
            if n.is_multiple_of(divisor) {
                return false;
            }
            divisor += 2;
        }
        true
    }

    const PRIMES: [u128; 11] = [
        2,
        3,
        53,
        59,
        61,
        (1 << 61) - 1,
        Q100,
        (1 << 89) - 1,
        (1 << 107) - 1,
        (1 << 127) - 1,
        u128::MAX - 158,
    ];
    // Carmichael numbers, strong pseudoprimes to base 2, and two products
    // of large primes, one of them full-width.
    const COMPOSITES: [u128; 12] = [
        0,
        1,
        561,
        1105,
        1729,
        2047,
        3277,
        4033,
        (1 << 100) - 1,
        (1 << 127) + 1,
        ((1 << 64) - 59) * ((1 << 64) - 83),
        u128::MAX,
    ];

    #[test]
    fn montgomery_multiplication_matches_the_reference() {
        let mut draw = stream(1);
        for modulus in [
            59,
            (1 << 61) - 1,
            Q100,
            (1 << 127) - 1,
            u128::MAX - 158,
            u128::MAX,
        ] {
            let field = Montgomery::new(modulus);
            assert_eq!(field.to_form(1), field.one);
            assert_eq!(field.mul(field.one, 1), 1);
            for _ in 0..64 {
                let (a, b) = (draw() % modulus, draw() % modulus);
                let product = field.mul(field.mul(field.to_form(a), field.to_form(b)), 1);
                assert_eq!(
                    product,
                    mul_mod_reference(a, b, modulus),
                    "{a} * {b} mod {modulus}"
                );
            }
            for a in [0, 1, modulus - 1] {
                let product = field.mul(field.mul(field.to_form(a), field.to_form(a)), 1);
                assert_eq!(
                    product,
                    mul_mod_reference(a, a, modulus),
                    "{a}^2 mod {modulus}"
                );
            }
        }
    }

    #[test]
    fn probable_prime_decides_known_primes_and_composites() {
        let mut draw = stream(2);
        for prime in PRIMES {
            assert!(is_probable_prime(prime, 16, &mut draw), "{prime} is prime");
        }
        for composite in COMPOSITES {
            assert!(
                !is_probable_prime(composite, 16, &mut draw),
                "{composite} is composite"
            );
        }
    }

    #[test]
    fn sampling_below_masks_then_rejects() {
        let mut draws = [7u128, 5, 8 | 3].into_iter();
        assert_eq!(sample_below(&mut || draws.next().unwrap(), 5), 3);
        assert!(draws.next().is_none());

        let mut draws = 0;
        assert_eq!(
            sample_below(
                &mut || {
                    draws += 1;
                    u128::MAX
                },
                8
            ),
            7
        );
        assert_eq!(draws, 1);

        assert_eq!(sample_below(&mut || panic!("forced answer draws"), 1), 0);
    }

    #[test]
    fn sampled_primes_are_prime_and_in_the_interval() {
        let range = interval(40);
        for seed in 0..32 {
            let prime = sample(stream(seed), &range).unwrap();
            assert!(range.contains(&prime), "seed {seed}: {prime} out of range");
            assert!(
                is_prime_by_trial_division(prime),
                "seed {seed}: {prime} is composite"
            );
        }
        for width in [2, 64, 100, 113, 126, 128] {
            let range = interval(width);
            let prime = sample(stream(u64::from(width)), &range).unwrap();
            assert!(range.contains(&prime), "{width} bits: {prime} out of range");
            assert!(is_probable_prime(prime, 8, &mut stream(99)));
        }
    }

    #[test]
    fn interval_edge_cases() {
        let draw = || stream(3);
        assert_eq!(
            sample(draw(), &(5..=3)),
            Err(PrimeSamplingError::InvalidInterval { min: 5, max: 3 })
        );
        assert_eq!(
            sample(draw(), &(100..=100)),
            Err(PrimeSamplingError::NoOddCandidate { min: 100, max: 100 })
        );
        assert_eq!(
            sample(draw(), &(1..=2)),
            Err(PrimeSamplingError::NoOddCandidate { min: 1, max: 2 })
        );
        assert_eq!(sample(draw(), &(101..=101)), Ok(101));
        assert_eq!(sample(draw(), &(100..=102)), Ok(101));
        assert!([5, 7].contains(&sample(draw(), &(4..=9)).unwrap()));
        assert_eq!(
            sample(draw(), &(25..=27)),
            Err(PrimeSamplingError::Exhausted {
                min: 25,
                max: 27,
                attempts: 320
            })
        );
    }

    #[test]
    fn search_policy_matches_the_poc_budget() {
        let interval = interval(100);
        assert_eq!(*interval.start(), 1 << 99);
        assert_eq!(*interval.end(), (1 << 100) - 1);
        assert_eq!(
            SearchPolicy::for_interval(*interval.start(), *interval.end()),
            SearchPolicy {
                max_candidates: 6400,
                rounds: 79
            }
        );
        assert_eq!(
            SearchPolicy::for_interval(1 << 63, u64::MAX.into()),
            SearchPolicy {
                max_candidates: 4096,
                rounds: 78
            }
        );
        assert_eq!(
            SearchPolicy::for_interval(101, 101),
            SearchPolicy {
                max_candidates: 1,
                rounds: 72
            }
        );
    }
}
