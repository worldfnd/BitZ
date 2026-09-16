//! Exponentiation, inversion, and the primitive-element test.

use std::fmt::{Debug, Formatter, Result as FmtResult};

use super::{F128, kernel};
use num_traits::{ConstOne, Inv, Pow, Zero};

/// The order of the multiplicative group, `2^128 - 1`.
///
/// Exactly `u128::MAX`, so for a generator `alpha` the map `n -> alpha^n` is
/// injective on `[0, 2^128 - 1)`: every `u128` but `u128::MAX` names a distinct
/// element.
pub const MULT_ORDER: u128 = u128::MAX;

/// The nine distinct primes dividing [`MULT_ORDER`], which is squarefree.
///
/// `2^128 - 1 = prod_{k=0}^{6} (2^(2^k) + 1)`: the five Fermat primes
/// `3, 5, 17, 257, 65537`, then `2^32 + 1 = 641 * 6700417` and
/// `2^64 + 1 = 274177 * 67280421310721`.
///
/// [`is_generator`] tests one condition per entry, so a composite or missing
/// entry would weaken it silently. The tests re-derive the product and the
/// primality of every entry rather than trusting this list.
pub const ORDER_PRIME_FACTORS: [u128; 9] =
    [3, 5, 17, 257, 641, 65537, 274177, 6700417, 67280421310721];

impl F128 {
    /// `self^(2^k)`. On NEON the value stays in one vector register, so `k`
    /// squarings cost `k` PMULL pairs and one load/store — why
    /// [`Inv::inv`] works in runs.
    pub fn square_n(self, k: u32) -> Self {
        kernel::square_n(self.words(), k).into()
    }
}

impl Pow<u128> for F128 {
    type Output = Self;

    /// Square-and-multiply, low bit first. `self^0` is `ONE`, including for
    /// `ZERO`.
    fn pow(self, exp: u128) -> Self {
        let mut acc = Self::ONE;
        let mut base = self;
        let mut e = exp;
        while e != 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }
}

impl Pow<&u128> for F128 {
    type Output = Self;
    fn pow(self, rhs: &u128) -> Self {
        self.pow(*rhs)
    }
}

impl Pow<u32> for F128 {
    type Output = Self;
    fn pow(self, rhs: u32) -> Self {
        self.pow(u128::from(rhs))
    }
}

impl Inv for F128 {
    type Output = Option<Self>;

    /// The multiplicative inverse, or `None` for zero.
    ///
    /// Itoh–Tsujii, Information and Computation 78(3):171-177, 1988: with
    /// `b_k = self^(2^k - 1)`, the identity `b_(m+n) = (b_m)^(2^n) * b_n`
    /// turns each step of the addition chain
    /// `1, 2, 3, 6, 12, 24, 48, 96, 120, 126, 127` into one squaring run and
    /// one multiply. `b_127` squared is `self^(2^128 - 2)`, the inverse.
    /// 127 squarings and 10 multiplies, against 127 and 127 for the same
    /// exponent by square-and-multiply.
    ///
    /// The paper counts only the multiplies, since a normal basis squares by
    /// cyclic shift. This basis is polynomial, so the squarings are real work
    /// and [`F128::square_n`] is what keeps them cheap.
    fn inv(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let b1 = self;
        let b2 = b1.square_n(1) * b1;
        let b3 = b2.square_n(1) * b1;
        let b6 = b3.square_n(3) * b3;
        let b12 = b6.square_n(6) * b6;
        let b24 = b12.square_n(12) * b12;
        let b48 = b24.square_n(24) * b24;
        let b96 = b48.square_n(48) * b48;
        let b120 = b96.square_n(24) * b24;
        let b126 = b120.square_n(6) * b6;
        let b127 = b126.square_n(1) * b1;
        Some(b127.square())
    }
}

/// Returns `true` if `a` generates the whole multiplicative group.
///
/// The primitive-element test: `a^((2^128 - 1)/p) != 1` for every prime `p`
/// dividing the order. Zero needs its own guard rather than a fast path —
/// `0^e` is `0`, never `1`, so every factor would pass.
///
/// `prod (1 - 1/p)` is 49.9%, so a random `a` passes after two tries on
/// average.
pub fn is_generator(a: F128) -> bool {
    !a.is_zero()
        && ORDER_PRIME_FACTORS
            .iter()
            .all(|&p| a.pow(MULT_ORDER / p) != F128::ONE)
}

/// The smallest `F128::from(n)`, `n >= 2`, that generates the group — a
/// deterministic choice for tests and benches. The protocol accepts any `a`
/// that [`is_generator`] does.
pub fn smallest_generator() -> F128 {
    (2u128..)
        .map(F128::from)
        .find(|&a| is_generator(a))
        .expect("a finite field's multiplicative group is cyclic")
}

/// Comb table for raising one fixed base to many different exponents.
///
/// [`F128::pow`] re-runs the `alpha, alpha^2, alpha^4, ...` squaring chain on
/// every call, though for a fixed base it never changes. Precomputing
/// `table[i][d] = alpha^(d * 2^(win*i))` turns each exponentiation into one
/// multiply per non-zero window, with no squarings.
pub struct FixedBasePow {
    table: Vec<Vec<F128>>,
    win: u32,
}

/// The base and the window, not the table: that is `ceil(128 / win) * 2^win`
/// elements, 4096 of them at the window the tests use.
impl Debug for FixedBasePow {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("FixedBasePow")
            .field("base", &self.pow(1))
            .field("win", &self.win)
            .finish_non_exhaustive()
    }
}

impl FixedBasePow {
    /// Covers the full 128-bit exponent range; `win` trades table size
    /// (`ceil(128/win) * 2^win` elements) against multiplies per call
    /// (`ceil(128/win)` at worst).
    pub fn new(alpha: F128, win: u32) -> Self {
        assert!(
            (1..=16).contains(&win),
            "window must be in 1..=16, got {win}"
        );
        let windows = 128u32.div_ceil(win);
        let mut table = Vec::with_capacity(windows as usize);
        let mut base = alpha;
        for _ in 0..windows {
            let mut row = Vec::with_capacity(1 << win);
            let mut cur = F128::ONE;
            for _ in 0..(1u32 << win) {
                row.push(cur);
                cur *= base;
            }
            table.push(row);
            base = base.square_n(win);
        }
        Self { table, win }
    }

    pub fn pow(&self, exp: u128) -> F128 {
        let mask = (1u128 << self.win) - 1;
        let mut acc = F128::ONE;
        let mut e = exp;
        let mut i = 0;
        while e != 0 {
            let d = (e & mask) as usize;
            if d != 0 {
                acc *= self.table[i][d];
            }
            e >>= self.win;
            i += 1;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ConstZero;
    use rand_core::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    fn f128(rng: &mut Pcg64) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    fn is_prime(n: u128) -> bool {
        if n < 2 {
            return false;
        }
        let mut d = 2u128;
        while d * d <= n {
            if n.is_multiple_of(d) {
                return false;
            }
            d += 1;
        }
        true
    }

    /// [`is_generator`] tests one condition per entry, so a composite entry or
    /// a missing one would silently weaken it into accepting elements of a
    /// proper subgroup. Trial division rather than a probabilistic test: the
    /// largest entry is under `2^47`, so this is a proof, not evidence.
    #[test]
    fn order_factorization_is_complete() {
        let mut product = 1u128;
        for &p in &ORDER_PRIME_FACTORS {
            assert!(is_prime(p), "{p} is not prime");
            product = product.checked_mul(p).expect("product overflows u128");
        }
        assert_eq!(product, MULT_ORDER);
        assert_eq!(MULT_ORDER, u128::MAX);
    }

    #[test]
    fn square_n_is_repeated_squaring() {
        let mut rng = Pcg64::seed_from_u64(301);
        for _ in 0..64 {
            let a = f128(&mut rng);
            let mut expected = a;
            for k in 0..=130u32 {
                assert_eq!(a.square_n(k), expected, "square_n({k}) of {a:?}");
                expected = expected.square();
            }
        }
    }

    #[test]
    fn pow_matches_repeated_multiplication() {
        let mut rng = Pcg64::seed_from_u64(302);
        for _ in 0..64 {
            let a = f128(&mut rng);
            let mut expected = F128::ONE;
            for e in 0..64u128 {
                assert_eq!(a.pow(e), expected, "{a:?}^{e}");
                expected *= a;
            }
            // A pure power of two exercises the squaring chain with a single
            // set bit, where an off-by-one in the ladder still shows up.
            assert_eq!(a.pow(1_u128 << 100), a.square_n(100));
        }
    }

    /// Lagrange: every non-zero element satisfies `a^(2^128 - 1) = 1`, and
    /// squaring 128 times is the identity (Frobenius over the prime field).
    #[test]
    fn group_order_and_frobenius() {
        let mut rng = Pcg64::seed_from_u64(303);
        for _ in 0..64 {
            let a = f128(&mut rng);
            assert_eq!(a.square_n(128), a);
            if !a.is_zero() {
                assert_eq!(a.pow(MULT_ORDER), F128::ONE);
            }
        }
    }

    /// Itoh–Tsujii against the definition it is an optimisation of.
    #[test]
    fn inverse_matches_fermat() {
        let mut rng = Pcg64::seed_from_u64(304);
        assert_eq!(F128::ZERO.inv(), None);
        assert_eq!(F128::ONE.inv(), Some(F128::ONE));
        for _ in 0..256 {
            let a = f128(&mut rng);
            if a.is_zero() {
                continue;
            }
            let inv = a.inv().expect("non-zero");
            assert_eq!(inv, a.pow(MULT_ORDER - 1), "{a:?}");
            assert_eq!(a * inv, F128::ONE, "{a:?}");
        }
    }

    /// `X` is a generator, and the smallest one — see [`F128::GENERATOR`] for
    /// why that is not automatic.
    #[test]
    fn generator_is_x() {
        assert!(is_generator(F128::GENERATOR));
        assert_eq!(smallest_generator(), F128::GENERATOR);
    }

    /// The test must reject as well as accept, or it would be vacuous. `g^3`
    /// has order `(2^128-1)/3`; `g^2` is still a generator because the order is
    /// odd, so squaring permutes the group.
    #[test]
    fn is_generator_rejects_proper_subgroups() {
        let g = F128::GENERATOR;
        assert!(!is_generator(F128::ZERO));
        assert!(!is_generator(F128::ONE));
        for &p in &ORDER_PRIME_FACTORS {
            let a = g.pow(p);
            assert!(!is_generator(a), "g^{p} should have order (2^128-1)/{p}");
        }
        assert!(is_generator(g.pow(2_u32)));
    }

    #[test]
    fn comb_matches_the_general_pow() {
        let mut rng = Pcg64::seed_from_u64(305);
        let alpha = smallest_generator();
        for win in [1u32, 4, 8] {
            let comb = FixedBasePow::new(alpha, win);
            assert_eq!(comb.pow(0), F128::ONE);
            for _ in 0..64 {
                let e = (rng.next_u64() as u128) << 64 | rng.next_u64() as u128;
                assert_eq!(comb.pow(e), alpha.pow(e), "win {win}, exp {e:#x}");
            }
            assert_eq!(comb.pow(u128::MAX), alpha.pow(u128::MAX));
        }
    }
}
