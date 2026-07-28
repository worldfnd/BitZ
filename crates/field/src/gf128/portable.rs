//! Scalar carryless-multiply pipeline for `GF(2^128)`.
//!
//! Two jobs: the production path on targets without a carryless-multiply
//! instruction, and the oracle the SIMD kernels are tested against. Written for
//! obviousness, not speed.

use super::REDUCTION;

/// Carryless product of two 64-bit polynomials, as `(low, high)` halves of the
/// 128-bit result.
pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
    let mut lo = 0u64;
    let mut hi = 0u64;
    for i in 0..64 {
        if (b >> i) & 1 == 1 {
            lo ^= a << i;
            // `a >> 64` is undefined; at `i == 0` nothing crosses the boundary.
            if i > 0 {
                hi ^= a >> (64 - i);
            }
        }
    }
    (lo, hi)
}

/// Schoolbook 128x128 carryless product, as four words in ascending
/// significance.
pub fn clmul128(a: [u64; 2], b: [u64; 2]) -> [u64; 4] {
    let (l0, l1) = clmul64(a[0], b[0]);
    let (h0, h1) = clmul64(a[1], b[1]);
    let (m0, m1) = clmul64(a[0], b[1]);
    let (n0, n1) = clmul64(a[1], b[0]);
    [l0, l1 ^ m0 ^ n0, h0 ^ m1 ^ n1, h1]
}

/// `w * g` where `g = X^7 + X^2 + X + 1`, as `(low, high)` 64-bit halves.
fn times_g(w: u64) -> (u64, u64) {
    (
        w ^ (w << 1) ^ (w << 2) ^ (w << 7),
        (w >> 63) ^ (w >> 62) ^ (w >> 57),
    )
}

/// Reduce a 256-bit carryless product modulo `X^128 + X^7 + X^2 + X + 1`.
///
/// Writing the input as `r0 + r1*X^64 + r2*X^128 + r3*X^192` and substituting
/// `X^128 = g`, the two high words fold down as `r2*g + (r3*g)*X^64`. That fold
/// spills at most seven bits back above `X^128`, which one more substitution
/// clears.
pub fn reduce(r: [u64; 4]) -> [u64; 2] {
    let (a0, a1) = times_g(r[2]);
    let (b0, b1) = times_g(r[3]);
    let w0 = r[0] ^ a0;
    let w1 = r[1] ^ a1 ^ b0;
    // `b1` holds at most 7 bits, so `b1*g` has at most 14 and stays in word 0.
    [w0 ^ b1 ^ (b1 << 1) ^ (b1 << 2) ^ (b1 << 7), w1]
}

pub fn mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    reduce(clmul128(a, b))
}

/// In characteristic 2 the cross terms of `(a0 + a1*X^64)^2` cancel, leaving
/// `a0^2 + a1^2*X^128`.
pub fn square(a: [u64; 2]) -> [u64; 2] {
    let (l0, l1) = clmul64(a[0], a[0]);
    let (h0, h1) = clmul64(a[1], a[1]);
    reduce([l0, l1, h0, h1])
}

/// Multiply by the generator `X`: a one-bit shift of the 128-bit value, folding
/// `X^128 = g` back in when the `X^127` coefficient overflows.
pub const fn mul_x(a: [u64; 2]) -> [u64; 2] {
    let carry = (a[1] >> 63).wrapping_neg();
    [
        (a[0] << 1) ^ (carry & REDUCTION),
        (a[1] << 1) | (a[0] >> 63),
    ]
}

/// This module is the oracle the SIMD kernels are checked against, so agreeing
/// with them proves nothing. Each primitive is pinned instead to an independent
/// derivation — `clmul64` to the definition of a polynomial product, `reduce`
/// to explicit long division — each written in a different shape from the code
/// under test, so a shared bug would have to survive two unrelated
/// formulations.
#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::{RngCore, SeedableRng};
    use rand_pcg::Pcg64;

    fn bit(words: &[u64], i: usize) -> bool {
        (words[i / 64] >> (i % 64)) & 1 == 1
    }

    fn flip(words: &mut [u64], i: usize) {
        words[i / 64] ^= 1 << (i % 64);
    }

    /// `(a*b)_k` is the parity of `{(i, j) : i + j = k, a_i = b_j = 1}` — the
    /// definition of a polynomial product, evaluated term by term.
    fn clmul64_by_definition(a: u64, b: u64) -> (u64, u64) {
        let mut out = [0u64; 2];
        for i in 0..64 {
            for j in 0..64 {
                if (a >> i) & 1 == 1 && (b >> j) & 1 == 1 {
                    flip(&mut out, i + j);
                }
            }
        }
        (out[0], out[1])
    }

    /// Long division by `f = X^128 + X^7 + X^2 + X + 1`: clear the highest set
    /// bit at or above 128 by subtracting a shifted copy of `f`, and repeat.
    /// Structurally unlike `reduce`, which folds twice in closed form.
    fn reduce_by_long_division(r: [u64; 4]) -> [u64; 2] {
        let mut r = r;
        for i in (128..256).rev() {
            if bit(&r, i) {
                flip(&mut r, i);
                for term in [7, 2, 1, 0] {
                    flip(&mut r, i - 128 + term);
                }
            }
        }
        [r[0], r[1]]
    }

    #[test]
    fn clmul64_matches_the_coefficient_definition() {
        let mut rng = Pcg64::seed_from_u64(101);
        let edges = [0, 1, 2, u64::MAX, 1 << 63, REDUCTION];
        for &a in &edges {
            for &b in &edges {
                assert_eq!(
                    clmul64(a, b),
                    clmul64_by_definition(a, b),
                    "{a:#x} * {b:#x}"
                );
            }
        }
        for _ in 0..512 {
            let (a, b) = (rng.next_u64(), rng.next_u64());
            assert_eq!(
                clmul64(a, b),
                clmul64_by_definition(a, b),
                "{a:#x} * {b:#x}"
            );
        }
    }

    #[test]
    fn reduce_matches_long_division() {
        let mut rng = Pcg64::seed_from_u64(102);
        // Every single-bit input in turn, so each of the 128 foldable positions
        // is exercised alone rather than only in random combination.
        for i in 0..256 {
            let mut r = [0u64; 4];
            flip(&mut r, i);
            assert_eq!(reduce(r), reduce_by_long_division(r), "bit {i}");
        }
        for _ in 0..512 {
            let r = [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ];
            assert_eq!(reduce(r), reduce_by_long_division(r), "{r:?}");
        }
    }

    /// The two-product shortcut is valid only because the cross terms cancel in
    /// characteristic 2; checked against the four-product path directly, not
    /// through whichever kernel is active.
    #[test]
    fn square_matches_the_general_multiply() {
        let mut rng = Pcg64::seed_from_u64(103);
        for _ in 0..512 {
            let a = [rng.next_u64(), rng.next_u64()];
            assert_eq!(square(a), mul(a, a), "{a:?}");
        }
    }

    /// Products that land exactly on the `X^128` boundary, where a wrong
    /// reduction polynomial or a swapped word order shows up immediately.
    #[test]
    fn boundary_products() {
        let x = [2, 0];
        assert_eq!(mul(x, [1 << 63, 0]), [0, 1]); // X * X^63  = X^64
        assert_eq!(mul(x, [0, 1 << 63]), [REDUCTION, 0]); // X * X^127 = g
        assert_eq!(mul([0, 1], [0, 1]), [REDUCTION, 0]); // X^64 * X^64 = g
        assert_eq!(mul_x([0, 1 << 63]), [REDUCTION, 0]);
    }
}
