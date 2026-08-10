//! Differential test against flock's `GF(2^128)`.
//!
//! flock implements the same field independently, and a later stage converts
//! between the two types by copying fields rather than re-encoding. Two things
//! have to hold for that to be sound: the arithmetic has to agree, and the
//! layouts have to match. Both are checked here, so a change on either side
//! fails at this crate rather than at the conversion.

use field::F128;
use flock_core::field::gf2_128::F128 as Flock;
use num_traits::{ConstOne, ConstZero, Inv, Zero};
use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64;

fn to_flock(a: F128) -> Flock {
    Flock::new(a.lo, a.hi)
}

fn from_flock(a: Flock) -> F128 {
    F128::new(a.lo, a.hi)
}

fn f128(rng: &mut Pcg64) -> F128 {
    F128::new(rng.next_u64(), rng.next_u64())
}

/// The boundary values, where a disagreement about the reduction polynomial or
/// the word order shows up without needing luck.
const EDGES: [(u64, u64); 8] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0x87, 0),
    (1 << 63, 0),
    (0, 1),
    (0, 1 << 63),
    (u64::MAX, u64::MAX),
];

fn cases() -> Vec<(F128, F128)> {
    let mut rng = Pcg64::seed_from_u64(601);
    let edges: Vec<F128> = EDGES.iter().map(|&(lo, hi)| F128::new(lo, hi)).collect();

    let mut out = Vec::new();
    for &a in &edges {
        for &b in &edges {
            out.push((a, b));
        }
        for _ in 0..64 {
            out.push((a, f128(&mut rng)));
            out.push((f128(&mut rng), a));
        }
    }
    for _ in 0..2048 {
        out.push((f128(&mut rng), f128(&mut rng)));
    }
    out
}

/// A field copy between the two types is only sound if these agree. `repr(C)`
/// fixes the field order, leaving size, alignment, and a round-trip by name.
#[test]
fn layouts_are_interchangeable() {
    assert_eq!(size_of::<F128>(), size_of::<Flock>());
    assert_eq!(align_of::<F128>(), align_of::<Flock>());
    assert_eq!(size_of::<F128>(), 16);
    assert_eq!(align_of::<F128>(), 16);

    let mut rng = Pcg64::seed_from_u64(602);
    for _ in 0..256 {
        let a = f128(&mut rng);
        assert_eq!(from_flock(to_flock(a)), a);
    }

    // The two crates agree on which element the constants name.
    assert_eq!(to_flock(F128::ZERO), Flock::ZERO);
    assert_eq!(to_flock(F128::ONE), Flock::ONE);
    assert_eq!(to_flock(F128::GENERATOR), Flock::generator());
}

/// flock's portable path is always compiled, so this runs on every target.
#[test]
fn multiply_matches_flock_software() {
    for (a, b) in cases() {
        let expected = from_flock(flock_core::field::gf2_128::software::ghash_mul(
            to_flock(a),
            to_flock(b),
        ));
        assert_eq!(a * b, expected, "{a:?} * {b:?}");
        assert_eq!(
            a.square(),
            from_flock(flock_core::field::gf2_128::software::ghash_mul(
                to_flock(a),
                to_flock(a)
            )),
            "square {a:?}"
        );
    }
}

/// flock's four PMULL variants are separate implementations of the same
/// product, so each is its own check rather than four ways of running one.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[test]
fn multiply_matches_every_flock_neon_variant() {
    use flock_core::field::gf2_128::aarch64;

    for (a, b) in cases() {
        let (x, y) = (to_flock(a), to_flock(b));
        // SAFETY: the module is compiled only with `aes` enabled, which is the
        // same gate these functions require.
        let variants = unsafe {
            [
                ("schoolbook", aarch64::ghash_mul_schoolbook(x, y)),
                ("karatsuba", aarch64::ghash_mul_karatsuba(x, y)),
                (
                    "karatsuba_barrett",
                    aarch64::ghash_mul_karatsuba_barrett(x, y),
                ),
                ("binius", aarch64::ghash_mul_binius(x, y)),
            ]
        };
        for (name, got) in variants {
            assert_eq!(a * b, from_flock(got), "flock {name}: {a:?} * {b:?}");
        }
    }
}

/// flock inverts by Fermat and this crate by Itoh–Tsujii — different addition
/// chains for the same exponent.
#[test]
fn inverse_matches_flock() {
    let mut rng = Pcg64::seed_from_u64(603);
    for _ in 0..256 {
        let a = f128(&mut rng);
        if a.is_zero() {
            continue;
        }
        assert_eq!(a.inv(), Some(from_flock(to_flock(a).inv())), "{a:?}");
    }
}
