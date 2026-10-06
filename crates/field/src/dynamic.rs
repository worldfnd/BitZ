//! Field with modulus not known at compile-time.
//!
//! We only expect to have one of those at any given time, so the modulus is shared globally.

use crate::{FieldWithDynamicModulus, helpers};
use crypto_primitives::{BaseField, LiftElement, WithAssociatedInteger};
use crypto_primitives_proc_macros::InfallibleCheckedOp;
use num_traits::{
    Bounded, CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, ConstOne, ConstZero, Inv,
    One, Pow, Zero,
};
use pastey::paste;
use std::fmt::Display;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The installed [`FieldMetadata`], one atomic per word: the config shared by
/// every [`DynField`].
///
/// Written only by [`DynField::set_modulus`], read by every operation. Its
/// contract rules out a writer concurrent with a reader, so the words need
/// neither a lock nor an ordering among them: `Relaxed` loads are plain
/// loads, and every thread reads the same cache line without writing it.
mod global {
    use crate::helpers::FieldMetadata;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

    static MODULUS_LO: AtomicU64 = AtomicU64::new(0);
    static MODULUS_HI: AtomicU64 = AtomicU64::new(0);
    static MU_LO: AtomicU64 = AtomicU64::new(0);
    static MU_HI: AtomicU64 = AtomicU64::new(0);
    static BITS: AtomicU32 = AtomicU32::new(0);

    #[inline(always)]
    fn load_u128(lo: &AtomicU64, hi: &AtomicU64) -> u128 {
        u128::from(lo.load(Relaxed)) | (u128::from(hi.load(Relaxed)) << 64)
    }

    fn store_u128(lo: &AtomicU64, hi: &AtomicU64, value: u128) {
        lo.store(value as u64, Relaxed);
        hi.store((value >> 64) as u64, Relaxed);
    }

    #[inline(always)]
    pub(super) fn modulus() -> u128 {
        load_u128(&MODULUS_LO, &MODULUS_HI)
    }

    #[inline(always)]
    pub(super) fn load() -> FieldMetadata {
        FieldMetadata {
            modulus: modulus(),
            bits: BITS.load(Relaxed),
            mu: load_u128(&MU_LO, &MU_HI),
        }
    }

    pub(super) fn store(cfg: &FieldMetadata) {
        store_u128(&MODULUS_LO, &MODULUS_HI, cfg.modulus);
        store_u128(&MU_LO, &MU_HI, cfg.mu);
        BITS.store(cfg.bits, Relaxed);
    }
}

/// An element of `Z/qZ` for the installed `q`, held reduced.
///
/// The modulus is process-wide, so values made under different moduli are
/// the same Rust type; keeping them apart is the caller's job.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct DynField {
    reduced_value: u128,
}

impl DynField {
    /// The installed modulus and its reduction constants.
    #[inline(always)]
    pub fn config() -> helpers::FieldMetadata {
        let cfg = global::load();
        debug_assert!(cfg.modulus != 0, "Field modulus has not been set yet!");
        cfg
    }
}

impl FieldWithDynamicModulus for DynField {
    /// Set modulus globally.
    ///
    /// The modulus must be an odd prime below `2^126`, the bound of the
    /// Barrett reduction shared with [`Fq`](crate::Fq); anything else panics.
    unsafe fn set_modulus(modulus: u128) {
        assert!(modulus >= 3, "modulus must be at least 3");
        assert_eq!(modulus % 2, 1, "modulus must be odd");
        assert!(
            modulus < 1u128 << helpers::MAX_MODULUS_BITS,
            "modulus must be below 2^126"
        );
        // `DynField` claims to be a prime field, so it enforces that itself
        // rather than trusting whoever drew the modulus.
        assert!(helpers::is_prime(modulus), "modulus must be prime");
        global::store(&helpers::FieldMetadata::new(modulus));
    }
}

//
// Core traits
//

impl Display for DynField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (mod {})", self.reduced_value, DynField::modulus())
    }
}

//
// Zero and One traits
//

impl Zero for DynField {
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.reduced_value == 0
    }
}

impl One for DynField {
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl ConstZero for DynField {
    const ZERO: Self = DynField { reduced_value: 0 };
}

impl ConstOne for DynField {
    const ONE: Self = DynField { reduced_value: 1 };
}

//
// Basic arithmetic operations
//

impl Neg for DynField {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else {
            DynField {
                reduced_value: Self::modulus() - self.reduced_value,
            }
        }
    }
}

macro_rules! impl_basic_op_forward_to_assign {
    ($trait:ident, $method:ident, $assign_method:ident) => {
        paste! {
            impl $trait for DynField {
                type Output = DynField;

                #[inline(always)]
                fn $method(mut self, rhs: DynField) -> Self::Output {
                    [<$trait Assign>]::$assign_method(&mut self, rhs);
                    self
                }
            }

            impl $trait<&Self> for DynField {
                type Output = DynField;

                #[inline(always)]
                fn $method(mut self, rhs: &DynField) -> Self::Output {
                    [<$trait Assign>]::$assign_method(&mut self, rhs);
                    self
                }
            }

            impl $trait<DynField> for &DynField {
                type Output = DynField;

                #[inline(always)]
                fn $method(self, rhs: DynField) -> Self::Output {
                    (*self).$method(rhs)
                }
            }

            impl $trait for &DynField {
                type Output = DynField;

                #[inline(always)]
                fn $method(self, rhs: &DynField) -> Self::Output {
                    (*self).$method(*rhs)
                }
            }
        }
    };
}

impl_basic_op_forward_to_assign!(Add, add, add_assign);
impl_basic_op_forward_to_assign!(Sub, sub, sub_assign);
impl_basic_op_forward_to_assign!(Mul, mul, mul_assign);
impl_basic_op_forward_to_assign!(Div, div, div_assign);

// Required by `crypto_primitives::Field`, not implemented yet.

impl Pow<u32> for DynField {
    type Output = Self;

    fn pow(self, _exp: u32) -> Self::Output {
        unimplemented!("exponentiation is not implemented yet")
    }
}

impl Pow<u128> for DynField {
    type Output = Self;

    fn pow(self, _exp: u128) -> Self::Output {
        unimplemented!("exponentiation is not implemented yet")
    }
}

impl Pow<&u128> for DynField {
    type Output = Self;

    fn pow(self, _exp: &u128) -> Self::Output {
        unimplemented!("exponentiation is not implemented yet")
    }
}

impl Inv for DynField {
    type Output = Option<Self>;

    fn inv(self) -> Self::Output {
        unimplemented!("inversion is not implemented yet")
    }
}

//
// Checked arithmetic operations
// (Note: Field operations do not overflow)
//

impl CheckedDiv for DynField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn checked_div(&self, rhs: &Self) -> Option<Self> {
        Some(self * Inv::inv(*rhs)?)
    }
}

//
// Arithmetic assign operations
//

macro_rules! impl_op_assign_boilerplate {
    ($trait:ident, $method:ident) => {
        impl<'a> $trait<&'a DynField> for DynField {
            #[inline(always)]
            fn $method(&mut self, rhs: &'a DynField) {
                self.$method(*rhs);
            }
        }
    };
}

impl_op_assign_boilerplate!(AddAssign, add_assign);
impl_op_assign_boilerplate!(SubAssign, sub_assign);
impl_op_assign_boilerplate!(MulAssign, mul_assign);
impl_op_assign_boilerplate!(DivAssign, div_assign);

impl AddAssign for DynField {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        let modulus = Self::modulus();
        // SAFETY: Both operands are below `modulus < 2^126`, so the sum cannot wrap.
        let sum = unsafe { self.reduced_value.unchecked_add(rhs.reduced_value) };
        self.reduced_value = if sum >= modulus { sum - modulus } else { sum };
    }
}

impl SubAssign for DynField {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.reduced_value = if self.reduced_value >= rhs.reduced_value {
            self.reduced_value - rhs.reduced_value
        } else {
            self.reduced_value + Self::modulus() - rhs.reduced_value
        };
    }
}

impl MulAssign for DynField {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.reduced_value = Self::config().mul(self.reduced_value, rhs.reduced_value);
    }
}

impl DivAssign for DynField {
    fn div_assign(&mut self, _rhs: Self) {
        unimplemented!("division is not implemented yet")
    }
}

//
// Aggregate operations
//

impl Sum for DynField {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Self> for DynField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product for DynField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a> Product<&'a Self> for DynField {
    #[allow(clippy::arithmetic_side_effects)] // False alert
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

//
// Conversions
//

impl From<bool> for DynField {
    fn from(value: bool) -> Self {
        if value { Self::ONE } else { Self::ZERO }
    }
}

impl From<&u64> for DynField {
    fn from(value: &u64) -> Self {
        Self::from(*value)
    }
}

/// Reduces its input, so any `u64` is accepted.
impl From<u64> for DynField {
    fn from(value: u64) -> Self {
        Self::from(u128::from(value))
    }
}

impl From<&u128> for DynField {
    fn from(value: &u128) -> Self {
        Self::from(*value)
    }
}

/// Reduces its input, so any `u128` is accepted.
impl From<u128> for DynField {
    fn from(value: u128) -> Self {
        DynField {
            reduced_value: value % Self::modulus(),
        }
    }
}

//
// crypto-primitives
//

impl Bounded for DynField {
    #[inline(always)]
    fn min_value() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn max_value() -> Self {
        DynField {
            reduced_value: DynField::modulus() - 1,
        }
    }
}

impl BaseField for DynField {
    #[inline(always)]
    fn modulus() -> Self::Integer {
        let modulus = global::modulus();
        debug_assert!(modulus != 0, "Field modulus has not been set yet!");
        modulus
    }

    fn modulus_minus_one_div_two() -> Self::Integer {
        (Self::modulus() - 1) / 2
    }
}

impl WithAssociatedInteger for DynField {
    type Integer = u128;
}

impl LiftElement<u128> for DynField {
    fn lift(&self) -> u128 {
        self.reduced_value
    }
}

//
// Other
//

#[allow(dead_code)] // Cannot be gated for #[cfg(test)] or it won't be accessible to other crates
pub mod test_support {
    use super::*;
    use std::sync::{Mutex, PoisonError};

    /// Runs `test_code` under `modulus`. The modulus is process-wide and the test
    /// harness is multi-threaded, so every test that needs one holds this
    /// lock for its whole run.
    pub fn with_modulus<T>(modulus: u128, test_code: impl FnOnce() -> T) -> T {
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: the lock keeps every other test's values and operations out.
        unsafe { DynField::set_modulus(modulus) };
        test_code()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::with_modulus;
    use super::*;
    use crate::{Fq, FqDefault, Q100};
    use crypto_primitives::{ConstField, WithExtensionDegree};
    use rand_core::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    /// A prime small enough to check every pair of operands.
    const SMALL: u128 = 251;
    /// A 114-bit prime.
    const WIDE: u128 = (1 << 114) - 11;

    fn u128_of(rng: &mut Pcg64) -> u128 {
        (rng.next_u64() as u128) << 64 | rng.next_u64() as u128
    }

    /// Every pair of `extremes`, then `count` random pairs below `q`.
    fn operand_pairs<'a>(
        extremes: &'a [u128],
        q: u128,
        count: usize,
        rng: &'a mut Pcg64,
    ) -> impl Iterator<Item = (u128, u128)> + 'a {
        extremes
            .iter()
            .flat_map(move |&a| extremes.iter().map(move |&b| (a, b)))
            .chain((0..count).map(move |_| (u128_of(rng) % q, u128_of(rng) % q)))
    }

    #[test]
    fn ensure_traits() {
        fn assert_impl<T: BaseField + ConstField>() {}
        assert_impl::<DynField>();
    }

    #[test]
    #[should_panic(expected = "modulus must be odd")]
    fn set_modulus_rejects_even() {
        // SAFETY: rejected before anything is stored.
        unsafe { DynField::set_modulus(Q100 + 1) };
    }

    #[test]
    #[should_panic(expected = "modulus must be prime")]
    fn set_modulus_rejects_composite() {
        // SAFETY: rejected before anything is stored.
        unsafe { DynField::set_modulus(((1u128 << 54) - 33) * ((1u128 << 53) - 111)) };
    }

    #[test]
    #[should_panic(expected = "modulus must be below 2^126")]
    fn set_modulus_rejects_the_barrett_bound() {
        // SAFETY: rejected before anything is stored.
        unsafe { DynField::set_modulus((1 << 127) - 1) };
    }

    #[test]
    fn config_reports_the_installed_modulus() {
        with_modulus(Q100, || {
            let cfg = DynField::config();
            assert_eq!(cfg.modulus, Q100);
            assert_eq!(cfg.bits, FqDefault::META.bits);
            assert_eq!(DynField::modulus(), Q100);
            assert_eq!(DynField::modulus_minus_one_div_two(), (Q100 - 1) / 2);
            assert_eq!(DynField::min_value(), DynField::ZERO);
            assert_eq!(DynField::max_value().lift(), Q100 - 1);
            assert_eq!(DynField::extension_degree(), 1);
            assert_eq!(DynField::ONE.lift(), 1);
            assert!(DynField::ZERO.is_zero());
        });
    }

    /// Every ring operation against `Fq<Q>`, whose Barrett is checked against
    /// long division.
    fn agrees_with_fq<const Q: u128>(rng: &mut Pcg64) {
        with_modulus(Q, || {
            for (a, b) in operand_pairs(&[0, 1, Q - 1, Q / 2], Q, 512, rng) {
                let (x, y) = (DynField::from(a), DynField::from(b));
                let (fx, fy) = (Fq::<Q>::from(a), Fq::<Q>::from(b));
                assert_eq!((x + y).lift(), (fx + fy).lift(), "{a} + {b}");
                assert_eq!((x - y).lift(), (fx - fy).lift(), "{a} - {b}");
                assert_eq!((x * y).lift(), (fx * fy).lift(), "{a} * {b}");
                assert_eq!((-x).lift(), (-fx).lift(), "-{a}");
            }
        });
    }

    #[test]
    fn matches_the_compile_time_field() {
        let mut rng = Pcg64::seed_from_u64(501);
        agrees_with_fq::<SMALL>(&mut rng);
        agrees_with_fq::<Q100>(&mut rng);
        agrees_with_fq::<WIDE>(&mut rng);
    }

    /// The largest prime below the Barrett bound, where the quotient estimate
    /// has the least slack, against the add-and-double multiply, which shares
    /// nothing with Barrett.
    #[test]
    fn multiplies_at_the_widest_modulus() {
        let mut q = (1u128 << 126) - 1;
        while !helpers::is_prime(q) {
            q -= 2;
        }
        let mut rng = Pcg64::seed_from_u64(502);
        with_modulus(q, || {
            for (a, b) in operand_pairs(&[0, 1, q - 1, q - 2, q / 2], q, 512, &mut rng) {
                let got = (DynField::from(a) * DynField::from(b)).lift();
                assert_eq!(got, helpers::mul_mod(a, b, q), "{a} * {b}");
            }
        });
    }

    #[test]
    fn exhaustive_over_a_small_modulus() {
        with_modulus(SMALL, || {
            for a in 0..SMALL {
                for b in 0..SMALL {
                    let (x, y) = (DynField::from(a), DynField::from(b));
                    assert_eq!((x * y).lift(), a * b % SMALL, "{a} * {b}");
                    assert_eq!((x + y).lift(), (a + b) % SMALL, "{a} + {b}");
                    assert_eq!((x - y).lift(), (a + SMALL - b) % SMALL, "{a} - {b}");
                }
            }
        });
    }

    #[test]
    fn from_reduces() {
        with_modulus(Q100, || {
            assert_eq!(DynField::from(Q100).lift(), 0);
            assert_eq!(DynField::from(Q100 + 1).lift(), 1);
            assert_eq!(DynField::from(u128::MAX).lift(), u128::MAX % Q100);
            assert_eq!(DynField::from(&u128::MAX), DynField::from(u128::MAX));
            assert_eq!(DynField::from(u64::MAX).lift(), u128::from(u64::MAX));
            assert_eq!(DynField::from(&u64::MAX), DynField::from(u64::MAX));
            assert_eq!(DynField::from(true), DynField::ONE);
            assert_eq!(DynField::from(false), DynField::ZERO);
        });
    }

    #[test]
    fn values_follow_the_installed_modulus() {
        with_modulus(SMALL, || {
            let seven = DynField::from(7u64);
            assert_eq!((seven * seven * seven).lift(), 343 % SMALL);
            // SAFETY: nothing made under `SMALL` is used past here; the lock
            // is held.
            unsafe { DynField::set_modulus(Q100) };
            let seven = DynField::from(7u64);
            assert_eq!((seven * seven * seven).lift(), 343);
            assert_eq!(DynField::from(SMALL).lift(), SMALL);
        });
    }
}
