//! Field with modulus not known at compile-time.
//!
//! We only expect to have one of those at any given time, so the modulus is shared globally.

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
use std::sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard};

static CFG: LazyLock<RwLock<DynFieldConfig>> = LazyLock::new(|| {
    RwLock::new(DynFieldConfig {
        initialized: false,
        modulus: 0,
    })
});

/// Global config for [`DynField`] shared among all instances.
#[derive(Debug, Copy, Clone)]
pub struct DynFieldConfig {
    initialized: bool,
    modulus: u128,
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, InfallibleCheckedOp)]
#[infallible_checked_unary_op((CheckedNeg, neg))]
#[infallible_checked_binary_op((CheckedAdd, add), (CheckedSub, sub), (CheckedMul, mul))]
#[repr(transparent)]
pub struct DynField {
    reduced_value: u128,
}

impl DynField {
    /// Set modulus globally. Marked `unsafe` to emphasize that there should be no [`DynField`] instances
    /// remaining anywhere, or they will silently become invalid.
    pub unsafe fn set_modulus(modulus: u128) {
        let mut cfg = CFG
            .write()
            .expect("Failed to acquire write lock on DynFieldConfig");
        cfg.initialized = true;
        cfg.modulus = modulus;
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
    const ONE: Self = DynField { reduced_value: 0 };
}

//
// Basic arithmetic operations
//

impl Neg for DynField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        todo!()
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

impl Pow<u32> for DynField {
    type Output = Self;

    fn pow(self, exp: u32) -> Self::Output {
        todo!()
    }
}

impl Pow<u128> for DynField {
    type Output = Self;

    fn pow(self, exp: u128) -> Self::Output {
        todo!()
    }
}

impl Pow<&u128> for DynField {
    type Output = Self;

    fn pow(self, exp: &u128) -> Self::Output {
        self.pow(*exp)
    }
}

impl Inv for DynField {
    type Output = Option<Self>;

    #[inline(always)]
    fn inv(self) -> Self::Output {
        todo!()
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
        todo!()
    }
}

impl SubAssign for DynField {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl MulAssign for DynField {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl DivAssign for DynField {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        todo!()
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
        todo!()
    }
}

impl From<u64> for DynField {
    fn from(value: u64) -> Self {
        todo!()
    }
}

impl From<&u128> for DynField {
    fn from(value: &u128) -> Self {
        todo!()
    }
}

impl From<u128> for DynField {
    fn from(value: u128) -> Self {
        todo!()
    }
}

// TODO!

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
    fn modulus() -> Self::Integer {
        let cfg = CFG
            .read()
            .expect("Failed to acquire read lock on DynFieldConfig");
        debug_assert!(cfg.initialized, "Field modulus has not been set yet!");
        cfg.modulus
    }

    fn modulus_minus_one_div_two() -> Self::Integer {
        todo!()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::{BaseField, ConstField};
}
