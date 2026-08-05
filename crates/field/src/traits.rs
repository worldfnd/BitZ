use std::ops::{Add, Mul, Sub};

use crate::{fq::Fq, gf128::F128};

/// Minimal field operations shared by the polynomial utilities.
pub trait Field:
    Copy + From<u128> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Default
{
}

/// A type with a constant additive identity.
pub trait ConstZero {
    const ZERO: Self;
}

/// A type with a constant multiplicative identity.
pub trait ConstOne {
    const ONE: Self;
}

impl Field for F128 {}

impl ConstZero for F128 {
    const ZERO: Self = F128::ZERO;
}

impl ConstOne for F128 {
    const ONE: Self = F128::ONE;
}

impl<const Q: u128> Field for Fq<Q> {}

impl<const Q: u128> ConstZero for Fq<Q> {
    const ZERO: Self = Fq::ZERO;
}

impl<const Q: u128> ConstOne for Fq<Q> {
    const ONE: Self = Fq::ONE;
}
