use std::ops::{Add, Mul, Sub};

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
