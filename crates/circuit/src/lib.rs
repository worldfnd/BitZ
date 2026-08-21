//! Backend-independent operations used to build an F2Z circuit.
//!
//! The traits in this crate deliberately separate symbolic witnesses from the
//! values used while generating a witness.  In particular, a [`Circuit::hint`]
//! receives a [`WitnessContext`], through which it can evaluate captured
//! symbolic witnesses when witness generation is run.

use std::error::Error;
use std::fmt::{self, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use ark_ff::BigInteger;
use num_traits::{One, Zero};

pub mod sha256;

/// An error raised while evaluating a witness hint.
///
/// Hints are fallible because their inputs may not be in the domain expected by
/// a gadget (for example, an unsigned decomposition hint may receive a negative
/// value).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HintError {
    message: String,
}

impl HintError {
    /// Creates a hint error with a human-readable message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for HintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for HintError {}

impl From<String> for HintError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for HintError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// The result returned by a witness hint.
pub type HintResult<T> = Result<T, HintError>;

/// Values available to a hint while generating a witness.
///
/// A context is intentionally read-only and exposes only evaluation.  This
/// prevents a hint from adding constraints or allocating witnesses as a side
/// effect.  The symbolic witness types may be handles, linear combinations, or
/// another backend-specific representation.
pub trait WitnessContext<ZW, BW, C> {
    /// Evaluates a Z-side symbolic witness (or linear combination).
    fn eval_z(&self, witness: &ZW) -> C;

    /// Evaluates an F2-side symbolic witness (or linear combination).
    fn eval_bool(&self, witness: &BW) -> bool;
}

/// Arithmetic required of coefficients on the Z side of an F2Z circuit.
///
/// This is a ring-like programming interface, rather than a claim that every
/// implementation obeys the ring laws.  It is kept separate from the symbolic
/// witness type so backends are not forced to use integers for coefficients.
pub trait Coefficient:
    Clone
    + Zero
    + One
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Sum<Self>
{
    /// Constructs a coefficient from an arbitrary-width arkworks integer.
    ///
    /// [`BigInteger`] stores little-endian `u64` limbs. This default
    /// implementation uses only the coefficient's ring operations, so it also
    /// works for coefficient types that reduce integers modulo some modulus.
    fn from_big_integer<B: BigInteger>(value: B) -> Self {
        let mut result = Self::zero();
        let mut bit_value = Self::one();

        for &limb in value.as_ref() {
            let mut remaining = limb;
            for _ in 0..u64::BITS {
                if remaining & 1 == 1 {
                    result += bit_value.clone();
                }
                remaining >>= 1;
                bit_value += bit_value.clone();
            }
        }

        result
    }
}

impl<T> Coefficient for T where
    T: Clone
        + Zero
        + One
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Mul<Output = T>
        + Neg<Output = T>
        + Sum<T>
{
}

/// Arithmetic required of a symbolic Z witness or linear combination.
///
/// Scalar multiplication is written `witness * coefficient`.  Putting the
/// backend-local witness type on the left lets generic backends implement the
/// operation even when the coefficient type comes from another crate.
pub trait ZWitness<C: Coefficient>:
    Clone
    + From<C>
    + Zero
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
    + Mul<C, Output = Self>
    + Sum<Self>
{
}

impl<T, C> ZWitness<C> for T
where
    C: Coefficient,
    T: Clone
        + From<C>
        + Zero
        + Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Neg<Output = T>
        + Mul<C, Output = T>
        + Sum<T>,
{
}

/// Arithmetic required of a symbolic Boolean witness or F2 linear combination.
///
/// Addition has F2 semantics: implementations should interpret it as XOR.
pub trait BoolWitness:
    Clone + From<bool> + Zero + Add<Output = Self> + AddAssign + Sum<Self>
{
}

impl<T> BoolWitness for T where T: Clone + From<bool> + Zero + Add<Output = T> + AddAssign + Sum<T> {}

/// Operations needed to construct an F2Z circuit.
///
/// `ZW`, `BW`, and `C` are deliberately independent: they are respectively the
/// backend's symbolic Z witnesses, symbolic Boolean witnesses, and Z-side
/// coefficient/value type.
pub trait Circuit<ZW, BW, C>
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness,
{
    /// Allocates `N` Boolean witnesses whose values are computed by `hint`.
    ///
    /// The closure is called during witness generation, not necessarily while
    /// the circuit is being built.  It can capture symbolic witnesses and use
    /// the supplied sub-context to evaluate them.  Consequently, captured data
    /// must be owned and safe to retain and invoke from a worker thread.
    fn hint<const N: usize, H>(&mut self, hint: H) -> [BW; N]
    where
        H: Fn(&dyn WitnessContext<ZW, BW, C>) -> HintResult<[bool; N]> + Send + Sync + 'static;

    /// Converts an F2 linear combination into a constrained Z witness.
    fn f2z(&mut self, value: BW) -> ZW;

    /// Asserts the rank-1 constraint `a * b = c` on the Z side.
    fn assert_r1c(&mut self, a: ZW, b: ZW, c: ZW);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{BigInteger64, BigInteger128};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Mod7(i32);

    impl Mod7 {
        fn new(value: i32) -> Self {
            Self(value.rem_euclid(7))
        }
    }

    impl Zero for Mod7 {
        fn zero() -> Self {
            Self(0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }

    impl One for Mod7 {
        fn one() -> Self {
            Self(1)
        }
    }

    impl Add for Mod7 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self::new(self.0 + rhs.0)
        }
    }

    impl AddAssign for Mod7 {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sub for Mod7 {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self::new(self.0 - rhs.0)
        }
    }

    impl SubAssign for Mod7 {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl Mul for Mod7 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self::new(self.0 * rhs.0)
        }
    }

    impl Neg for Mod7 {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self::new(-self.0)
        }
    }

    impl Sum for Mod7 {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), Add::add)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Z(Mod7);

    impl From<Mod7> for Z {
        fn from(value: Mod7) -> Self {
            Self(value)
        }
    }

    impl Zero for Z {
        fn zero() -> Self {
            Self(Mod7::zero())
        }

        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }

    impl Add for Z {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl AddAssign for Z {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sub for Z {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl SubAssign for Z {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl Neg for Z {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl Mul<Mod7> for Z {
        type Output = Self;

        fn mul(self, rhs: Mod7) -> Self::Output {
            Self(self.0 * rhs)
        }
    }

    impl Sum for Z {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self(Mod7::zero()), Add::add)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Bit(bool);

    impl From<bool> for Bit {
        fn from(value: bool) -> Self {
            Self(value)
        }
    }

    impl Zero for Bit {
        fn zero() -> Self {
            Self(false)
        }

        fn is_zero(&self) -> bool {
            !self.0
        }
    }

    impl Add for Bit {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 != rhs.0)
        }
    }

    impl AddAssign for Bit {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sum for Bit {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self(false), Add::add)
        }
    }

    struct Values;

    impl WitnessContext<Z, Bit, Mod7> for Values {
        fn eval_z(&self, witness: &Z) -> Mod7 {
            witness.0
        }

        fn eval_bool(&self, witness: &Bit) -> bool {
            witness.0
        }
    }

    #[derive(Default)]
    struct TestCircuit {
        constraints: Vec<(Z, Z, Z)>,
    }

    impl Circuit<Z, Bit, Mod7> for TestCircuit {
        fn hint<const N: usize, H>(&mut self, hint: H) -> [Bit; N]
        where
            H: Fn(&dyn WitnessContext<Z, Bit, Mod7>) -> HintResult<[bool; N]>
                + Send
                + Sync
                + 'static,
        {
            hint(&Values).expect("test hint should succeed").map(Bit)
        }

        fn f2z(&mut self, value: Bit) -> Z {
            Z(Mod7::new(i32::from(value.0)))
        }

        fn assert_r1c(&mut self, a: Z, b: Z, c: Z) {
            self.constraints.push((a, b, c));
        }
    }

    #[test]
    fn hints_evaluate_both_witness_kinds_through_the_context() {
        let mut circuit = TestCircuit::default();
        let z = Z(Mod7::new(5));
        let bit = Bit(true);
        let hint_z = z;
        let hint_bit = bit;

        let [high, low] = circuit.hint(move |context| {
            let z = context.eval_z(&hint_z).0;
            let bit = context.eval_bool(&hint_bit);
            Ok([z >= 4 && bit, z % 2 == 1])
        });

        assert_eq!([high, low], [Bit(true), Bit(true)]);

        let lifted = circuit.f2z(high);
        let expression = lifted * Mod7::new(3) + Z::from(Mod7::new(2));
        circuit.assert_r1c(Z::from(Mod7::one()), expression, z);
        assert_eq!(circuit.constraints, vec![(Z(Mod7::one()), z, z)]);
    }

    #[test]
    fn hint_error_preserves_its_message() {
        let error = HintError::new("negative input");
        assert_eq!(error.message(), "negative input");
        assert_eq!(error.to_string(), "negative input");
    }

    #[test]
    fn coefficient_accepts_arkworks_big_integers_of_any_width() {
        assert_eq!(
            Mod7::from_big_integer(BigInteger64::from(10_u64)),
            Mod7::new(10)
        );
        assert_eq!(
            Mod7::from_big_integer(BigInteger128::new([0, 1])),
            Mod7::new(2)
        );
    }
}
