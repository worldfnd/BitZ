//! A value-free circuit backend that only counts the generated layout.

use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use num_traits::{One, Zero};

use crate::{Circuit, HintResult, WitnessContext};

/// A zero-sized placeholder for coefficients and both kinds of witnesses.
///
/// Every operation discards its operands. This is sufficient for circuit
/// construction because [`Stats`] never invokes hint bodies or evaluates a
/// witness.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dummy;

impl From<bool> for Dummy {
    fn from(_: bool) -> Self {
        Self
    }
}

impl Zero for Dummy {
    fn zero() -> Self {
        Self
    }

    fn is_zero(&self) -> bool {
        true
    }
}

impl One for Dummy {
    fn one() -> Self {
        Self
    }
}

impl Add for Dummy {
    type Output = Self;

    fn add(self, _: Self) -> Self::Output {
        Self
    }
}

impl AddAssign for Dummy {
    fn add_assign(&mut self, _: Self) {}
}

impl Sub for Dummy {
    type Output = Self;

    fn sub(self, _: Self) -> Self::Output {
        Self
    }
}

impl SubAssign for Dummy {
    fn sub_assign(&mut self, _: Self) {}
}

impl Mul for Dummy {
    type Output = Self;

    fn mul(self, _: Self) -> Self::Output {
        Self
    }
}

impl Neg for Dummy {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self
    }
}

impl Sum for Dummy {
    fn sum<I: Iterator<Item = Self>>(_: I) -> Self {
        Self
    }
}

/// The dimensions printed by Freigen's constraint-system statistics.
///
/// Freigen defines `mRows` as the number of F2Z rows plus one. Its `mCols` is
/// the largest referenced, zero-based witness index plus two; for a densely
/// allocated circuit whose final witness is referenced, that is the witness
/// count plus one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LeanStats {
    pub m_rows: usize,
    pub m_cols: usize,
    pub r1cs_rows: usize,
}

/// Counts circuit allocation without retaining symbolic expressions or values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    /// Boolean witnesses supplied as inputs or allocated by hints.
    pub witnesses: usize,
    /// Calls to [`Circuit::f2z`].
    pub f2z_calls: usize,
    /// Calls to [`Circuit::assert_r1c`].
    pub constraints: usize,
}

impl Stats {
    /// Starts a counter with `input_witnesses` already allocated.
    pub const fn new(input_witnesses: usize) -> Self {
        Self {
            witnesses: input_witnesses,
            f2z_calls: 0,
            constraints: 0,
        }
    }

    /// Accounts for additional externally allocated Boolean witnesses.
    pub fn add_input_witnesses(&mut self, count: usize) {
        self.witnesses = self
            .witnesses
            .checked_add(count)
            .expect("witness count overflow");
    }

    /// Converts the raw counts to Freigen's dimensions, assuming witnesses are
    /// densely allocated and the final allocated witness is referenced.
    pub const fn lean_stats(&self) -> LeanStats {
        LeanStats {
            m_rows: self.f2z_calls + 1,
            m_cols: self.witnesses + 1,
            r1cs_rows: self.constraints,
        }
    }
}

impl Circuit<Dummy, Dummy, Dummy> for Stats {
    fn hint<const N: usize, H>(&mut self, _hint: H) -> [Dummy; N]
    where
        H: Fn(&dyn WitnessContext<Dummy, Dummy, Dummy>) -> HintResult<[bool; N]>
            + Send
            + Sync
            + 'static,
    {
        self.witnesses = self
            .witnesses
            .checked_add(N)
            .expect("witness count overflow");
        [Dummy; N]
    }

    fn f2z(&mut self, _: Dummy) -> Dummy {
        self.f2z_calls = self.f2z_calls.checked_add(1).expect("f2z count overflow");
        Dummy
    }

    fn assert_r1c(&mut self, _: Dummy, _: Dummy, _: Dummy) {
        self.constraints = self
            .constraints
            .checked_add(1)
            .expect("constraint count overflow");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_arithmetic_discards_values() {
        let mut value = Dummy::one();
        value += Dummy;
        value -= Dummy;
        assert_eq!(-value * Dummy, Dummy);
        assert!(Dummy::zero().is_zero());
    }

    #[test]
    fn counts_inputs_and_operations() {
        let mut stats = Stats::new(3);
        stats.add_input_witnesses(2);
        let _: [Dummy; 4] = stats.hint(|_| Ok([false; 4]));
        stats.f2z(Dummy);
        stats.assert_r1c(Dummy, Dummy, Dummy);

        assert_eq!(
            stats,
            Stats {
                witnesses: 9,
                f2z_calls: 1,
                constraints: 1,
            }
        );
        assert_eq!(
            stats.lean_stats(),
            LeanStats {
                m_rows: 2,
                m_cols: 10,
                r1cs_rows: 1,
            }
        );
    }
}
