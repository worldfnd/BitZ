//! Circuit proof composition and benchmark execution for the BitZ CLI.

pub mod benchmark;
pub mod circuits;
pub mod end_to_end;

use common::BitzClaimField;
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};

/// Trait for projecting a constraint onto a field.
/// Needs to be prepared right before the projection, as field might be
/// reconfigured.
pub trait ProjectConstraint<R, F>: Send + Sync {
    fn prepare() -> Self;

    fn project(&self, constraint: &R) -> F;
}

/// Projects [`BigInt`] constraints onto a prime field by reducing them
/// canonically modulo its modulus, read when prepared.
#[derive(Debug, Clone)]
pub struct ProjectBigIntToField {
    modulus: u128,
    wide_modulus: BigInt,
}

impl<F: BitzClaimField> ProjectConstraint<BigInt, F> for ProjectBigIntToField {
    fn prepare() -> Self {
        let modulus = F::modulus().to_u128().expect("the modulus fits a u128");
        Self {
            modulus,
            wide_modulus: BigInt::from(modulus),
        }
    }

    fn project(&self, constraint: &BigInt) -> F {
        // Coefficients are small in practice; reduce them without the heap.
        if let Some(value) = constraint.to_i128() {
            let magnitude = value.unsigned_abs();
            let magnitude = if magnitude < self.modulus {
                magnitude
            } else {
                magnitude % self.modulus
            };
            return F::from(if value < 0 && magnitude != 0 {
                self.modulus - magnitude
            } else {
                magnitude
            });
        }
        let mut reduced = constraint % &self.wide_modulus;
        if reduced.is_negative() {
            reduced += &self.wide_modulus;
        }
        F::from(
            reduced
                .to_u128()
                .expect("a canonical residue always fits a u128"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::Q100;
    use num_traits::{One, Pow};

    type F = field::FqDefault;

    /// Every coefficient lands on its canonical residue, whichever path
    /// reduces it.
    #[test]
    fn projection_is_canonical_reduction() {
        let projection = <ProjectBigIntToField as ProjectConstraint<BigInt, F>>::prepare();
        let reference = |value: &BigInt| {
            let modulus = BigInt::from(Q100);
            let mut residue = value % &modulus;
            if residue.is_negative() {
                residue += &modulus;
            }
            F::from(residue.to_u128().unwrap())
        };
        let q = Q100 as i128;
        let mut values: Vec<BigInt> = [
            0,
            1,
            -1,
            7,
            -7,
            q,
            -q,
            q + 1,
            -q - 1,
            2 * q,
            i128::MAX,
            i128::MIN,
        ]
        .into_iter()
        .map(BigInt::from)
        .collect();
        let wide = BigInt::from(3).pow(130u32) + BigInt::one();
        values.extend([wide.clone(), -wide.clone(), &wide * &wide, -(&wide * &wide)]);
        for value in &values {
            let projected: F = projection.project(value);
            assert_eq!(projected, reference(value), "{value}");
        }
    }
}
