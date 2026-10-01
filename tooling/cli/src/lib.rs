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
    modulus: BigInt,
}

impl<F: BitzClaimField> ProjectConstraint<BigInt, F> for ProjectBigIntToField {
    fn prepare() -> Self {
        let modulus = F::modulus().to_u128().expect("the modulus fits a u128");
        Self {
            modulus: BigInt::from(modulus),
        }
    }

    fn project(&self, constraint: &BigInt) -> F {
        let mut reduced = constraint % &self.modulus;
        if reduced.is_negative() {
            reduced += &self.modulus;
        }
        F::from(
            reduced
                .to_u128()
                .expect("a canonical residue always fits a u128"),
        )
    }
}
