//! Circuit proof composition and benchmark execution for the BitZ CLI.

pub mod benchmark;
pub mod circuits;
pub mod end_to_end;

use field::Fq;
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};

/// Trait for projecting a constraint onto a field.
/// Needs to be prepared right before the projection, as field might be
/// reconfigured.
pub trait ProjectConstraint<R, F>: Send + Sync {
    fn prepare() -> Self;

    fn project(&self, constraint: &R) -> F;
}

/// Projects [`BigInt`] constraints onto [`Fq`] by reducing it canonically modulo `Q`.
pub struct ProjectBigIntToFq {
    modulus: BigInt,
}

impl<const Q: u128> ProjectConstraint<BigInt, Fq<Q>> for ProjectBigIntToFq {
    fn prepare() -> Self {
        Self {
            modulus: BigInt::from(Q),
        }
    }

    fn project(&self, constraint: &BigInt) -> Fq<Q> {
        let mut reduced = constraint % &self.modulus;
        if reduced.is_negative() {
            reduced += &self.modulus;
        }
        Fq::from(
            reduced
                .to_u128()
                .expect("a canonical residue always fits a u128"),
        )
    }
}
