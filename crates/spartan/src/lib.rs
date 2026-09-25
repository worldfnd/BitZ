//! Spartan PIOP interfaces.
//!
//! The outer prover and reusable sumcheck verifier are implemented here.

pub mod matrix;
pub mod piop;
pub mod sumcheck;

pub use matrix::{
    PreparedConstraintMatrices, SpartanMatrixError, build_assignment_mle, build_product_mles,
};
pub use piop::{
    SpartanError, SpartanPiopProof, prove_spartan_piop, verify_spartan_proof,
    verify_spartan_with_mle_claim,
};
pub use sumcheck::{
    InnerSumcheckOutput, OuterSumcheckOutput, OuterSumcheckProof, OuterSumcheckVerifierOutput,
    R1csProductMles, SumcheckError, SumcheckProof, SumcheckProverOutput, prove_inner_sumcheck,
    prove_outer_sumcheck,
};

use common::BitzClaimField;
use field::{FqDefault, Q100};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};
use std::sync::LazyLock;

/// Reduces a signed integer canonically modulo `modulus`, which must be
/// `F::modulus()`; the caller lifts it once for all its coefficients.
pub fn bigint_to_field<F: BitzClaimField>(value: &BigInt, modulus: &BigInt) -> F {
    let mut reduced = value % modulus;
    if reduced.is_negative() {
        reduced += modulus;
    }
    F::from(
        reduced
            .to_u128()
            .expect("a canonical residue always fits a u128"),
    )
}

/// Reduces a signed integer canonically modulo Q100.
pub fn bigint_to_fq(value: &BigInt) -> FqDefault {
    static FQ_DEFAULT_MODULUS: LazyLock<BigInt> = LazyLock::new(|| BigInt::from(Q100));
    bigint_to_field(value, &FQ_DEFAULT_MODULUS)
}

#[cfg(test)]
mod tests {
    use common::BitzRing;

    #[test]
    fn ensure_traits() {
        fn assert_impl<T: BitzRing>() {}
        assert_impl::<num_bigint::BigInt>();
    }
}
