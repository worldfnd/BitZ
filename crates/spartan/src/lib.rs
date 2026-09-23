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

use field::{FqDefault, Q100};
use num_traits::{Signed, ToPrimitive};
use std::sync::LazyLock;

/// Reduces a signed integer canonically modulo Q100.
pub fn bigint_to_fq(value: &num_bigint::BigInt) -> FqDefault {
    static FQ_DEFAULT_MODULUS: LazyLock<num_bigint::BigInt> =
        LazyLock::new(|| num_bigint::BigInt::from(Q100));
    let modulus = &*FQ_DEFAULT_MODULUS;
    let mut reduced = value % modulus;
    if reduced.is_negative() {
        reduced += modulus;
    }
    FqDefault::from(
        reduced
            .to_u128()
            .expect("a canonical Q100 residue always fits a u128"),
    )
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
