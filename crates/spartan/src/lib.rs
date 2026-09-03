//! Spartan PIOP interfaces.
//!
//! The outer prover and reusable sumcheck verifier are implemented here.

pub mod matrix;
pub mod piop;
pub mod sumcheck;

pub use matrix::{
    PreparedConstraintMatrices, SpartanMatrixError, bigint_to_fq, build_assignment_mle,
    build_product_mles,
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
