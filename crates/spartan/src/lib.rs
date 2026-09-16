//! Spartan PIOP interfaces.
//!
//! The outer prover and reusable sumcheck verifier are implemented here.

pub mod matrix;
pub mod piop;
pub mod sumcheck;

pub use matrix::{
    IntegerCoefficient, PreparedConstraintMatrices, PreparedIntegerMatrices, SpartanMatrixError,
    bigint_to_fq, bigint_to_fq_in, build_assignment_mle, build_product_mles,
    build_product_mles_in,
};
pub use piop::{
    SpartanError, SpartanPiopProof, prove_spartan_piop, prove_spartan_piop_absorbed,
    prove_spartan_piop_sampled, verify_spartan_proof, verify_spartan_proof_absorbed,
    verify_spartan_proof_sampled, verify_spartan_with_mle_claim,
};

pub use sumcheck::{
    InnerSumcheckOutput, OuterSumcheckOutput, OuterSumcheckProof, OuterSumcheckVerifierOutput,
    R1csProductMles, SumcheckError, SumcheckProof, SumcheckProverOutput, prove_inner_sumcheck,
    prove_outer_sumcheck,
};
