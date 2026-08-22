//! Spartan PIOP interfaces.
//!
//! The outer prover and reusable sumcheck verifier are implemented here.

pub mod sumcheck;

pub use sumcheck::{
    InnerSumcheckOutput, OuterSumcheckOutput, OuterSumcheckProof, OuterSumcheckVerifierOutput,
    R1csProductMles, SumcheckError, SumcheckProof, SumcheckProverOutput, prove_inner_sumcheck,
    prove_outer_sumcheck, verify_outer_sumcheck,
};
