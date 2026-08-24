//! Spartan PIOP interfaces.
//!
//! The outer prover and reusable sumcheck verifier are implemented here.

pub mod sumcheck;

pub use sumcheck::{
    OuterSumcheckOutput, OuterSumcheckProof, OuterSumcheckVerifierOutput, R1csProductMles,
    SumcheckError, SumcheckProof, SumcheckProverOutput, prove_outer_sumcheck,
};
