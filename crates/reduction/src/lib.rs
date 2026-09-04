//! Step 4: the grand product over the folds, and the sumcheck that turns its
//! leaves into a claim on the committed bits.
//!
//! The fold round left both sides holding `g^{eta_j}` for every column and
//! `y_i = g^{pi_q^{-1}(v1_i)}` for every row. What is still unproved is that
//! each column's fold really was computed from the committed bits:
//!
//! ```text
//! g^{eta_j} = prod_i y_i^{f_ij}.
//! ```
//!
//! Each `f_ij` is a bit, so `y_i^{f_ij} = 1 + f_ij (y_i - 1)` is affine in it.
//! That makes the whole statement one product tree per column over leaves the
//! committed bits determine, which is what [`gkr`] proves.
//!
//! The forest leaves every column with its own claim, all at one shared point
//! `rho`. Two steps turn those into the single claim the opening discharges:
//!
//! 1. Batch the columns at `xi`, the challenge the fold round already drew:
//!    `y_xi = sum_j eq(j, xi) (leaf_j(rho) - 1)`. Subtracting one is what
//!    strips the affine constant and leaves something linear in the bits.
//! 2. Sumcheck `y_xi = sum_i R(i) m(i)` over the rows, where
//!    `R(i) = eq(i, rho) (y_i - 1)` is public and
//!    `m(i) = sum_j eq(j, xi) f_ij` is the bits batched the same way.
//!
//! The sumcheck ends at a row point `r`, and `m(r)` is the committed bits'
//! multilinear extension at `r ++ xi`. Dividing the sumcheck's terminal value
//! by the public `R(r)` is what recovers it.

pub mod grand_product;

pub use grand_product::{GrandProduct, ReductionError};
