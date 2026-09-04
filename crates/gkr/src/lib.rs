//! Grand products by GKR, over a forest of product trees.
//!
//! Proves `prod_{x in {0,1}^d} v_t(x) = root_t` for every tree `t` at once.
//! Trees of the same depth share their layer sumchecks, so a forest of `2^s`
//! trees costs one batched sumcheck per layer rather than `2^s` of them.
//!
//! F2Z reaches this with one tree per column: column `j`'s claim is
//! `g^{eta_j} = prod_i y_i^{f_ij}`, and because each `f_ij` is a bit,
//! `y_i^{f_ij} = 1 + f_ij (y_i - 1)` is affine in the committed bit. The
//! caller binds the leaf evaluations this returns to the committed bits.
//!
//! ## Layers
//!
//! Layer `k` holds `2^k` entries, layer `d` the leaves, layer `0` the root:
//!
//! ```text
//! layer_k[i] = layer_{k+1}[i] * layer_{k+1}[i + 2^k]
//! ```
//!
//! Reducing layer `k` to layer `k + 1` proves
//! `layer_k(r) = sum_z eq(r, z) left(z) right(z)`, a cubic sumcheck over
//! `z in {0,1}^k`, where `left` and `right` are the halves of layer `k + 1`.
//! Layer 0 has no variables, so it is the direct check `root = left * right`.
//! Each layer closes by drawing one challenge that folds the two child claims
//! into the single claim the next layer reduces.

pub mod forest;

pub use forest::{ForestError, TreeClaim, prove_product_forest, verify_product_forest};
