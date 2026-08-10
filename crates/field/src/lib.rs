//! Binary and prime field arithmetic.

pub mod fq;
pub mod gf128;

pub use fq::{Fq, FqDefault, Q100};
pub use gf128::{F128, FixedBasePow, Wide256};
