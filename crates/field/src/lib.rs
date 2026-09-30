//! Binary and prime field arithmetic.

#[cfg(feature = "spongefish")]
mod codec;
pub mod dynamic;
pub mod fq;
pub mod gf128;
pub(crate) mod helpers;

pub use fq::{Fq, FqDefault, Q100};
pub use gf128::{F128, FixedBasePow, Wide256};
pub use helpers::MAX_MODULUS_BITS;
