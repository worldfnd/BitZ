//! Binary and prime field arithmetic.

#[cfg(feature = "spongefish")]
mod codec;
pub mod fq;
pub mod gf128;

pub use fq::{Fq, FqDefault, FqRuntime, ModulusError, Q100, RUNTIME, is_probable_prime, modulus, set_modulus};
pub use gf128::{F128, FixedBasePow, Wide256};
