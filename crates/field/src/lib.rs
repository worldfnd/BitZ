//! Binary and prime field arithmetic.

#[cfg(feature = "spongefish")]
mod codec;
pub mod dynamic;
pub mod fq;
pub mod gf128;
pub mod helpers;

pub use dynamic::DynField;
pub use fq::{Fq, FqDefault, Q100};
pub use gf128::{F128, FixedBasePow, Wide256};
pub use helpers::MAX_MODULUS_BITS;

/// A prime field whose modulus is installed at run time, so a proof can run
/// over a prime its transcript draws.
pub trait FieldWithDynamicModulus {
    /// Installs `modulus` process-wide.
    ///
    /// # Safety
    ///
    /// No instances of this field value may be alive, in any thread, or they may
    /// become invalid.
    ///
    /// Despite that, no low-level Rust guarantees are violated and thus no true
    /// Undefined Behavior™ could happen under any circumstances.
    unsafe fn set_modulus(modulus: u128);
}
