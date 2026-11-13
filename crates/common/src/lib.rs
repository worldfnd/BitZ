//! What the prover and the verifier must arrive at identically.
//!
//! Nothing here is secret or transmitted: every value is a function of the
//! public shape and the caller's claim, so both sides derive it rather than
//! one side sending it.

// `BitZParams::table` and `VirtualParams::table` read the prover's `F128`s in
// place as `Word`s. That keeps bit `v` of `lo || hi`, the coefficient of
// `X^v`, at bit `v` of the word only on a little-endian target: on a
// big-endian one the cast swaps the two `u64` halves, and the table returns
// row `r ^ 64` when asked for row `r`. spongefish refuses big-endian targets
// too, but that is its policy, not a guarantee this crate can lean on.
#[cfg(target_endian = "big")]
compile_error!("`common` reads `F128`s in place as `u128` words and needs a little-endian target");

pub mod claim;
pub mod fold;
pub mod matrix;
pub mod opening;
pub mod params;
pub mod security;
pub mod shape;
pub mod table;
pub mod virtual_map;

pub use claim::{ClaimError, LinearClaim, Root};
pub use fold::{
    Fold, FoldError, column_images, fold_column, fold_columns, reconstruct, row_images,
};
pub use matrix::BitMatrix;
pub use opening::OpeningQuery;
pub use params::{
    BitZParams, MIN_PRIME_BITS, ParamsError, VirtualParams, VirtualParamsError, prime_bits,
};
pub use security::SecurityLevel;
pub use shape::{Shape, ShapeError};
pub use table::{BitTable, TableError};
pub use virtual_map::{
    TransposedWeights, VirtualMap, VirtualMapError, VirtualStatement, VirtualStatementError,
};

use crypto_primitives::{BaseField, Field, Semiring, WithAssociatedInteger};
use num_traits::{Bounded, FromBytes, ToBytes, ToPrimitive};
use spongefish::{Encoding, NargDeserialize};
use std::ops::{BitAnd, Neg, ShrAssign};

/// Define an empty trait with the given supertraits, and make a blanket
/// implementation for it.
#[macro_export]
macro_rules! define_blanket_trait {
    ($(#[$attr:meta])* $vis:vis trait $trait_name:ident: $($bound:tt)+) => {
        $(#[$attr])*
        $vis trait $trait_name: $($bound)+ {}

        impl<T> $trait_name for T where T: $($bound)* {}
    };
}

define_blanket_trait! {
    pub trait BitzSemiring:
        Semiring
        + BitAnd<Output = Self>
        + ShrAssign<u32>
        + From<u64>
        + ToPrimitive
}

define_blanket_trait! {
    // Since BigInt does not support CheckedNeg and CheckedRem, we can't use Ring here

    /// The ring in which BitZ constraints live.
    pub trait BitzConstraintRing: BitzSemiring + Neg<Output = Self>
}

define_blanket_trait! {
    /// Any BitZ field type (base or GF128)
    pub trait BitzField:
        Field
        + Copy
        + Encoding<[u8]>
        + transcript::TranscriptChallenge
}

define_blanket_trait! {
    /// The prime field of a BitZ claim. Its representatives are the fold
    /// exponents, so the modulus must fit the `u128` exponent of the `F128`
    /// group; `From<u128>` takes a fold back into the field.
    pub trait BitzClaimField:
        BitzField
        + BaseField
        + WithAssociatedInteger<
            Integer:
                BitzSemiring
                + Bounded
                + From<u64>
                + From<u128>
                + FromBytes<Bytes: AsRef<[u8]> + Encoding<[u8]> + NargDeserialize>
                + ToBytes<Bytes: AsRef<[u8]>
        >
        + From<u64>>
        + From<u128>
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::dynamic::DynField;
    use field::{F128, FqDefault};

    #[test]
    fn ensure_traits() {
        fn assert_impl_semiring<T: BitzSemiring>() {}
        assert_impl_semiring::<u128>();
        assert_impl_semiring::<num_bigint::BigInt>();
        assert_impl_semiring::<num_bigint::BigUint>();

        fn assert_impl_ring<T: BitzConstraintRing>() {}
        assert_impl_ring::<i128>();
        assert_impl_ring::<num_bigint::BigInt>();

        fn assert_impl_field<T: BitzField>() {}
        assert_impl_field::<FqDefault>();
        assert_impl_field::<F128>();
        assert_impl_field::<DynField>();

        fn assert_impl_claim_field<T: BitzClaimField>() {}
        assert_impl_claim_field::<FqDefault>();
        assert_impl_claim_field::<DynField>();
    }
}
