//! What the prover and the verifier must arrive at identically.
//!
//! Nothing here is secret or transmitted: every value is a function of the
//! public shape and the caller's claim, so both sides derive it rather than
//! one side sending it.

pub mod claim;
pub mod fold;
pub mod opening;
pub mod params;
pub mod shape;
pub mod table;
pub mod virtual_map;

pub use claim::{ClaimError, LinearClaim, Root};
pub use fold::{
    Fold, FoldError, column_images, fold_column, fold_columns, reconstruct, row_images,
};
pub use opening::OpeningQuery;
pub use params::{BitZParams, ParamsError, VirtualParams, VirtualParamsError};
pub use shape::{Shape, ShapeError};
pub use table::{BitTable, TableError, TransposeError, TransposedBitTable};
pub use virtual_map::{
    TransposedWeights, VirtualMap, VirtualMapError, VirtualStatement, VirtualStatementError,
};

use crypto_primitives::{BaseField, Field, Semiring};
use std::ops::Neg;

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
    //+ Bits + IntoWords
    pub trait BitzSemiring: Semiring + From<u64>
}

define_blanket_trait! {
    // Since BigInt does not support CheckedNeg and CheckedRem, we can't use Ring here
    pub trait BitzRing: BitzSemiring + Neg<Output = Self>
}

define_blanket_trait! {
    pub trait BitzField:
        Field
        + Copy
        + spongefish::Encoding<[u8]>
        + transcript::TranscriptChallenge
}

define_blanket_trait! {
    pub trait BitzBaseField: BitzField + BaseField
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::dynamic::DynField;
    use field::{F128, FqDefault};

    #[test]
    fn ensure_traits() {
        fn assert_impl_semiring<T: BitzSemiring>() {}
        assert_impl_semiring::<num_bigint::BigInt>();
        assert_impl_semiring::<num_bigint::BigUint>();
        assert_impl_semiring::<FqDefault>();
        assert_impl_semiring::<F128>();
        assert_impl_semiring::<DynField>();

        fn assert_impl_ring<T: BitzRing>() {}
        assert_impl_ring::<num_bigint::BigInt>();
        assert_impl_ring::<FqDefault>();
        assert_impl_ring::<F128>();
        assert_impl_ring::<DynField>();

        fn assert_impl_field<T: BitzField>() {}
        assert_impl_field::<FqDefault>();
        assert_impl_field::<F128>();
        assert_impl_field::<DynField>();

        fn assert_impl_base_field<T: BitzBaseField>() {}
        assert_impl_base_field::<FqDefault>();
        assert_impl_base_field::<DynField>();
    }
}
