//! Spongefish codecs for the field types.
//!
//! Wire formats, fixed here and nowhere else:
//!
//! - `F128`: [`F128::to_bytes`], 16 bytes. Total in both directions — every
//!   16-byte string is an element — so `Decoding` squeezes 16 bytes and gets a
//!   uniform sample through the same bijection.
//! - `Fq<Q>`: the reduced representative as 16 little-endian bytes. Decoding
//!   rejects anything at or above `Q`, so each element has exactly one wire
//!   form. There is no `Decoding`: in the basic case, `deg(K) <= 1`, every
//!   challenge is `F128` and `Fq` is never sampled. The `deg(K) > 1`
//!   projection step samples `α` from a prime field, which would need a
//!   wider squeeze to kill the modular bias — added when that step lands.
//!
//! `NargSerialize` comes from spongefish's blanket impl over `Encoding`.

use crypto_primitives::LiftElement;
use spongefish::{
    ByteArray, Decoding, Encoding, NargDeserialize, VerificationError, VerificationResult,
};

use crate::{F128, Fq};

impl Encoding<[u8]> for F128 {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.to_bytes()
    }
}

impl Decoding<[u8]> for F128 {
    type Repr = ByteArray<16>;

    fn decode(buf: Self::Repr) -> Self {
        Self::from_bytes(*buf.as_ref())
    }
}

impl NargDeserialize for F128 {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        <[u8; 16]>::deserialize_from_narg(buf).map(Self::from_bytes)
    }
}

impl<const Q: u128> Encoding<[u8]> for Fq<Q> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.lift().to_le_bytes()
    }
}

impl<const Q: u128> NargDeserialize for Fq<Q> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        // Stage the cursor: the contract requires `buf` untouched on failure,
        // and the range check can still fail after the read succeeds.
        let mut rest = *buf;
        let value = u128::from_le_bytes(<[u8; 16]>::deserialize_from_narg(&mut rest)?);
        if value >= Q {
            return Err(VerificationError);
        }
        *buf = rest;
        Ok(Self::from(value))
    }
}

#[cfg(test)]
mod tests {
    use num_traits::{ConstOne, ConstZero};
    use spongefish::NargSerialize;

    use super::*;
    use crate::{FqDefault, Q100};

    fn f128_cases() -> [F128; 4] {
        [
            F128::ZERO,
            F128::ONE,
            F128::GENERATOR,
            F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
        ]
    }

    #[test]
    fn f128_encoding_is_to_bytes() {
        for a in f128_cases() {
            assert_eq!(a.encode().as_ref(), a.to_bytes());
            assert_eq!(a.serialize_into_new_narg().as_ref(), a.to_bytes());
        }
    }

    #[test]
    fn f128_decoding_is_from_bytes() {
        let mut repr = <F128 as Decoding>::Repr::default();
        repr.as_mut().copy_from_slice(&[0xa5; 16]);
        assert_eq!(F128::decode(repr), F128::from_bytes([0xa5; 16]));
    }

    #[test]
    fn f128_narg_round_trip() {
        for a in f128_cases() {
            let mut narg = Vec::new();
            a.serialize_into_narg(&mut narg);
            narg.extend_from_slice(b"tail");

            let mut buf = narg.as_slice();
            assert_eq!(F128::deserialize_from_narg(&mut buf).unwrap(), a);
            assert_eq!(buf, b"tail");
        }
    }

    #[test]
    fn f128_narg_rejects_short_input() {
        let narg = [0u8; 15];
        let mut buf = narg.as_slice();
        assert!(F128::deserialize_from_narg(&mut buf).is_err());
        assert_eq!(buf, narg);
    }

    #[test]
    fn fq_encoding_is_le_value() {
        for v in [0u128, 1, 12345, Q100 - 1] {
            let a = FqDefault::from(v);
            assert_eq!(a.encode().as_ref(), v.to_le_bytes());
        }
    }

    #[test]
    fn fq_narg_round_trip() {
        for v in [0u128, 1, 12345, Q100 - 1] {
            let a = FqDefault::from(v);
            let mut narg = Vec::new();
            a.serialize_into_narg(&mut narg);

            let mut buf = narg.as_slice();
            assert_eq!(FqDefault::deserialize_from_narg(&mut buf).unwrap(), a);
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn fq_narg_rejects_non_canonical() {
        // One wire form per element: the representative must be reduced.
        for v in [Q100, Q100 + 1, u128::MAX] {
            let narg = v.to_le_bytes();
            let mut buf = narg.as_slice();
            assert!(FqDefault::deserialize_from_narg(&mut buf).is_err());
            assert_eq!(buf, narg, "cursor must not move on failure");
        }
    }
}
