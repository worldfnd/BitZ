//! Proof host container: how a proof leaves and enters the host.
//!
//! The header is 32 bytes, then the narg stream, then the hint stream:
//!
//! ```text
//! offset size field
//! 0      8    magic = 46 32 5a 50 43 53 00 00   ("F2ZPCS\0\0")
//! 8      2    wire_version = 1
//! 10     2    header_len = 32
//! 12     4    flags = 0
//! 16     8    narg_byte_len
//! 24     8    hint_byte_len
//! 32     ..   narg bytes, then hint bytes
//! ```

use crate::reader::{self, BoundedReader};
use thiserror::Error;
use transcript::Proof;

/// The eight-byte proof container magic.
pub const MAGIC: [u8; 8] = *b"F2ZPCS\0\0";
/// The only wire version this codec reads or writes.
pub const WIRE_VERSION: u16 = 1;
/// The fixed header length, in bytes.
pub const HEADER_LEN: usize = 32;
/// No flag is defined in v1, so every bit is reserved and must be zero.
pub const FLAGS: u32 = 0;

/// A proof as it appears on the wire.
///
/// Decoding borrows: the two streams are subslices of the caller's buffer, so
/// nothing here allocates from a length the input chose. [`WireProof::unwrap`]
/// is where a caller opts into owning the bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireProof<'a> {
    /// The narg stream, as declared.
    pub narg_string: &'a [u8],
    /// The hint stream, as declared.
    pub hints: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ProofDecodingError {
    #[error("Proof is too short")]
    OutOfInput(#[from] reader::OutOfInput),
    #[error("Proof decoding has leftover data")]
    LeftoverInput(#[from] reader::LeftoverInput),
    #[error("The first eight bytes are not {:X?}", MAGIC)]
    BadMagic,
    #[error("Codec does not implement version {0}")]
    UnsupportedVersion(u16),
    #[error("The declared header length is not {HEADER_LEN}")]
    BadHeaderLen(usize),
    #[error("A reserved flag bit is set")]
    ReservedFlagsSet(u32),
    #[error("The declared lengths do not fit the address space")]
    LengthOverflow,
    #[error("Length mismatch: declared {declared}, was {actual}")]
    LengthMismatch { declared: usize, actual: usize },
}

impl<'a> WireProof<'a> {
    #[inline]
    pub fn new(proof: &'a Proof) -> Self {
        Self {
            narg_string: &proof.narg_string,
            hints: &proof.hints,
        }
    }

    /// Copies the streams out into an owned proof.
    #[inline]
    pub fn unwrap(&self) -> Proof {
        Proof {
            narg_string: self.narg_string.to_vec(),
            hints: self.hints.to_vec(),
        }
    }

    /// The length of this proof's encoding.
    pub fn byte_len(&self) -> usize {
        HEADER_LEN
            .saturating_add(self.narg_string.len())
            .saturating_add(self.hints.len())
    }

    /// Encodes proof into wire format, as per module's docs.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        self.extend_with_bytes(&mut out);
        out
    }

    /// Appends wire-encoded proof to `out`, as per module's docs.
    pub fn extend_with_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        out.extend_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        out.extend_from_slice(&FLAGS.to_le_bytes());
        out.extend_from_slice(&(self.narg_string.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.hints.len() as u64).to_le_bytes());
        out.extend_from_slice(self.narg_string);
        out.extend_from_slice(self.hints);
    }

    /// Decode proof from wire format, as per module's docs.
    ///
    /// Strictly checks all the lengths and conversions before decoding the actual data.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ProofDecodingError> {
        let mut reader = BoundedReader::new(bytes);

        let magic = reader.read_array::<8>()?;
        if magic != MAGIC {
            return Err(ProofDecodingError::BadMagic);
        }
        let version = reader.read_u16()?;
        if version != WIRE_VERSION {
            return Err(ProofDecodingError::UnsupportedVersion(version));
        }
        let header_len = reader.read_u16()? as usize;
        if header_len != HEADER_LEN {
            return Err(ProofDecodingError::BadHeaderLen(header_len));
        }
        let flags = reader.read_u32()?;
        if flags != FLAGS {
            return Err(ProofDecodingError::ReservedFlagsSet(flags));
        }

        let narg_byte_len =
            usize::try_from(reader.read_u64()?).map_err(|_| ProofDecodingError::LengthOverflow)?;
        let hint_byte_len =
            usize::try_from(reader.read_u64()?).map_err(|_| ProofDecodingError::LengthOverflow)?;
        debug_assert_eq!(reader.pos, HEADER_LEN);

        let declared = HEADER_LEN
            .checked_add(narg_byte_len)
            .and_then(|len| len.checked_add(hint_byte_len))
            .ok_or(ProofDecodingError::LengthOverflow)?;
        if declared != bytes.len() {
            return Err(ProofDecodingError::LengthMismatch {
                declared,
                actual: bytes.len(),
            });
        }

        // Neither read can fail, and `finish` cannot either: the equality above
        // makes both streams exactly fill what is left. `finish` is kept as the
        // backstop for a future weakening of that check, which is also why no
        // test can observe it -- removing it changes nothing today.
        let narg_string = reader.read_slice(narg_byte_len)?;
        let hints = reader.read_slice(hint_byte_len)?;
        reader.finish()?;

        Ok(Self { narg_string, hints })
    }
}

/// See [`WireProof::to_bytes`]
#[inline]
pub fn encode(proof: &Proof) -> Vec<u8> {
    WireProof::new(proof).to_bytes()
}

/// See [`WireProof::from_bytes`]
#[inline]
pub fn decode(bytes: &[u8]) -> Result<Proof, ProofDecodingError> {
    Ok(WireProof::from_bytes(bytes)?.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    /// One fold record: a `u128` exponent, little endian, as
    /// `prover::send_fold` writes it.
    const ONE_RECORD: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    /// Byte offsets of the header fields, in order.
    pub mod offset {
        pub const MAGIC: usize = 0;
        pub const WIRE_VERSION: usize = 8;
        pub const HEADER_LEN: usize = 10;
        pub const FLAGS: usize = 12;
        pub const NARG_BYTE_LEN: usize = 16;
        pub const HINT_BYTE_LEN: usize = 24;
    }

    fn proof(narg: &[u8], hints: &[u8]) -> Proof {
        Proof {
            narg_string: narg.to_vec(),
            hints: hints.to_vec(),
        }
    }

    fn one_record() -> Proof {
        proof(&ONE_RECORD, &[])
    }

    /// The positive case: the grammar read off bytes written by hand, then a
    /// realistic record and the empty proof.
    #[test]
    fn the_encoding_is_the_grammar_in_the_module_docs() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"F2ZPCS\0\0");
        bytes.extend_from_slice(&[0x01, 0x00]); // wire_version = 1
        bytes.extend_from_slice(&[0x20, 0x00]); // header_len = 32
        bytes.extend_from_slice(&[0x00; 4]); // flags = 0
        bytes.extend_from_slice(&[0x04, 0, 0, 0, 0, 0, 0, 0]); // narg_byte_len = 4
        bytes.extend_from_slice(&[0x03, 0, 0, 0, 0, 0, 0, 0]); // hint_byte_len = 3
        assert_eq!(bytes.len(), HEADER_LEN, "the fields above");
        bytes.extend_from_slice(b"NNNN");
        bytes.extend_from_slice(b"HHH");

        let both = proof(b"NNNN", b"HHH");
        assert_eq!(encode(&both), bytes, "the narg stream comes first");
        assert_eq!(decode(&bytes), Ok(both.clone()));
        assert_eq!(WireProof::new(&both).byte_len(), bytes.len());

        assert_eq!(encode(&one_record()).len(), HEADER_LEN + 16);
        assert_eq!(decode(&encode(&one_record())), Ok(one_record()));

        let empty = proof(&[], &[]);
        assert_eq!(encode(&empty).len(), HEADER_LEN);
        assert_eq!(decode(&encode(&empty)), Ok(empty));
    }

    /// Below the header a prefix runs out inside a fixed field; from the
    /// header up the fields parse and the declared lengths outrun the input.
    #[test]
    fn nothing_but_the_exact_encoding_is_accepted() {
        let bytes = encode(&proof(&[1, 2, 3, 4], &[5, 6]));
        assert_eq!(bytes.len(), HEADER_LEN + 6);

        for len in 0..bytes.len() {
            let error = decode(&bytes[..len]).expect_err("a prefix is not a container");
            if len < HEADER_LEN {
                assert_matches!(error, ProofDecodingError::OutOfInput(_), "prefix of {len}");
            } else {
                assert_eq!(
                    error,
                    ProofDecodingError::LengthMismatch {
                        declared: bytes.len(),
                        actual: len,
                    },
                    "prefix of {len}"
                );
            }
        }

        // Trailing zeros are what a lenient decoder is likeliest to ignore.
        for suffix in [vec![0u8], vec![0u8; 64], vec![0xff; 3]] {
            let mut padded = bytes.clone();
            padded.extend_from_slice(&suffix);
            assert_eq!(
                decode(&padded),
                Err(ProofDecodingError::LengthMismatch {
                    declared: bytes.len(),
                    actual: padded.len(),
                })
            );
        }

        assert!(decode(&bytes).is_ok(), "and the exact encoding is accepted");
    }

    #[test]
    fn the_magic_is_checked_before_anything_else() {
        let mut bytes = encode(&one_record());
        bytes[offset::MAGIC] ^= 1;
        assert_eq!(decode(&bytes), Err(ProofDecodingError::BadMagic));

        // A header wrong in several ways at once still reports the magic: the
        // order of the checks is part of the format.
        let mut bytes = encode(&one_record());
        bytes[offset::MAGIC + 7] = 0xff;
        bytes[offset::WIRE_VERSION] = 9;
        bytes[offset::FLAGS] = 1;
        assert_eq!(decode(&bytes), Err(ProofDecodingError::BadMagic));
    }

    #[test]
    fn an_unknown_version_is_refused() {
        for version in [0u16, 2, 0x0100, u16::MAX] {
            let mut bytes = encode(&one_record());
            bytes[offset::WIRE_VERSION..offset::WIRE_VERSION + 2]
                .copy_from_slice(&version.to_le_bytes());
            assert_eq!(
                decode(&bytes),
                Err(ProofDecodingError::UnsupportedVersion(version))
            );
        }
    }

    #[test]
    fn a_header_length_other_than_thirty_two_is_refused() {
        for header_len in [0u16, 31, 33, 40, u16::MAX] {
            let mut bytes = encode(&one_record());
            bytes[offset::HEADER_LEN..offset::HEADER_LEN + 2]
                .copy_from_slice(&header_len.to_le_bytes());
            assert_eq!(
                decode(&bytes),
                Err(ProofDecodingError::BadHeaderLen(header_len as usize))
            );
        }
    }

    #[test]
    fn every_reserved_flag_bit_is_refused() {
        for bit in 0..u32::BITS {
            let flags = 1u32 << bit;
            let mut bytes = encode(&one_record());
            bytes[offset::FLAGS..offset::FLAGS + 4].copy_from_slice(&flags.to_le_bytes());
            assert_eq!(
                decode(&bytes),
                Err(ProofDecodingError::ReservedFlagsSet(flags))
            );
        }
    }

    #[test]
    fn a_declared_length_that_does_not_match_the_input_is_refused() {
        // Each field is probed away from its one admissible value.
        for (name, offset, truth) in [
            ("narg", offset::NARG_BYTE_LEN, 16u64),
            ("hint", offset::HINT_BYTE_LEN, 0u64),
        ] {
            for declared in [0u64, 15, 16, 17, u64::from(u32::MAX)] {
                if declared == truth {
                    continue;
                }
                let mut bytes = encode(&one_record());
                bytes[offset..offset + 8].copy_from_slice(&declared.to_le_bytes());
                assert_matches!(
                    decode(&bytes),
                    Err(ProofDecodingError::LengthMismatch { .. }),
                    "{name} length {declared}",
                );
            }
        }
    }

    #[test]
    fn declared_lengths_that_overflow_are_refused_rather_than_wrapping() {
        // `32 + u64::MAX` wraps to 31 in unchecked arithmetic, which would
        // then be compared against a 48-byte input.
        let mut bytes = encode(&one_record());
        bytes[offset::NARG_BYTE_LEN..offset::NARG_BYTE_LEN + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(decode(&bytes), Err(ProofDecodingError::LengthOverflow));

        // And the pair overflowing only once added together.
        let mut bytes = encode(&one_record());
        let half = u64::MAX / 2 + 1;
        bytes[offset::NARG_BYTE_LEN..offset::NARG_BYTE_LEN + 8]
            .copy_from_slice(&half.to_le_bytes());
        bytes[offset::HINT_BYTE_LEN..offset::HINT_BYTE_LEN + 8]
            .copy_from_slice(&half.to_le_bytes());
        assert_eq!(decode(&bytes), Err(ProofDecodingError::LengthOverflow));
    }

    /// The complement of the header tests: this is a frame, not an integrity
    /// check. Every byte past the header is opaque to it, so a flip there has
    /// to decode -- to a different proof that re-encodes to the bytes actually
    /// supplied. Catching it is the sponge's job, which `tests/host.rs` pins.
    #[test]
    fn a_flip_anywhere_in_the_payload_is_carried_not_caught() {
        let honest = proof(&[1, 2, 3, 4], &[5, 6]);
        let bytes = encode(&honest);

        for offset in HEADER_LEN..bytes.len() {
            let mut tampered = bytes.clone();
            tampered[offset] ^= 0x80;

            let decoded = decode(&tampered).expect("a payload is opaque to the frame");
            assert_ne!(decoded, honest, "a flip at {offset} reaches the proof");
            assert_eq!(encode(&decoded), tampered, "and keeps its one spelling");
        }
    }
}
