//! Proof host container: how a proof leaves and enters the host.
//!
//! The header is 40 bytes, then the narg stream, then the hint stream:
//!
//! ```text
//! offset size field
//! 0      8    magic = 46 32 5a 50 43 53 00 00   ("F2ZPCS\0\0")
//! 8      2    wire_version = 1
//! 10     2    header_len = 40
//! 12     4    flags = 0
//! 16     4    narg_record_count
//! 20     4    hint_record_count
//! 24     8    narg_byte_len
//! 32     8    hint_byte_len
//! 40     ..   narg bytes, then hint bytes
//! ```

use crate::reader::{self, BoundedReader};
use thiserror::Error;
use transcript::Proof;

/// The eight-byte proof container magic.
pub const MAGIC: [u8; 8] = *b"F2ZPCS\0\0";
/// The only wire version this codec reads or writes.
pub const WIRE_VERSION: u16 = 1;
/// The fixed header length, in bytes.
pub const HEADER_LEN: usize = 40;
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
    /// How many records the writer says `narg` holds.
    ///
    /// Declared, not verified. Checking it against the records a verifier
    /// actually consumed needs the ledger.
    pub narg_records: u32,
    /// How many records the writer says `hints` holds, on the same terms.
    pub hint_records: u32,
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
            narg_records: proof.narg_records,
            hint_records: proof.hint_records,
        }
    }

    /// Copies the streams out into an owned proof.
    #[inline]
    pub fn unwrap(&self) -> Proof {
        Proof {
            narg_string: self.narg_string.to_vec(),
            hints: self.hints.to_vec(),
            narg_records: self.narg_records,
            hint_records: self.hint_records,
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
        out.extend_from_slice(&self.narg_records.to_le_bytes());
        out.extend_from_slice(&self.hint_records.to_le_bytes());
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

        let narg_records = reader.read_u32()?;
        let hint_records = reader.read_u32()?;
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

        Ok(Self {
            narg_string,
            hints,
            narg_records,
            hint_records,
        })
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

    /// The normative known-answer set from `docs/f2z-pcs-spec/wire-v1-vectors.txt`,
    /// named in section 6.10: one narg record holding `Vec<K>([1])`, no hints.
    const KAT_HEADER: &str =
        "46325a50435300000100280000000000010000000000000014000000000000000000000000000000";
    const KAT_NARG_RECORD: &str = "0100000001000000000000000000000000000000";
    const KAT_HOST: &str = "46325a504353000001002800000000000100000000000000140000000000000000000000000000000100000001000000000000000000000000000000";

    /// Byte offsets of the header fields, in order.
    pub mod offset {
        pub const MAGIC: usize = 0;
        pub const WIRE_VERSION: usize = 8;
        pub const HEADER_LEN: usize = 10;
        pub const FLAGS: usize = 12;
        pub const NARG_RECORD_COUNT: usize = 16;
        pub const HINT_RECORD_COUNT: usize = 20;
        pub const NARG_BYTE_LEN: usize = 24;
        pub const HINT_BYTE_LEN: usize = 32;
    }

    fn from_hex(hex: &str) -> Vec<u8> {
        assert!(hex.len().is_multiple_of(2), "hex must be whole bytes");
        (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex digits"))
            .collect()
    }

    fn kat_proof() -> Proof {
        Proof {
            narg_string: from_hex(KAT_NARG_RECORD),
            hints: Vec::new(),
            narg_records: 1,
            hint_records: 0,
        }
    }

    fn proof(narg: &[u8], hints: &[u8], records: (u32, u32)) -> Proof {
        Proof {
            narg_string: narg.to_vec(),
            hints: hints.to_vec(),
            narg_records: records.0,
            hint_records: records.1,
        }
    }

    /// The positive case, in three parts: the vector the spec publishes, the
    /// same grammar written out by hand, and the empty proof.
    ///
    /// The hand-written part is what makes this more than a self-consistency
    /// check. A field written and read at the same wrong offset survives any
    /// number of `encode`/`decode` round trips, so the bytes have to come from
    /// somewhere other than `encode`; the two streams differ in length and in
    /// content so that reading them in the wrong order is visible.
    #[test]
    fn the_encoding_is_the_one_the_spec_publishes() {
        let host = from_hex(KAT_HOST);
        assert_eq!(host.len(), 60, "40 header bytes and one 20-byte record");
        assert_eq!(from_hex(KAT_HEADER), host[..HEADER_LEN]);
        assert_eq!(from_hex(KAT_NARG_RECORD), host[HEADER_LEN..]);
        assert_eq!(encode(&kat_proof()), host);
        assert_eq!(decode(&host), Ok(kat_proof()));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"F2ZPCS\0\0");
        bytes.extend_from_slice(&[0x01, 0x00]); // wire_version = 1
        bytes.extend_from_slice(&[0x28, 0x00]); // header_len = 40
        bytes.extend_from_slice(&[0x00; 4]); // flags = 0
        bytes.extend_from_slice(&[0x02, 0x00, 0x00, 0x00]); // narg_record_count = 2
        bytes.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // hint_record_count = 1
        bytes.extend_from_slice(&[0x04, 0, 0, 0, 0, 0, 0, 0]); // narg_byte_len = 4
        bytes.extend_from_slice(&[0x03, 0, 0, 0, 0, 0, 0, 0]); // hint_byte_len = 3
        assert_eq!(bytes.len(), HEADER_LEN, "the eight fields above");
        bytes.extend_from_slice(b"NNNN");
        bytes.extend_from_slice(b"HHH");

        let both = proof(b"NNNN", b"HHH", (2, 1));
        assert_eq!(encode(&both), bytes, "the narg stream comes first");
        assert_eq!(decode(&bytes), Ok(both.clone()));
        assert_eq!(WireProof::new(&both).byte_len(), bytes.len());

        let empty = proof(&[], &[], (0, 0));
        assert_eq!(encode(&empty).len(), HEADER_LEN);
        assert_eq!(decode(&encode(&empty)), Ok(empty));
    }

    /// Below the header a prefix runs out inside a fixed field; from the
    /// header up the fields parse and the declared lengths outrun the input.
    #[test]
    fn nothing_but_the_exact_encoding_is_accepted() {
        let bytes = encode(&proof(&[1, 2, 3, 4], &[5, 6], (1, 1)));
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
        let mut bytes = encode(&kat_proof());
        bytes[offset::MAGIC] ^= 1;
        assert_eq!(decode(&bytes), Err(ProofDecodingError::BadMagic));

        // A header wrong in several ways at once still reports the magic: the
        // order of the checks is part of the format.
        let mut bytes = encode(&kat_proof());
        bytes[offset::MAGIC + 7] = 0xff;
        bytes[offset::WIRE_VERSION] = 9;
        bytes[offset::FLAGS] = 1;
        assert_eq!(decode(&bytes), Err(ProofDecodingError::BadMagic));
    }

    #[test]
    fn an_unknown_version_is_refused() {
        for version in [0u16, 2, 0x0100, u16::MAX] {
            let mut bytes = encode(&kat_proof());
            bytes[offset::WIRE_VERSION..offset::WIRE_VERSION + 2]
                .copy_from_slice(&version.to_le_bytes());
            assert_eq!(
                decode(&bytes),
                Err(ProofDecodingError::UnsupportedVersion(version))
            );
        }
    }

    #[test]
    fn a_header_length_other_than_forty_is_refused() {
        for header_len in [0u16, 39, 41, u16::MAX] {
            let mut bytes = encode(&kat_proof());
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
            let mut bytes = encode(&kat_proof());
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
            ("narg", offset::NARG_BYTE_LEN, 20u64),
            ("hint", offset::HINT_BYTE_LEN, 0u64),
        ] {
            for declared in [0u64, 19, 20, 21, u64::from(u32::MAX)] {
                if declared == truth {
                    continue;
                }
                let mut bytes = encode(&kat_proof());
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
        // `40 + u64::MAX` wraps to 39 in unchecked arithmetic, which would
        // then be compared against a 60-byte input.
        let mut bytes = encode(&kat_proof());
        bytes[offset::NARG_BYTE_LEN..offset::NARG_BYTE_LEN + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(decode(&bytes), Err(ProofDecodingError::LengthOverflow));

        // And the pair overflowing only once added together.
        let mut bytes = encode(&kat_proof());
        let half = u64::MAX / 2 + 1;
        bytes[offset::NARG_BYTE_LEN..offset::NARG_BYTE_LEN + 8]
            .copy_from_slice(&half.to_le_bytes());
        bytes[offset::HINT_BYTE_LEN..offset::HINT_BYTE_LEN + 8]
            .copy_from_slice(&half.to_le_bytes());
        assert_eq!(decode(&bytes), Err(ProofDecodingError::LengthOverflow));
    }

    /// The two header fields this codec carries without verifying, exactly as
    /// [`WireProof::narg_records`] says: records are not self-delimiting, so
    /// only a replay knows how many were consumed. Rewriting one must decode,
    /// to a proof whose declared count differs and whose streams do not.
    #[test]
    fn the_record_counts_are_carried_not_checked() {
        let honest = proof(&[1, 2, 3, 4], &[5, 6], (1, 1));
        let bytes = encode(&honest);

        for (offset, count) in [
            (offset::NARG_RECORD_COUNT, 7u32),
            (offset::HINT_RECORD_COUNT, u32::MAX),
        ] {
            let mut tampered = bytes.clone();
            tampered[offset..offset + 4].copy_from_slice(&count.to_le_bytes());

            let decoded = decode(&tampered).expect("a count is transport metadata");
            assert_ne!(decoded, honest, "the rewritten count reaches the proof");
            assert_eq!(decoded.narg_string, honest.narg_string, "and nothing else");
            assert_eq!(decoded.hints, honest.hints);
        }
    }

    /// The complement of the header tests: this is a frame, not an integrity
    /// check. Every byte past the header is opaque to it, so a flip there has
    /// to decode -- to a different proof that re-encodes to the bytes actually
    /// supplied. Catching it is the sponge's job, which `tests/host.rs` pins.
    #[test]
    fn a_flip_anywhere_in_the_payload_is_carried_not_caught() {
        let honest = proof(&[1, 2, 3, 4], &[5, 6], (1, 1));
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
