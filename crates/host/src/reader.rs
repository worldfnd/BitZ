use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("Out of input")]
pub struct OutOfInput;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("Leftover input after finishing the stream")]
pub struct LeftoverInput;

/// A bounded reader over input bytes.
#[derive(Debug, Clone, Copy)]
pub struct BoundedReader<'a> {
    /// How many bytes have been read.
    pub pos: usize,
    /// Input bytes that has not been read yet.
    rest: &'a [u8],
}

impl<'a> BoundedReader<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            pos: 0,
            rest: bytes,
        }
    }

    /// The next `N` bytes.
    pub fn read_array<const N: usize>(&mut self) -> Result<[u8; N], OutOfInput> {
        let (head, tail) = self.rest.split_at_checked(N).ok_or(OutOfInput)?;
        self.rest = tail;
        self.pos = self.pos.saturating_add(N);
        head.try_into().map_err(|_| OutOfInput)
    }

    /// The next `n` bytes, borrowed rather than copied.
    pub fn read_slice(&mut self, n: usize) -> Result<&'a [u8], OutOfInput> {
        let (head, tail) = self.rest.split_at_checked(n).ok_or(OutOfInput)?;
        self.rest = tail;
        self.pos = self.pos.saturating_add(n);
        Ok(head)
    }

    pub fn read_u16(&mut self) -> Result<u16, OutOfInput> {
        self.read_array().map(u16::from_le_bytes)
    }

    pub fn read_u32(&mut self) -> Result<u32, OutOfInput> {
        self.read_array().map(u32::from_le_bytes)
    }

    pub fn read_u64(&mut self) -> Result<u64, OutOfInput> {
        self.read_array().map(u64::from_le_bytes)
    }

    /// Consumes the reader, failing unless the input was read to its end.
    pub fn finish(self) -> Result<(), LeftoverInput> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(LeftoverInput)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_advance_the_position_and_stop_at_the_end() {
        let bytes = [1u8, 0, 0, 0, 7, 8];
        let mut reader = BoundedReader::new(&bytes);

        assert_eq!(reader.pos, 0);
        assert_eq!(reader.read_u32(), Ok(1), "little endian");
        assert_eq!(reader.pos, 4);
        assert_eq!(reader.read_slice(2), Ok(&[7u8, 8][..]));
        assert_eq!(reader.finish(), Ok(()));
        assert_eq!(reader.read_u32(), Err(OutOfInput));
    }

    #[test]
    fn a_failed_read_consumes_nothing() {
        let short = [1u8, 2, 3];
        let mut reader = BoundedReader::new(&short);
        assert_eq!(reader.read_u32(), Err(OutOfInput));
        assert_eq!(reader.pos, 0);
        assert_eq!(
            reader.read_u16(),
            Ok(0x0201),
            "so the next read still works"
        );

        // A container's declared stream length reaches `read_slice` verbatim,
        // and `pos + n` overflows here: an unchecked split could wrap to a
        // small end and hand back a slice that was never there.
        let wide = [0u8; 8];
        let mut reader = BoundedReader::new(&wide);
        assert_eq!(reader.read_u32(), Ok(0));
        assert_eq!(reader.read_slice(usize::MAX), Err(OutOfInput));
        assert_eq!(reader.pos, 4);

        let mut empty = BoundedReader::new(&[]);
        assert_eq!(empty.read_array::<1>(), Err(OutOfInput));
        assert_eq!(empty.read_slice(0), Ok(&[][..]), "but nothing always fits");
    }

    #[test]
    fn finishing_requires_the_input_to_be_exhausted() {
        let bytes = [0u8; 8];

        let mut reader = BoundedReader::new(&bytes);
        assert_eq!(reader.read_slice(8), Ok(&bytes[..]));
        assert_eq!(reader.finish(), Ok(()));
        assert_eq!(
            BoundedReader::new(&[]).finish(),
            Ok(()),
            "nothing to exhaust"
        );
        assert_eq!(BoundedReader::new(&bytes).finish(), Err(LeftoverInput));

        let mut reader = BoundedReader::new(&bytes);
        assert_eq!(reader.read_u32(), Ok(0));
        assert_eq!(
            reader.finish(),
            Err(LeftoverInput),
            "four of eight bytes read: position and remainder are equal here, \
             which a predicate comparing them would wave through"
        );
    }
}
