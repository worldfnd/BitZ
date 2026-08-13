use spongefish::{Decoding, Encoding, NargDeserialize, VerificationError, VerificationResult};

/// The verifier half of the transcript.
///
/// Replays the narg string through the sponge and drains the hint stream
/// front to back.
pub struct VerifierState<'a> {
    pub(crate) inner: spongefish::VerifierState<'a>,
    pub(crate) hints: &'a [u8],
}

impl VerifierState<'_> {
    /// Absorbs a message both parties already know; nothing is read.
    pub fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public_message(message);
    }

    /// Reads the next prover message from the narg string and absorbs its
    /// canonical re-encoding.
    pub fn prover_message<T: Encoding<[u8]> + NargDeserialize>(&mut self) -> VerificationResult<T> {
        self.inner.prover_message()
    }

    /// Squeezes a challenge.
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> T {
        self.inner.verifier_message()
    }

    /// Reads the next value from the hint stream. The sponge is untouched.
    pub fn hint<T: NargDeserialize>(&mut self) -> VerificationResult<T> {
        T::deserialize_from_narg(&mut self.hints)
    }

    /// Reads one bounded, length-prefixed byte string from the hint stream.
    pub fn hint_bytes(&mut self, max_len: usize) -> VerificationResult<Vec<u8>> {
        let mut rest = self.hints;
        let len = u32::deserialize_from_narg(&mut rest)? as usize;
        if len > max_len || rest.len() < len {
            return Err(VerificationError);
        }
        let bytes = rest[..len].to_vec();
        self.hints = &rest[len..];
        Ok(bytes)
    }

    /// Fails unless both the narg string and the hint stream were consumed
    /// exactly.
    pub fn check_eof(self) -> VerificationResult<()> {
        self.inner.check_eof()?;
        if self.hints.is_empty() {
            Ok(())
        } else {
            Err(VerificationError)
        }
    }
}
