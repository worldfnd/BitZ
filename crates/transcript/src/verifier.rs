use field::F128;
use spongefish::{Decoding, Encoding, NargDeserialize, VerificationError, VerificationResult};

use crate::{PublicTranscript, bytes::ProverMessageBytes};

/// The verifier half of the transcript.
///
/// Replays the narg string through the sponge and drains the hint stream
/// front to back.
pub struct VerifierState<'a> {
    pub(crate) inner: spongefish::VerifierState<'a>,
    pub(crate) hints: &'a [u8],
}

impl PublicTranscript for VerifierState<'_> {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public_message(message);
    }

    fn verifier_message_f128(&mut self) -> F128 {
        self.inner.verifier_message()
    }
}

impl VerifierState<'_> {
    /// Absorbs a message both parties already know; nothing is read.
    pub fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public_message(message);
    }

    /// Reads the next prover message from the narg string and absorbs its
    /// canonical re-encoding.
    pub fn prover_message<T: Encoding<[u8]> + NargDeserialize>(&mut self) -> VerificationResult<T> {
        let message = self.inner.prover_message()?;
        Ok(message)
    }

    /// Reads and absorbs one bounded, length-prefixed byte string.
    pub fn prover_message_bytes<const MAX_LEN: usize>(&mut self) -> VerificationResult<Vec<u8>> {
        self.inner
            .prover_message::<ProverMessageBytes<MAX_LEN>>()
            .map(ProverMessageBytes::into_bytes)
    }

    /// Squeezes a challenge.
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> T {
        self.inner.verifier_message()
    }

    /// Reads the next value from the hint stream. The sponge is untouched.
    pub fn hint<T: NargDeserialize>(&mut self) -> VerificationResult<T> {
        let hint = T::deserialize_from_narg(&mut self.hints)?;
        Ok(hint)
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
