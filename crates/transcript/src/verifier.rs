use spongefish::{Decoding, Encoding, NargDeserialize, VerificationError, VerificationResult};

/// The verifier half of the transcript.
///
/// Replays the narg string through the sponge and drains the hint stream
/// front to back.
pub struct VerifierState<'a> {
    pub(crate) inner: spongefish::VerifierState<'a>,
    pub(crate) hints: &'a [u8],
    pub(crate) narg_records: u32,
    pub(crate) hint_records: u32,
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
        // Counted only on success
        self.narg_records = self
            .narg_records
            .checked_add(1)
            .expect("reaching `u32::MAX` records should not be physically possible");
        Ok(message)
    }

    /// Squeezes a challenge.
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> T {
        self.inner.verifier_message()
    }

    /// Reads the next value from the hint stream. The sponge is untouched.
    pub fn hint<T: NargDeserialize>(&mut self) -> VerificationResult<T> {
        let hint = T::deserialize_from_narg(&mut self.hints)?;
        self.hint_records = self
            .hint_records
            .checked_add(1)
            .expect("reaching `u32::MAX` records should not be physically possible");
        Ok(hint)
    }

    /// How many records the replay consumed from each stream.
    ///
    /// The host container declares both counts; this is the other side of
    /// that comparison, and it is meaningful only once the replay is over.
    pub fn records(&self) -> (u32, u32) {
        (self.narg_records, self.hint_records)
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
