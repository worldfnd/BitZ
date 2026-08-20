use field::F128 as LocalF128;
use transcript::{Encoding, NargDeserialize, ProverState, VerifierState};

use crate::CommitError;

pub(crate) trait PublicTranscript {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T);

    fn verifier_message_f128(&mut self) -> LocalF128;
}

impl PublicTranscript for ProverState {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        ProverState::public_message(self, message);
    }

    fn verifier_message_f128(&mut self) -> LocalF128 {
        ProverState::verifier_message(self)
    }
}

impl PublicTranscript for VerifierState<'_> {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        VerifierState::public_message(self, message);
    }

    fn verifier_message_f128(&mut self) -> LocalF128 {
        VerifierState::verifier_message(self)
    }
}

pub(crate) trait PcsTranscript: PublicTranscript {
    /// Writes `expected`, or reads and verifies the next prover message.
    fn bind_prover_message<T>(&mut self, expected: &T) -> Result<(), CommitError>
    where
        T: Encoding<[u8]> + NargDeserialize + PartialEq;
}

impl PcsTranscript for ProverState {
    fn bind_prover_message<T>(&mut self, expected: &T) -> Result<(), CommitError>
    where
        T: Encoding<[u8]> + NargDeserialize + PartialEq,
    {
        ProverState::prover_message(self, expected);
        Ok(())
    }
}

impl PcsTranscript for VerifierState<'_> {
    fn bind_prover_message<T>(&mut self, expected: &T) -> Result<(), CommitError>
    where
        T: Encoding<[u8]> + NargDeserialize + PartialEq,
    {
        let observed =
            VerifierState::prover_message::<T>(self).map_err(|_| CommitError::MalformedProof)?;
        if observed != *expected {
            return Err(CommitError::MalformedProof);
        }
        Ok(())
    }
}
