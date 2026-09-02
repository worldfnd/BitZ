use transcript::{Encoding, NargDeserialize, ProverState, VerifierState};

pub(crate) use transcript::PublicTranscript;

use crate::CommitError;

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
        self.prover_message(expected);
        Ok(())
    }
}

impl PcsTranscript for VerifierState<'_> {
    fn bind_prover_message<T>(&mut self, expected: &T) -> Result<(), CommitError>
    where
        T: Encoding<[u8]> + NargDeserialize + PartialEq,
    {
        let observed = self
            .prover_message::<T>()
            .map_err(|_| CommitError::MalformedProof)?;
        if observed != *expected {
            return Err(CommitError::MalformedProof);
        }
        Ok(())
    }
}
