use spongefish::{Decoding, Encoding, NargSerialize};

use crate::{Proof, PublicTranscript};

/// The prover half of the transcript.
///
/// Wraps spongefish's prover state and carries the hint stream, which the
/// sponge never sees.
pub struct ProverState {
    pub(crate) inner: spongefish::ProverState,
    pub(crate) hints: Vec<u8>,
}

impl PublicTranscript for ProverState {
    fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public_message(message);
    }
}

impl ProverState {
    /// Absorbs a message both parties already know; nothing is written.
    pub fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public_message(message);
    }

    /// Absorbs a message and writes it to the narg string.
    pub fn prover_message<T: Encoding<[u8]> + NargSerialize + ?Sized>(&mut self, message: &T) {
        self.inner.prover_message(message);
    }

    /// Squeezes a challenge.
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> T {
        self.inner.verifier_message()
    }

    /// Writes a value to the hint stream. The sponge is untouched, so hints
    /// cannot influence any challenge.
    pub fn hint<T: NargSerialize + ?Sized>(&mut self, hint: &T) {
        hint.serialize_into_narg(&mut self.hints);
    }

    /// Writes one length-prefixed byte string to the hint stream.
    pub fn hint_bytes(&mut self, bytes: &[u8]) {
        u32::try_from(bytes.len())
            .expect("hint byte string exceeds u32")
            .serialize_into_narg(&mut self.hints);
        self.hints.extend_from_slice(bytes);
    }

    pub fn finish(self) -> Proof {
        Proof {
            narg_string: self.inner.narg_string().to_vec(),
            hints: self.hints,
        }
    }
}
