use spongefish::{Decoding, Encoding, NargSerialize};

use crate::Proof;

/// The prover half of the transcript.
///
/// Wraps spongefish's prover state and carries the hint stream, which the
/// sponge never sees.
pub struct ProverState {
    pub(crate) inner: spongefish::ProverState,
    pub(crate) hints: Vec<u8>,
    pub(crate) narg_records: u32,
    pub(crate) hint_records: u32,
}

impl ProverState {
    /// Absorbs a message both parties already know; nothing is written.
    pub fn public_message<T: Encoding<[u8]> + ?Sized>(&mut self, message: &T) {
        self.inner.public_message(message);
    }

    /// Absorbs a message and writes it to the narg string.
    pub fn prover_message<T: Encoding<[u8]> + NargSerialize + ?Sized>(&mut self, message: &T) {
        self.inner.prover_message(message);
        self.narg_records = self
            .narg_records
            .checked_add(1)
            .expect("reaching `u32::MAX` records should not be physically possible");
    }

    /// Squeezes a challenge.
    pub fn verifier_message<T: Decoding<[u8]>>(&mut self) -> T {
        self.inner.verifier_message()
    }

    /// Writes a value to the hint stream. The sponge is untouched, so hints
    /// cannot influence any challenge.
    pub fn hint<T: NargSerialize + ?Sized>(&mut self, hint: &T) {
        hint.serialize_into_narg(&mut self.hints);
        self.hint_records = self
            .hint_records
            .checked_add(1)
            .expect("reaching `u32::MAX` records should not be physically possible");
    }

    /// How many records went to the narg string and to the hint stream.
    pub fn records(&self) -> (u32, u32) {
        (self.narg_records, self.hint_records)
    }

    pub fn finish(self) -> Proof {
        Proof {
            narg_string: self.inner.narg_string().to_vec(),
            hints: self.hints,
            narg_records: self.narg_records,
            hint_records: self.hint_records,
        }
    }
}
