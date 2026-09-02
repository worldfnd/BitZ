use spongefish::{Encoding, NargDeserialize, VerificationError, VerificationResult};

pub(crate) struct ProverMessageBytes<const MAX_LEN: usize>(Vec<u8>);

impl<const MAX_LEN: usize> ProverMessageBytes<MAX_LEN> {
    pub(crate) fn new(bytes: &[u8]) -> Self {
        let len = u32::try_from(bytes.len()).expect("prover message byte string exceeds u32");
        let mut encoded = Vec::with_capacity(size_of::<u32>() + bytes.len());
        encoded.extend_from_slice(&len.to_le_bytes());
        encoded.extend_from_slice(bytes);
        Self(encoded)
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.0.drain(..size_of::<u32>());
        self.0
    }
}

impl<const MAX_LEN: usize> Encoding<[u8]> for ProverMessageBytes<MAX_LEN> {
    fn encode(&self) -> impl AsRef<[u8]> {
        &self.0
    }
}

impl<const MAX_LEN: usize> NargDeserialize for ProverMessageBytes<MAX_LEN> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        let mut rest = *buf;
        let len = u32::deserialize_from_narg(&mut rest)? as usize;
        if len > MAX_LEN || rest.len() < len {
            return Err(VerificationError);
        }

        let message = Self::new(&rest[..len]);
        *buf = &rest[len..];
        Ok(message)
    }
}
