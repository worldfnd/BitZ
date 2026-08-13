//! Flock challenger adapters over the project transcript.

use field::F128 as LocalF128;
use flock_core::challenger::Challenger;
use flock_core::field::F128 as BackendF128;
use transcript::{ProverState, VerifierState};

const VECTOR_SQUEEZE_TAG: &[u8] = b"pcs/flock/sample-vector/v1";
const POW_TAG: &[u8] = b"pcs/flock/pow/v1";

pub(crate) struct ProverChallenger<'a> {
    transcript: &'a mut ProverState,
}

impl<'a> ProverChallenger<'a> {
    pub(crate) fn new(transcript: &'a mut ProverState) -> Self {
        Self { transcript }
    }
}

pub(crate) struct VerifierChallenger<'a, 'proof> {
    transcript: &'a mut VerifierState<'proof>,
    failed: bool,
}

impl<'a, 'proof> VerifierChallenger<'a, 'proof> {
    pub(crate) fn new(transcript: &'a mut VerifierState<'proof>) -> Self {
        Self {
            transcript,
            failed: false,
        }
    }

    pub(crate) fn failed(&self) -> bool {
        self.failed
    }

    fn read<T>(&mut self) -> Option<T>
    where
        T: transcript::Encoding<[u8]> + transcript::NargDeserialize,
    {
        match self.transcript.prover_message() {
            Ok(value) => Some(value),
            Err(_) => {
                self.failed = true;
                None
            }
        }
    }
}

#[inline]
fn local(value: BackendF128) -> LocalF128 {
    LocalF128::new(value.lo, value.hi)
}

#[inline]
fn backend(value: LocalF128) -> BackendF128 {
    BackendF128::new(value.lo, value.hi)
}

impl Challenger for ProverChallenger<'_> {
    fn observe_label(&mut self, label: &[u8]) {
        self.transcript.public_message(label);
    }

    fn observe_f128(&mut self, value: BackendF128) {
        self.transcript.prover_message(&local(value));
    }

    fn observe_f128_slice(&mut self, values: &[BackendF128]) {
        self.transcript.prover_message(
            &u32::try_from(values.len()).expect("observed field slice exceeds u32"),
        );
        for &value in values {
            self.observe_f128(value);
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.transcript
            .prover_message(&u32::try_from(bytes.len()).expect("observed byte slice exceeds u32"));
        for byte in bytes {
            self.transcript.prover_message(&[*byte]);
        }
    }

    fn sample_f128(&mut self) -> BackendF128 {
        backend(self.transcript.verifier_message::<LocalF128>())
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<BackendF128> {
        self.transcript.public_message(VECTOR_SQUEEZE_TAG);
        self.transcript.public_message(&(n as u64));
        (0..n).map(|_| self.sample_f128()).collect()
    }

    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.transcript.public_message(POW_TAG);
        self.transcript.public_message(&bits);
        let seed = self.transcript.verifier_message::<LocalF128>().to_bytes();
        let nonce = find_pow(&seed, bits);
        self.transcript.prover_message(&nonce.to_le_bytes());
        nonce
    }

    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        unreachable!("the prover challenger cannot verify proof of work")
    }
}

impl Challenger for VerifierChallenger<'_, '_> {
    fn observe_label(&mut self, label: &[u8]) {
        self.transcript.public_message(label);
    }

    fn observe_f128(&mut self, value: BackendF128) {
        if self.read::<LocalF128>() != Some(local(value)) {
            self.failed = true;
        }
    }

    fn observe_f128_slice(&mut self, values: &[BackendF128]) {
        if self.read::<u32>() != Some(values.len() as u32) {
            self.failed = true;
        }
        for &value in values {
            self.observe_f128(value);
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        if self.read::<u32>() != Some(bytes.len() as u32) {
            self.failed = true;
        }
        for &byte in bytes {
            if self.read::<[u8; 1]>() != Some([byte]) {
                self.failed = true;
            }
        }
    }

    fn sample_f128(&mut self) -> BackendF128 {
        backend(self.transcript.verifier_message::<LocalF128>())
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<BackendF128> {
        self.transcript.public_message(VECTOR_SQUEEZE_TAG);
        self.transcript.public_message(&(n as u64));
        (0..n).map(|_| self.sample_f128()).collect()
    }

    fn grind_pow(&mut self, _bits: u32) -> u64 {
        unreachable!("the verifier challenger cannot grind proof of work")
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.transcript.public_message(POW_TAG);
        self.transcript.public_message(&bits);
        let seed = self.transcript.verifier_message::<LocalF128>().to_bytes();
        let encoded = self.read::<[u8; 8]>().map(u64::from_le_bytes);
        let matches_stream = encoded == Some(nonce);
        let valid = pow_valid(&seed, nonce, bits);
        if !matches_stream || !valid {
            self.failed = true;
        }
        matches_stream && valid
    }
}

fn find_pow(seed: &[u8; 16], bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    let mut nonce = 0u64;
    loop {
        if pow_valid(seed, nonce, bits) {
            return nonce;
        }
        nonce = nonce.checked_add(1).expect("proof-of-work nonce exhausted");
    }
}

fn pow_valid(seed: &[u8; 16], nonce: u64, bits: u32) -> bool {
    if bits == 0 {
        return nonce == 0;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"f2z-pcs-pow-v1");
    hasher.update(seed);
    hasher.update(&nonce.to_le_bytes());
    let digest = hasher.finalize();
    leading_zero_bits(digest.as_bytes()) >= bits
}

fn leading_zero_bits(bytes: &[u8]) -> u32 {
    let mut total = 0;
    for byte in bytes {
        let zeros = byte.leading_zeros();
        total += zeros;
        if zeros != 8 {
            break;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_bit_pow_has_one_canonical_nonce() {
        assert!(pow_valid(&[0; 16], 0, 0));
        assert!(!pow_valid(&[0; 16], 1, 0));
    }

    #[test]
    fn positive_pow_rejects_a_changed_nonce() {
        let seed = [7; 16];
        let nonce = find_pow(&seed, 8);
        assert!(pow_valid(&seed, nonce, 8));
        let mut changed = nonce.wrapping_add(1);
        while pow_valid(&seed, changed, 8) {
            changed = changed.wrapping_add(1);
        }
        assert!(!pow_valid(&seed, changed, 8));
    }
}
