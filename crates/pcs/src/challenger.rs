//! Flock challenger adapters over the project transcript.

use crate::bridge::{as_flock_f128, from_flock_f128};
use field::F128 as LocalF128;
use flock_core::challenger::Challenger;
use flock_core::field::F128 as FlockF128;
use transcript::{ProverState, VerifierState};

const VECTOR_SQUEEZE_TAG: &[u8] = b"pcs/flock/sample-vector/v1";
const POW_TAG: &[u8] = b"pcs/flock/pow/v1";
const LIGERITO_BASIS_LABEL: &[u8] = b"flock-ligerito-basis-v0";

// F2Z binds beta under tag 0x5001 before Ligerito. Flock always observes its own
// entry label and target, so the adapter validates and suppresses that duplicate prefix.
#[derive(Clone, Copy)]
struct OpeningTargetPrefix {
    expected_ligerito_target: FlockF128,
    label_seen: bool,
}

pub(crate) struct ProverChallenger<'a> {
    transcript: &'a mut ProverState,
    failed: bool,
    opening_target: Option<OpeningTargetPrefix>,
}

impl<'a> ProverChallenger<'a> {
    #[cfg(test)]
    pub(crate) fn new(transcript: &'a mut ProverState) -> Self {
        Self {
            transcript,
            failed: false,
            opening_target: None,
        }
    }

    /// Creates the Ligerito adapter after PCS derives and binds `beta0` under tag 5001.
    /// The expected target lets the adapter check and omit Flock's duplicate entry pair.
    pub(crate) fn new_ligerito(
        transcript: &'a mut ProverState,
        expected_ligerito_target: FlockF128,
    ) -> Self {
        // Tag 5001 already binds the derived target. Suppress Flock's legacy entry pair.
        Self {
            transcript,
            failed: false,
            opening_target: Some(OpeningTargetPrefix {
                expected_ligerito_target,
                label_seen: false,
            }),
        }
    }

    pub(crate) fn failed(&self) -> bool {
        // A pending target means Flock returned before consuming its required entry prefix.
        self.failed || self.opening_target.is_some()
    }
}

pub(crate) struct VerifierChallenger<'a, 'proof> {
    transcript: &'a mut VerifierState<'proof>,
    failed: bool,
    opening_target: Option<OpeningTargetPrefix>,
}

impl<'a, 'proof> VerifierChallenger<'a, 'proof> {
    #[cfg(test)]
    pub(crate) fn new(transcript: &'a mut VerifierState<'proof>) -> Self {
        Self {
            transcript,
            failed: false,
            opening_target: None,
        }
    }

    /// Creates the verifier counterpart with the same derived `beta0` expectation.
    pub(crate) fn new_ligerito(
        transcript: &'a mut VerifierState<'proof>,
        expected_ligerito_target: FlockF128,
    ) -> Self {
        // Tag 5001 already binds the derived target. Suppress Flock's legacy entry pair.
        Self {
            transcript,
            failed: false,
            opening_target: Some(OpeningTargetPrefix {
                expected_ligerito_target,
                label_seen: false,
            }),
        }
    }

    pub(crate) fn failed(&self) -> bool {
        // A pending target means Flock returned before consuming its required entry prefix.
        self.failed || self.opening_target.is_some()
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

impl Challenger for ProverChallenger<'_> {
    fn observe_label(&mut self, label: &[u8]) {
        // After the prefix, the branch predictor consistently sees `None`.
        if let Some(prefix) = &mut self.opening_target
            && !prefix.label_seen
        {
            if label != LIGERITO_BASIS_LABEL {
                self.failed = true;
            }
            prefix.label_seen = true;
            return;
        }
        self.transcript.public_message(label);
    }

    fn observe_f128(&mut self, value: FlockF128) {
        // After the prefix, the branch predictor consistently sees `None`.
        if let Some(prefix) = self.opening_target.take() {
            if !prefix.label_seen || value != prefix.expected_ligerito_target {
                self.failed = true;
            }
            return;
        }
        self.transcript.prover_message(&from_flock_f128(value));
    }

    fn observe_f128_slice(&mut self, values: &[FlockF128]) {
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

    fn sample_f128(&mut self) -> FlockF128 {
        as_flock_f128(self.transcript.verifier_message::<LocalF128>())
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<FlockF128> {
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
        if let Some(prefix) = &mut self.opening_target
            && !prefix.label_seen
        {
            if label != LIGERITO_BASIS_LABEL {
                self.failed = true;
            }
            prefix.label_seen = true;
            return;
        }
        self.transcript.public_message(label);
    }

    fn observe_f128(&mut self, value: FlockF128) {
        if let Some(prefix) = self.opening_target.take() {
            if !prefix.label_seen || value != prefix.expected_ligerito_target {
                self.failed = true;
            }
            return;
        }
        if self.read::<LocalF128>() != Some(from_flock_f128(value)) {
            self.failed = true;
        }
    }

    fn observe_f128_slice(&mut self, values: &[FlockF128]) {
        let len = u32::try_from(values.len()).expect("observed field slice exceeds u32");
        if self.read::<u32>() != Some(len) {
            self.failed = true;
        }
        for &value in values {
            self.observe_f128(value);
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        let len = u32::try_from(bytes.len()).expect("observed byte slice exceeds u32");
        if self.read::<u32>() != Some(len) {
            self.failed = true;
        }
        for &byte in bytes {
            if self.read::<[u8; 1]>() != Some([byte]) {
                self.failed = true;
            }
        }
    }

    fn sample_f128(&mut self) -> FlockF128 {
        as_flock_f128(self.transcript.verifier_message::<LocalF128>())
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<FlockF128> {
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

/// todo: parallel pow? use potentially spongefish?
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
    use proptest::prelude::*;
    use transcript::{build_prover, build_verifier};

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

    #[test]
    fn observed_root_changes_the_following_challenge() {
        fn sample_after_root(root: &[u8; 32]) -> FlockF128 {
            let mut transcript = build_prover(b"pcs-challenger-test", b"root-binding");
            let mut challenger = ProverChallenger::new(&mut transcript);
            challenger.observe_bytes(root);
            challenger.sample_f128()
        }

        let root = [7; 32];
        let mut changed_root = root;
        changed_root[0] ^= 1;

        assert_ne!(sample_after_root(&root), sample_after_root(&changed_root));
    }

    #[test]
    fn verifier_rejects_an_invalid_positive_pow_nonce() {
        const SESSION: &[u8] = b"pcs-challenger-test";
        const INSTANCE: &[u8] = b"invalid-positive-pow";
        const BITS: u32 = 8;

        let seed = {
            let mut transcript = build_prover(SESSION, INSTANCE);
            transcript.public_message(POW_TAG);
            transcript.public_message(&BITS);
            transcript.verifier_message::<LocalF128>().to_bytes()
        };

        let mut prover = build_prover(SESSION, INSTANCE);
        let nonce = {
            let mut challenger = ProverChallenger::new(&mut prover);
            challenger.grind_pow(BITS)
        };
        let mut proof = prover.finish();

        let mut changed_nonce = nonce.wrapping_add(1);
        while pow_valid(&seed, changed_nonce, BITS) {
            changed_nonce = changed_nonce.wrapping_add(1);
        }
        proof
            .narg_string
            .copy_from_slice(&changed_nonce.to_le_bytes());

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        {
            let mut challenger = VerifierChallenger::new(&mut verifier);
            assert!(!challenger.verify_pow(changed_nonce, BITS));
            assert!(challenger.failed());
        }
        verifier.check_eof().unwrap();
    }

    #[test]
    fn ligerito_public_target_prefix_adds_no_narg_bytes() {
        let target = FlockF128::new(1, 2);
        let next_message = FlockF128::new(3, 4);
        let mut prover = build_prover(b"pcs-challenger-test", b"public-opening-target");
        {
            let mut challenger = ProverChallenger::new_ligerito(&mut prover, target);
            challenger.observe_label(LIGERITO_BASIS_LABEL);
            challenger.observe_f128(target);
            challenger.observe_f128(next_message);
            assert!(!challenger.failed());
        }
        let proof = prover.finish();
        assert_eq!(proof.narg_string, from_flock_f128(next_message).to_bytes());

        let mut verifier = build_verifier(b"pcs-challenger-test", b"public-opening-target", &proof);
        {
            let mut challenger = VerifierChallenger::new_ligerito(&mut verifier, target);
            challenger.observe_label(LIGERITO_BASIS_LABEL);
            challenger.observe_f128(target);
            challenger.observe_f128(next_message);
            assert!(!challenger.failed());
        }
        verifier.check_eof().unwrap();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn challenger_methods_round_trip(
            scalar_words in (any::<u64>(), any::<u64>()),
            slice_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..16),
            bytes in prop::collection::vec(any::<u8>(), 0..64),
            sample_count in 0usize..8,
            pow_bits in 0u32..=6,
        ) {
            let scalar = FlockF128::new(scalar_words.0, scalar_words.1);
            let slice = slice_words
                .iter()
                .map(|&(lo, hi)| FlockF128::new(lo, hi))
                .collect::<Vec<_>>();

            let mut prover = build_prover(b"pcs-challenger-test", b"method-round-trip");
            let (sampled_scalar, sampled_vector, nonce) = {
                let mut challenger = ProverChallenger::new(&mut prover);
                challenger.observe_label(b"test-label");
                challenger.observe_f128(scalar);
                challenger.observe_f128_slice(&slice);
                challenger.observe_bytes(&bytes);
                (
                    challenger.sample_f128(),
                    challenger.sample_f128_vec(sample_count),
                    challenger.grind_pow(pow_bits),
                )
            };
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-challenger-test",
                b"method-round-trip",
                &proof,
            );
            {
                let mut challenger = VerifierChallenger::new(&mut verifier);
                challenger.observe_label(b"test-label");
                challenger.observe_f128(scalar);
                challenger.observe_f128_slice(&slice);
                challenger.observe_bytes(&bytes);
                prop_assert_eq!(challenger.sample_f128(), sampled_scalar);
                prop_assert_eq!(challenger.sample_f128_vec(sample_count), sampled_vector);
                prop_assert!(challenger.verify_pow(nonce, pow_bits));
                prop_assert!(!challenger.failed());
            }
            prop_assert!(verifier.check_eof().is_ok());
        }

        #[test]
        fn find_pow_returns_the_first_valid_nonce(
            seed in any::<[u8; 16]>(),
            bits in 0u32..=8,
        ) {
            let nonce = find_pow(&seed, bits);

            prop_assert!(pow_valid(&seed, nonce, bits));
            prop_assert!((0..nonce).all(|candidate| !pow_valid(&seed, candidate, bits)));
        }
    }
}
