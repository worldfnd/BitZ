//! Shared nonce search and explicit transcript grinding boundaries.

use crate::PublicTranscript;

const POW_HASH_TAG: &[u8] = b"bitz-pcs-pow-v1";
const POW_TRANSCRIPT_TAG: &[u8] = b"bitz-transcript-pow-v1";

/// The largest supported grinding difficulty.
pub const MAX_GRINDING_BITS: u32 = 32;

pub(crate) fn absorb_header(transcript: &mut impl PublicTranscript, label: &[u8], bits: u32) {
    transcript.public_message(POW_TRANSCRIPT_TAG);
    transcript.public_message(&(label.len() as u64));
    transcript.public_message(label);
    transcript.public_message(&bits);
}

/// Returns the first valid nonce in ascending order.
///
/// The caller supplies transcript framing. Difficulties above 32 panic.
pub fn find(seed: &[u8; 16], bits: u32) -> u64 {
    assert!(
        bits <= MAX_GRINDING_BITS,
        "grinding difficulty exceeds 32 bits"
    );
    let mut nonce = 0u64;
    loop {
        if valid(seed, nonce, bits) {
            return nonce;
        }
        nonce = nonce.checked_add(1).expect("proof-of-work nonce exhausted");
    }
}

/// Checks the hash difficulty. Zero difficulty accepts only nonce zero.
///
/// Difficulties above 32 return false.
pub fn valid(seed: &[u8; 16], nonce: u64, bits: u32) -> bool {
    if bits > MAX_GRINDING_BITS {
        return false;
    }
    if bits == 0 {
        return nonce == 0;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(POW_HASH_TAG);
    hasher.update(seed);
    hasher.update(&nonce.to_le_bytes());
    let digest = hasher.finalize();
    // The 32-bit cap makes the first four hash bytes sufficient.
    u32::from_be_bytes(digest.as_bytes()[..4].try_into().unwrap()).leading_zeros() >= bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_prover, build_verifier};
    use field::F128;

    const SESSION: &[u8] = b"grinding-test";
    const INSTANCE: &[u8] = b"instance";
    const LABEL: &[u8] = b"test/cubic/v1";

    #[test]
    fn zero_grinding_leaves_the_transcript_unchanged() {
        let mut unmodified = build_prover(SESSION, INSTANCE);
        let expected = unmodified.verifier_message::<F128>();
        let unmodified_proof = unmodified.finish();

        let mut prover = build_prover(SESSION, INSTANCE);
        prover.grind(LABEL, 0);
        assert_eq!(prover.verifier_message::<F128>(), expected);
        let proof = prover.finish();
        assert_eq!(proof.narg_string, unmodified_proof.narg_string);

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        verifier.grind(LABEL, 0).unwrap();
        assert_eq!(verifier.verifier_message::<F128>(), expected);
        verifier.check_eof().unwrap();
        assert_eq!(find(&[0; 16], 0), 0);
        assert!(valid(&[0; 16], 0, 0));
        assert!(!valid(&[0; 16], 1, 0));
    }

    #[test]
    fn grinding_replays_and_rejects_an_invalid_nonce() {
        const BITS: u32 = 8;
        let mut prover = build_prover(SESSION, INSTANCE);
        prover.grind(LABEL, BITS);
        let expected = prover.verifier_message::<F128>();
        let mut proof = prover.finish();
        assert_eq!(proof.narg_string.len(), 8);

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        verifier.grind(LABEL, BITS).unwrap();
        assert_eq!(verifier.verifier_message::<F128>(), expected);
        verifier.check_eof().unwrap();

        let mut seed_transcript = build_prover(SESSION, INSTANCE);
        absorb_header(&mut seed_transcript, LABEL, BITS);
        let seed = seed_transcript.verifier_message::<F128>().to_bytes();
        let invalid = (0..).find(|&nonce| !valid(&seed, nonce, BITS)).unwrap();
        proof.narg_string.copy_from_slice(&invalid.to_le_bytes());
        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        assert!(verifier.grind(LABEL, BITS).is_err());
    }

    #[test]
    fn grinding_binds_label_difficulty_and_history() {
        fn seed(label: &[u8], bits: u32, message: &[u8]) -> F128 {
            let mut prover = build_prover(SESSION, INSTANCE);
            prover.public_message(message);
            absorb_header(&mut prover, label, bits);
            prover.verifier_message()
        }
        let expected = seed(LABEL, 8, b"first");
        assert_ne!(seed(b"test/affine/v1", 8, b"first"), expected);
        assert_ne!(seed(LABEL, 9, b"first"), expected);
        assert_ne!(seed(LABEL, 8, b"second"), expected);
    }

    #[test]
    fn verifier_rejects_excessive_difficulty_and_truncated_nonces() {
        let mut prover = build_prover(SESSION, INSTANCE);
        prover.grind(LABEL, 2);
        let mut proof = prover.finish();
        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        assert!(verifier.grind(LABEL, MAX_GRINDING_BITS + 1).is_err());
        assert!(!valid(&[0; 16], 0, MAX_GRINDING_BITS + 1));
        proof.narg_string.pop();
        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        assert!(verifier.grind(LABEL, 2).is_err());
    }
}
