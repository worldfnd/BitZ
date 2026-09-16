//! Typed Fiat–Shamir challenge sampling.

use field::{F128, Fq};

use crate::{ProverState, VerifierState};

/// A type that knows how to construct itself from transcript squeezes.
///
/// Implementations must deterministically map independent, uniformly random
/// `u128` inputs to an exactly uniform `Self`, with finite expected input
/// consumption. Violating this distribution contract can weaken the
/// soundness of protocols that use the sampled value as a Fiat–Shamir
/// challenge.
pub trait TranscriptChallenge: Sized {
    /// Samples a challenge using successive deterministic `u128` squeezes.
    ///
    /// Through [`ProverState::squeeze`] or [`VerifierState::squeeze`], every
    /// invocation advances the captured transcript state, so rejection
    /// sampling consumes the same challenge stream on the prover and verifier.
    fn from_squeezes(next_u128: impl FnMut() -> u128) -> Self;
}

impl ProverState {
    /// Samples a typed Fiat–Shamir challenge from this transcript.
    pub fn squeeze<T: TranscriptChallenge>(&mut self) -> T {
        T::from_squeezes(|| self.verifier_message::<u128>())
    }
}

impl VerifierState<'_> {
    /// Samples a typed Fiat–Shamir challenge from this transcript.
    pub fn squeeze<T: TranscriptChallenge>(&mut self) -> T {
        T::from_squeezes(|| self.verifier_message::<u128>())
    }
}

impl<const Q: u128> TranscriptChallenge for Fq<Q> {
    fn from_squeezes(mut next_u128: impl FnMut() -> u128) -> Self {
        // Validate the const-generic modulus before using it below.
        let _ = Self::BITS;
        let q = Self::modulus();

        // The u128 range is not generally an exact multiple of Q. Reject its
        // incomplete final interval before reducing so every residue has the
        // same number of preimages.
        let rejection_remainder = (u128::MAX % q + 1) % q;
        let max_accepted = u128::MAX - rejection_remainder;

        loop {
            let candidate = next_u128();
            if candidate <= max_accepted {
                return Self::from(candidate);
            }
        }
    }
}

/// A `bits`-bit probable prime from successive `u128` squeezes: each
/// candidate is the squeeze masked to `bits` bits with its top bit and its
/// low bit set, the first one passing [`field::is_probable_prime`] is
/// taken. Both sides squeeze the same stream, so both land on the same
/// prime; a protocol installs it with [`field::set_modulus`] and works in
/// `Fq<RUNTIME>` from then on. `bits` is at most 126 (the field's bound).
pub fn prime_from_squeezes(mut next_u128: impl FnMut() -> u128, bits: u32) -> u128 {
    assert!((2..=126).contains(&bits), "a prime of 2 to 126 bits");
    let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
    loop {
        let candidate = (next_u128() & mask) | (1u128 << (bits - 1)) | 1;
        if field::is_probable_prime(candidate) {
            return candidate;
        }
    }
}

impl ProverState {
    /// Samples a `bits`-bit probable prime from this transcript.
    pub fn squeeze_prime(&mut self, bits: u32) -> u128 {
        prime_from_squeezes(|| self.verifier_message::<u128>(), bits)
    }
}

impl VerifierState<'_> {
    /// Samples a `bits`-bit probable prime from this transcript.
    pub fn squeeze_prime(&mut self, bits: u32) -> u128 {
        prime_from_squeezes(|| self.verifier_message::<u128>(), bits)
    }
}

impl TranscriptChallenge for F128 {
    fn from_squeezes(mut next_u128: impl FnMut() -> u128) -> Self {
        // Every 128-bit string is exactly one binary-field element.
        Self::from(next_u128())
    }
}

#[cfg(test)]
mod tests {
    use field::{F128, FqDefault, Q100};

    use super::TranscriptChallenge;
    use crate::{build_prover, build_verifier};

    const SESSION: &[u8] = b"transcript/typed-challenge/test";
    const INSTANCE: &[u8] = b"fq-rejection-sampling";

    #[test]
    fn fq_rejects_the_incomplete_final_interval() {
        let rejection_remainder = (u128::MAX % Q100 + 1) % Q100;
        let max_accepted = u128::MAX - rejection_remainder;
        let mut candidates = [max_accepted + 1, u128::MAX, max_accepted].into_iter();
        let mut squeezes = 0;

        assert_eq!(max_accepted % Q100, Q100 - 1);
        assert_eq!((max_accepted + 1) % Q100, 0);

        let challenge = FqDefault::from_squeezes(|| {
            squeezes += 1;
            candidates.next().unwrap()
        });

        assert_eq!(squeezes, 3);
        assert_eq!(challenge, FqDefault::from(Q100 - 1));
    }

    #[test]
    fn prover_and_verifier_squeeze_in_lockstep() {
        let mut prover = build_prover(SESSION, INSTANCE);
        let prover_fq = prover.squeeze::<FqDefault>();
        let prover_f128 = prover.squeeze::<F128>();
        let proof = prover.finish();

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        let verifier_fq = verifier.squeeze::<FqDefault>();
        let verifier_f128 = verifier.squeeze::<F128>();

        assert_eq!(verifier_fq, prover_fq);
        assert_eq!(verifier_f128, prover_f128);
        verifier.check_eof().unwrap();
    }

    #[test]
    fn prover_and_verifier_sample_the_same_prime() {
        let mut prover = build_prover(SESSION, b"sampled-prime");
        let prime = prover.squeeze_prime(100);
        let after = prover.squeeze::<field::FqRuntime>();
        let proof = prover.finish();
        assert!(proof.narg_string.is_empty());
        assert_eq!(128 - prime.leading_zeros(), 100);
        assert_eq!(prime % 2, 1);
        assert!(field::is_probable_prime(prime));
        assert_ne!(prime, Q100);

        let mut verifier = build_verifier(SESSION, b"sampled-prime", &proof);
        assert_eq!(verifier.squeeze_prime(100), prime);
        assert_eq!(verifier.squeeze::<field::FqRuntime>(), after);
        verifier.check_eof().unwrap();
    }

    #[test]
    fn typed_f128_squeeze_matches_direct_decoding() {
        let mut typed = build_prover(SESSION, INSTANCE);
        let typed_challenge = typed.squeeze::<F128>();

        let mut direct = build_prover(SESSION, INSTANCE);
        let direct_challenge = direct.verifier_message::<F128>();

        assert_eq!(typed_challenge, direct_challenge);
    }
}
