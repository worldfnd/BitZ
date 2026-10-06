//! Typed Fiat–Shamir challenge sampling.

pub mod prime;

use crate::{ProverState, VerifierState};
use field::dynamic::DynField;
use field::{F128, Fq};

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

pub trait SqueezableTranscript {
    /// Samples a typed Fiat–Shamir challenge from this transcript.
    fn squeeze<T: TranscriptChallenge>(&mut self) -> T;

    /// Squeezes a `bits`-bit prime from this transcript.
    fn squeeze_prime(&mut self, bits: u32) -> u128;
}

impl SqueezableTranscript for ProverState {
    fn squeeze<T: TranscriptChallenge>(&mut self) -> T {
        T::from_squeezes(|| self.verifier_message::<u128>())
    }

    fn squeeze_prime(&mut self, bits: u32) -> u128 {
        prime::sample(|| self.verifier_message::<u128>(), bits)
    }
}

impl SqueezableTranscript for VerifierState<'_> {
    fn squeeze<T: TranscriptChallenge>(&mut self) -> T {
        T::from_squeezes(|| self.verifier_message::<u128>())
    }

    fn squeeze_prime(&mut self, bits: u32) -> u128 {
        prime::sample(|| self.verifier_message::<u128>(), bits)
    }
}

/// Sample `u128` whose residue modulo `modulus` is uniform, from successive
/// squeezes.
///
/// The u128 range is not generally an exact multiple of the modulus. Reject
/// its incomplete final interval before reducing so every residue has the
/// same number of preimages.
fn unbiased_u128(modulus: u128, mut next_u128: impl FnMut() -> u128) -> u128 {
    let rejection_remainder = (u128::MAX % modulus + 1) % modulus;
    let max_accepted = u128::MAX - rejection_remainder;

    loop {
        let candidate = next_u128();
        if candidate <= max_accepted {
            return candidate;
        }
    }
}

impl<const Q: u128> TranscriptChallenge for Fq<Q> {
    fn from_squeezes(next_u128: impl FnMut() -> u128) -> Self {
        // Validate the const-generic modulus before using it below.
        let _ = Self::META;
        Self::from(unbiased_u128(Q, next_u128))
    }
}

impl TranscriptChallenge for DynField {
    fn from_squeezes(next_u128: impl FnMut() -> u128) -> Self {
        Self::from(unbiased_u128(DynField::config().modulus, next_u128))
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
    use super::*;
    use crate::{build_prover, build_verifier};
    use field::Q100;
    use spongefish::Encoding;

    const SESSION: &[u8] = b"transcript/typed-challenge/test";
    const INSTANCE: &[u8] = b"fq-rejection-sampling";

    type F = field::FqDefault;

    #[test]
    fn fq_rejects_the_incomplete_final_interval() {
        let rejection_remainder = (u128::MAX % Q100 + 1) % Q100;
        let max_accepted = u128::MAX - rejection_remainder;
        let mut candidates = [max_accepted + 1, u128::MAX, max_accepted].into_iter();
        let mut squeezes = 0;

        assert_eq!(max_accepted % Q100, Q100 - 1);
        assert_eq!((max_accepted + 1) % Q100, 0);

        let challenge = F::from_squeezes(|| {
            squeezes += 1;
            candidates.next().unwrap()
        });

        assert_eq!(squeezes, 3);
        assert_eq!(challenge, F::from(Q100 - 1));
    }

    /// Under the same modulus, the same stream gives the same element.
    #[test]
    fn dyn_field_squeezes_as_fq() {
        field::dynamic::test_support::with_modulus(Q100, || {
            let mut typed = build_prover(SESSION, INSTANCE);
            let dynamic = typed.squeeze::<DynField>();
            let mut fixed = build_prover(SESSION, INSTANCE);
            let fq = fixed.squeeze::<F>();
            assert_eq!(dynamic.encode().as_ref(), fq.encode().as_ref());

            let rejection_remainder = (u128::MAX % Q100 + 1) % Q100;
            let max_accepted = u128::MAX - rejection_remainder;
            let mut candidates = [max_accepted + 1, max_accepted].into_iter();
            let challenge = DynField::from_squeezes(|| candidates.next().unwrap());
            assert_eq!(challenge, DynField::from(Q100 - 1));
        });
    }

    #[test]
    fn prover_and_verifier_squeeze_in_lockstep() {
        let mut prover = build_prover(SESSION, INSTANCE);
        let prover_fq = prover.squeeze::<F>();
        let prover_f128 = prover.squeeze::<F128>();
        let proof = prover.finish();

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        let verifier_fq = verifier.squeeze::<F>();
        let verifier_f128 = verifier.squeeze::<F128>();

        assert_eq!(verifier_fq, prover_fq);
        assert_eq!(verifier_f128, prover_f128);
        verifier.check_eof().unwrap();
    }

    #[test]
    fn prover_and_verifier_squeeze_the_same_prime() {
        let mut prover = build_prover(SESSION, INSTANCE);
        let prime = prover.squeeze_prime(F::META.bits);
        let after = prover.squeeze::<F>();
        let proof = prover.finish();
        assert_eq!(u128::BITS - prime.leading_zeros(), F::META.bits);
        assert!(field::helpers::is_prime(prime));

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        assert_eq!(verifier.squeeze_prime(F::META.bits), prime);
        assert_eq!(verifier.squeeze::<F>(), after);
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
