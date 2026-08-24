//! Typed Fiat–Shamir challenge sampling.

use field::Fq;

use crate::{ProverState, VerifierState};

/// A type that knows how to construct itself from transcript squeezes.
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

        // The u128 range is not generally an exact multiple of Q. Reject its
        // incomplete final interval before reducing so every residue has the
        // same number of preimages.
        let rejection_remainder = (u128::MAX % Q + 1) % Q;
        let max_accepted = u128::MAX - rejection_remainder;

        loop {
            let candidate = next_u128();
            if candidate <= max_accepted {
                return Self::from(candidate);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use field::{FqDefault, Q100};

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
        let prover_challenges = [prover.squeeze::<FqDefault>(), prover.squeeze::<FqDefault>()];
        let proof = prover.finish();

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        let verifier_challenges = [
            verifier.squeeze::<FqDefault>(),
            verifier.squeeze::<FqDefault>(),
        ];

        assert_eq!(verifier_challenges, prover_challenges);
        verifier.check_eof().unwrap();
    }
}
