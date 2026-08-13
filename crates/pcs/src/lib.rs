//! Shared interface for the binary polynomial commitment.

mod bridge;
mod challenger;
mod commitment;
mod opening;

use field::F128;
use transcript::{ProverState, VerifierState};

pub use commitment::{Commitment, MerkleHash, Pcs, PcsConfig, ProverData};

/// Errors from commitment and linear-query operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommitError {
    /// The bit vector has no supported commitment shape.
    InvalidBitLength { len: usize },
    /// The coefficient vector does not match the committed bit vector.
    CoefficientLengthMismatch { expected: usize, actual: usize },
    /// The scheme configuration is not valid for the selected backend.
    InvalidConfiguration,
    /// The one-shot prover data has already produced an opening.
    ProverDataConsumed,
    /// The coefficients do not encode a supported flock ring-switch point.
    UnsupportedCoefficients,
    /// The transcript does not contain a complete canonical proof.
    MalformedProof,
    /// The commitment backend rejected an operation.
    Backend,
    /// The linear-query proof did not verify.
    VerificationFailed,
}

/// The terminal binary-field PCS used by Section 3.5, Construction 3.10.
///
/// Construction 3.10 reduces its `R`-valued claim through Construction 3.4.
/// This trait implements Construction 3.4, Phase 3, with `F_{2^nu} = F128`.
pub trait CommitScheme {
    /// The public commitment.
    type Commitment;
    /// Private data retained by the prover after commitment.
    type ProverData;

    /// Commits to `Enc_C(Pack_128(pi_2(bits)))`, Construction 3.10's oracle.
    fn commit(&self, bits: &[bool]) -> Result<(Self::Commitment, Self::ProverData), CommitError>;

    /// Proves `<coeffs, pi_2(bits)>_{F128} = target`.
    ///
    /// This is Construction 3.4's residual Phase 3 claim, with `nu = 128`.
    /// `coeffs` has one entry per bit. It must encode Flock's structured
    /// univariate-skip and multilinear evaluation weights. Other linear
    /// functionals return [`CommitError::UnsupportedCoefficients`]. `target`
    /// is the residual `mu_prime`, not Construction 3.10's original
    /// `R`-valued `mu`.
    fn prove_lin(
        &self,
        data: &Self::ProverData,
        coeffs: &[F128],
        target: F128,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError>;

    /// Verifies the same Construction 3.4, Phase 3 claim against `commitment`.
    fn verify_lin(
        &self,
        commitment: &Self::Commitment,
        coeffs: &[F128],
        target: F128,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), CommitError>;
}

#[cfg(test)]
mod tests {
    use transcript::{Proof, build_prover, build_verifier};

    use super::*;

    const SESSION: &[u8] = b"pcs-interface-test";
    const INSTANCE: &[u8] = b"two-bits";

    struct TestScheme;
    struct TestCommitment {
        len: usize,
    }
    struct TestProverData {
        len: usize,
    }

    impl CommitScheme for TestScheme {
        type Commitment = TestCommitment;
        type ProverData = TestProverData;

        fn commit(
            &self,
            bits: &[bool],
        ) -> Result<(Self::Commitment, Self::ProverData), CommitError> {
            Ok((
                TestCommitment { len: bits.len() },
                TestProverData { len: bits.len() },
            ))
        }

        fn prove_lin(
            &self,
            data: &Self::ProverData,
            coeffs: &[F128],
            _target: F128,
            _transcript: &mut ProverState,
        ) -> Result<(), CommitError> {
            check_coefficient_length(data.len, coeffs.len())
        }

        fn verify_lin(
            &self,
            commitment: &Self::Commitment,
            coeffs: &[F128],
            _target: F128,
            _transcript: &mut VerifierState<'_>,
        ) -> Result<(), CommitError> {
            check_coefficient_length(commitment.len, coeffs.len())
        }
    }

    fn check_coefficient_length(expected: usize, actual: usize) -> Result<(), CommitError> {
        if actual == expected {
            Ok(())
        } else {
            Err(CommitError::CoefficientLengthMismatch { expected, actual })
        }
    }

    #[test]
    fn interface_accepts_bits_and_f128_linear_queries() {
        let scheme = TestScheme;
        let (commitment, data) = scheme.commit(&[false, true]).unwrap();
        let coeffs = [F128::from(3u64), F128::from(5u64)];
        let target = F128::from(5u64);

        let mut prover = build_prover(SESSION, INSTANCE);
        scheme
            .prove_lin(&data, &coeffs, target, &mut prover)
            .unwrap();
        let proof = prover.finish();

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        scheme
            .verify_lin(&commitment, &coeffs, target, &mut verifier)
            .unwrap();
        verifier.check_eof().unwrap();
    }

    #[test]
    fn interface_reports_coefficient_length_mismatches() {
        let scheme = TestScheme;
        let (commitment, data) = scheme.commit(&[false, true]).unwrap();
        let coeffs = [F128::from(3u64)];
        let target = F128::from(0u64);
        let expected = CommitError::CoefficientLengthMismatch {
            expected: 2,
            actual: 1,
        };

        let mut prover = build_prover(SESSION, INSTANCE);
        assert_eq!(
            scheme.prove_lin(&data, &coeffs, target, &mut prover),
            Err(expected)
        );

        let proof = Proof::default();
        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        assert_eq!(
            scheme.verify_lin(&commitment, &coeffs, target, &mut verifier),
            Err(expected)
        );
    }
}
