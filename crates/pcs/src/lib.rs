//! Shared interface for the binary polynomial commitment.

mod bridge;
mod challenger;
mod commitment;
mod open;
mod protocol;
mod verify;

use field::F128;
use transcript::{ProverState, VerifierState};

pub use commitment::{Commitment, HashKind, Pcs, ProverData};

/// A standard multilinear evaluation claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningQuery {
    /// The evaluation point, in low-index-bit-first order.
    pub point: Vec<F128>,
    /// The claimed multilinear evaluation at `point`.
    pub target: F128,
}

/// Errors from commitment and linear-query operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommitError {
    /// The bit vector has no supported commitment shape.
    InvalidBitLength,
    /// The evaluation point does not match the committed polynomial.
    PointLengthMismatch,
    /// The scheme configuration is not valid for FLoCK.
    InvalidConfiguration,
    /// The transcript does not contain a complete canonical proof.
    MalformedProof,
    /// FLoCK rejected an operation.
    Flock,
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

    /// Proves `MLE(pi_2(bits))(query.point) = query.target`.
    ///
    /// This is Construction 3.4's residual Phase 3 claim, with `nu = 128`.
    /// `query.target` is the residual `mu_prime`, not Construction 3.10's
    /// original `R`-valued `mu`.
    fn prove_lin(
        &self,
        data: Self::ProverData,
        query: &OpeningQuery,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError>;

    /// Verifies the same Construction 3.4, Phase 3 claim against `commitment`.
    fn verify_lin(
        &self,
        commitment: &Self::Commitment,
        query: &OpeningQuery,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), CommitError>;
}

impl CommitScheme for Pcs {
    type Commitment = Commitment;
    type ProverData = ProverData;

    fn commit(&self, bits: &[bool]) -> Result<(Self::Commitment, Self::ProverData), CommitError> {
        Pcs::commit(self, bits)
    }

    fn prove_lin(
        &self,
        data: Self::ProverData,
        query: &OpeningQuery,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError> {
        open::open(self, data, query, transcript)
    }

    fn verify_lin(
        &self,
        commitment: &Self::Commitment,
        query: &OpeningQuery,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), CommitError> {
        verify::verify(self, commitment, query, transcript)
    }
}

#[cfg(test)]
mod tests {
    use flock_core::pcs::ligerito::LigeritoProfile;
    use transcript::{Proof, build_prover, build_verifier};

    use super::*;

    const SESSION: &[u8] = b"pcs-interface-test";
    const INSTANCE: &[u8] = b"two-bits";
    const REAL_INSTANCE: &[u8] = b"m22-singleton-opening";

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
            data: Self::ProverData,
            query: &OpeningQuery,
            _transcript: &mut ProverState,
        ) -> Result<(), CommitError> {
            check_point_length(data.len, query.point.len())
        }

        fn verify_lin(
            &self,
            commitment: &Self::Commitment,
            query: &OpeningQuery,
            _transcript: &mut VerifierState<'_>,
        ) -> Result<(), CommitError> {
            check_point_length(commitment.len, query.point.len())
        }
    }

    fn check_point_length(expected: usize, actual: usize) -> Result<(), CommitError> {
        if actual == expected {
            Ok(())
        } else {
            Err(CommitError::PointLengthMismatch)
        }
    }

    #[test]
    fn interface_accepts_a_multilinear_opening_query() {
        let scheme = TestScheme;
        let (commitment, data) = scheme.commit(&[false, true]).unwrap();
        let query = OpeningQuery {
            point: vec![F128::from(3u64), F128::from(5u64)],
            target: F128::from(7u64),
        };

        let mut prover = build_prover(SESSION, INSTANCE);
        scheme.prove_lin(data, &query, &mut prover).unwrap();
        let proof = prover.finish();

        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        scheme
            .verify_lin(&commitment, &query, &mut verifier)
            .unwrap();
        verifier.check_eof().unwrap();
    }

    #[test]
    fn interface_reports_point_length_mismatches() {
        let scheme = TestScheme;
        let (commitment, data) = scheme.commit(&[false, true]).unwrap();
        let query = OpeningQuery {
            point: vec![F128::from(3u64)],
            target: F128::from(0u64),
        };
        let expected = CommitError::PointLengthMismatch;

        let mut prover = build_prover(SESSION, INSTANCE);
        assert_eq!(scheme.prove_lin(data, &query, &mut prover), Err(expected));

        let proof = Proof::default();
        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        assert_eq!(
            scheme.verify_lin(&commitment, &query, &mut verifier),
            Err(expected)
        );
    }

    fn singleton_target(point: &[F128], index: usize) -> F128 {
        let one = F128::from(1u64);
        point
            .iter()
            .copied()
            .enumerate()
            .fold(one, |target, (coordinate, value)| {
                let weight = if ((index >> coordinate) & 1) == 1 {
                    value
                } else {
                    one + value
                };
                target * weight
            })
    }

    #[test]
    fn real_pcs_opening_round_trip_rejects_mutations() {
        const M: usize = 22;
        const SINGLETON: usize = (1 << 21) | (1 << 7) | 0b101_0101;

        let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3);
        let mut bits = vec![false; pcs.bit_len()];
        bits[SINGLETON] = true;

        let point = (0..M)
            .map(|coordinate| F128::from(coordinate as u64 + 2))
            .collect::<Vec<_>>();
        let query = OpeningQuery {
            target: singleton_target(&point, SINGLETON),
            point,
        };

        let (commitment, data) = pcs.commit(&bits).unwrap();
        let mut prover = build_prover(SESSION, REAL_INSTANCE);
        pcs.prove_lin(data, &query, &mut prover).unwrap();
        let proof = prover.finish();

        let mut verifier = build_verifier(SESSION, REAL_INSTANCE, &proof);
        pcs.verify_lin(&commitment, &query, &mut verifier).unwrap();
        verifier.check_eof().unwrap();

        let mut changed_query = query.clone();
        changed_query.target += F128::from(1u64);
        let mut verifier = build_verifier(SESSION, REAL_INSTANCE, &proof);
        assert!(
            pcs.verify_lin(&commitment, &changed_query, &mut verifier)
                .is_err()
        );

        let mut changed_root = *commitment.root();
        changed_root[0] ^= 1;
        let changed_commitment = Commitment::from_root(changed_root);
        let mut verifier = build_verifier(SESSION, REAL_INSTANCE, &proof);
        assert!(
            pcs.verify_lin(&changed_commitment, &query, &mut verifier)
                .is_err()
        );

        let mut changed_stream = proof.clone();
        changed_stream.narg_string[0] ^= 1;
        let mut verifier = build_verifier(SESSION, REAL_INSTANCE, &changed_stream);
        assert_eq!(
            pcs.verify_lin(&commitment, &query, &mut verifier),
            Err(CommitError::MalformedProof)
        );

        let mut truncated_hint = proof.clone();
        truncated_hint.hints.pop();
        let mut verifier = build_verifier(SESSION, REAL_INSTANCE, &truncated_hint);
        assert_eq!(
            pcs.verify_lin(&commitment, &query, &mut verifier),
            Err(CommitError::MalformedProof)
        );

        let mut changed_hint = proof.clone();
        *changed_hint.hints.last_mut().unwrap() ^= 1;
        let mut verifier = build_verifier(SESSION, REAL_INSTANCE, &changed_hint);
        assert!(pcs.verify_lin(&commitment, &query, &mut verifier).is_err());

        let mut trailing_hint = proof;
        trailing_hint.hints.push(0);
        let mut verifier = build_verifier(SESSION, REAL_INSTANCE, &trailing_hint);
        pcs.verify_lin(&commitment, &query, &mut verifier).unwrap();
        assert!(verifier.check_eof().is_err());
    }
}
