//! Shared polynomial-commitment interface.

use field::F128;

/// Supported recursive Ligerito parameter profiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LigeritoProfile {
    #[default]
    Fast,
    Slim,
    Secure,
}

/// Supported Merkle hash functions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MerkleHash {
    #[default]
    Sha256,
    Blake3,
}

/// Public parameters for committing a packed bit polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitParams {
    /// Number of variables in the unpacked bit polynomial.
    pub m: usize,
    /// Logarithm of the inverse Reed--Solomon rate.
    pub log_inv_rate: usize,
    /// Logarithm of the number of interleaved Reed--Solomon codewords.
    pub log_batch_size: usize,
    /// Recursive Ligerito security profile.
    pub profile: LigeritoProfile,
    /// Hash used by the Merkle commitment.
    pub merkle_hash: MerkleHash,
}

/// One owned ring-switched evaluation claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningClaim {
    /// Claimed bit-MLE evaluation.
    pub value: F128,
    /// Public univariate-skip coordinate.
    pub z_skip: F128,
    /// Transcript-derived outer evaluation point.
    pub x_outer: Vec<F128>,
}

impl OpeningClaim {
    pub fn as_ref(&self) -> OpeningClaimRef<'_> {
        OpeningClaimRef {
            value: self.value,
            z_skip: self.z_skip,
            x_outer: &self.x_outer,
        }
    }
}

/// One borrowed ring-switched evaluation claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningClaimRef<'a> {
    /// Claimed bit-MLE evaluation.
    pub value: F128,
    /// Public univariate-skip coordinate.
    pub z_skip: F128,
    /// Transcript-derived outer evaluation point.
    pub x_outer: &'a [F128],
}

/// A commitment scheme for packed bit polynomials.
pub trait CommitScheme {
    type Commitment;
    type ProverState;
    type Opening;
    type Error;

    /// Commits to `packed_witness` and returns public and private outputs.
    fn commit(
        &self,
        packed_witness: &[F128],
        params: &CommitParams,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error>;

    /// Opens one or more transcript-derived evaluation claims.
    fn open(
        &self,
        state: &Self::ProverState,
        claims: &[OpeningClaim],
        transcript: &mut transcript::ProverState,
    ) -> Result<Self::Opening, Self::Error>;

    /// Verifies one or more evaluation claims against `commitment`.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        claims: &[OpeningClaimRef<'_>],
        opening: &Self::Opening,
        transcript: &mut transcript::VerifierState<'_>,
    ) -> Result<(), Self::Error>;
}
