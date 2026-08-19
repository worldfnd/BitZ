//! Binary polynomial commitments for multilinear extensions of bit tables.
//!
//! # Statement
//!
//! A vector of `2^m` bits defines a function `q: {0,1}^m → F2`.
//! Bit-vector indices and evaluation-point coordinates use low-bit-first order.
//! For `r ∈ F128^m`, the multilinear extension is
//! `q̂(r) = Σ_{b ∈ {0,1}^m} q(b) · eq(b, r)`, where
//! `eq(b, r) = ∏_i (b_i · r_i + (1 - b_i) · (1 - r_i))`.
//! An [`OpeningQuery`] claims that `q̂(query.point) = query.target`.
//!
//! # Packing and opening
//!
//! The caller packs the seven low Boolean coordinates into one `F128` element:
//! `q_pkd(y) = Σ_{v ∈ {0,1}^7} q(y, v) · basis[v]`.
//! Flock Reed–Solomon-encodes `q_pkd` and commits to its codeword with a Merkle root.
//!
//! The opening splits `r` into `r_lo = r[0..7]` and `r_hi = r[7..m]`.
//! It computes `s_v = q̂(r_hi, v)` and checks
//! `query.target = Σ_v eq(r_lo, v) · s_v`.
//! Ring-switching transposes `(s_v)` into `(s_u)` and samples `r_dprime`.
//! It sets `beta0 = Σ_u eq(r_dprime, u) · s_u`.
//! Recursive Ligerito proves `Σ_y B(y) · q_pkd(y) = beta0` against the committed root.
//!
//! # Interface
//!
//! - [`Pcs`] stores trusted Flock parameters and the expected bit length.
//! - [`Commitment`] contains the public Merkle root.
//! - [`ProverData`] retains the codeword and Merkle tree after commitment.
//! - [`OpeningQuery`] contains one evaluation point and its claimed value.
//! - [`CommitScheme`] connects commitment, proving, and verification to project transcripts.
//!
//! The caller packs and retains the witness after [`CommitScheme::commit`].
//! [`CommitScheme::prove_lin`] consumes the packed witness and [`ProverData`] for one opening.
//! The caller must use matching transcript session and instance labels.
//! The caller must also call `VerifierState::check_eof` after successful verification.
//!
//! # Example
//!
//! This example uses the zero polynomial, so its evaluation is zero at every point.
//!
//! ```
//! use field::F128;
//! use pcs::{CommitScheme, HashKind, LigeritoProfile, OpeningQuery, Pcs};
//! use transcript::{build_prover, build_verifier};
//!
//! const M: usize = 22;
//! let pcs = Pcs::new(M, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
//! let packed_witness = vec![F128::default(); pcs.packed_len()];
//! let point = (0..M)
//!     .map(|coordinate| F128::from(coordinate as u64 + 2))
//!     .collect();
//! let query = OpeningQuery {
//!     point,
//!     target: F128::from(0u64),
//! };
//!
//! let (commitment, prover_data) = pcs.commit(&packed_witness).unwrap();
//! let mut prover = build_prover(b"pcs-example", b"zero-polynomial");
//! pcs.prove_lin(prover_data, packed_witness, &query, &mut prover)
//!     .unwrap();
//! let proof = prover.finish();
//!
//! let mut verifier = build_verifier(b"pcs-example", b"zero-polynomial", &proof);
//! pcs.verify_lin(&commitment, &query, &mut verifier)
//!     .unwrap();
//! verifier.check_eof().unwrap();
//! ```

mod bridge;
mod challenger;
mod commitment;
mod open;
mod protocol;
mod verify;

use field::F128;
use transcript::{ProverState, VerifierState};

pub use commitment::{Commitment, HashKind, Pcs, ProverData};
pub use flock_core::pcs::ligerito::LigeritoProfile;

/// A standard multilinear evaluation claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningQuery {
    /// The evaluation point, in low-index-bit-first order.
    pub point: Vec<F128>,
    /// The claimed multilinear evaluation at `point`.
    pub target: F128,
}

/// Errors from commitment and linear-query operations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommitError {
    /// The packed witness represents an unsupported bit length.
    InvalidBitLength,
    /// The evaluation point does not match the committed polynomial.
    PointLengthMismatch,
    /// The scheme configuration is not valid for Flock, with a description of the failed check.
    InvalidConfiguration(String),
    /// The transcript does not contain a complete canonical proof.
    MalformedProof,
    /// The opening proof could not be serialized, with the serializer message.
    SerializationFailed(String),
    /// The serialized opening proof exceeds the transcript hint limit.
    ProofTooLarge,
    /// The linear-query proof did not verify.
    VerificationFailed,
}

impl CommitError {
    pub(crate) fn invalid_configuration(description: impl Into<String>) -> Self {
        Self::InvalidConfiguration(description.into())
    }
}

/// A polynomial commitment scheme for multilinear extensions of bit tables.
///
/// Let `q: {0,1}^m → F2` be the committed bit table. For `r ∈ F128^m`,
/// this trait proves
/// `q̂(r) = Σ_{b ∈ {0,1}^m} q(b) · eq(b, r) = query.target`, where
/// `eq(b, r) = ∏_i (b_i · r_i + (1 - b_i) · (1 - r_i))`.
pub trait CommitScheme {
    /// The public commitment.
    type Commitment;
    /// Private data retained by the prover after commitment.
    type ProverData;

    /// Commits the caller-owned packed witness to `Enc_C(q_pkd)`, where
    /// `q_pkd(y) = Σ_{v ∈ {0,1}^7} q(y, v) · basis[v]`.
    /// Bit `r` of element `i` must equal logical bit `128 * i + r`.
    fn commit(
        &self,
        packed_witness: &[F128],
    ) -> Result<(Self::Commitment, Self::ProverData), CommitError>;

    /// Consumes the exact packed witness passed to [`Self::commit`] and proves the claim.
    fn prove_lin(
        &self,
        data: Self::ProverData,
        packed_witness: Vec<F128>,
        query: &OpeningQuery,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError>;

    /// Verifies the same multilinear claim against `commitment`.
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

    fn commit(
        &self,
        packed_witness: &[F128],
    ) -> Result<(Self::Commitment, Self::ProverData), CommitError> {
        Pcs::commit(self, packed_witness)
    }

    fn prove_lin(
        &self,
        data: Self::ProverData,
        packed_witness: Vec<F128>,
        query: &OpeningQuery,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError> {
        open::open(self, data, packed_witness, query, transcript)
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
