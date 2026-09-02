//! Binary polynomial commitments with MLE and arbitrary inner-product openings.
//!
//! # Statement
//!
//! A vector of `2^m` bits defines a function `q: {0,1}^m → F2`.
//! Bit-vector indices and evaluation-point coordinates use low-bit-first order.
//! For `r ∈ F128^m`, the multilinear extension is
//! `q̂(r) = Σ_{b ∈ {0,1}^m} q(b) · eq(b, r)`, where
//! `eq(b, r) = ∏_i (b_i · r_i + (1 - b_i) · (1 - r_i))`.
//! [`OpeningQuery::Mle`] claims `q̂(point) = target`.
//! [`OpeningQuery::InnerProduct`] claims `Σ_i q(i) · weights[i] = target`.
//!
//! # Packing and opening
//!
//! The caller packs the seven low Boolean coordinates into one `F128` element:
//! `q_pkd(y) = Σ_{v ∈ {0,1}^7} q(y, v) · basis[v]`.
//! Flock Reed–Solomon-encodes `q_pkd` and commits to its codeword with a Merkle root.
//!
//! The opening splits `r` into `r_lo = r[0..7]` and `r_hi = r[7..m]`.
//! It computes `s_v = q̂(r_hi, v)` and checks
//! `target = Σ_v eq(r_lo, v) · s_v`.
//! Ring-switching transposes `(s_v)` into `(s_u)` and samples `r_dprime`.
//! It sets `beta0 = Σ_u eq(r_dprime, u) · s_u`.
//! Recursive Ligerito proves `Σ_y B(y) · q_pkd(y) = beta0` against the committed root.
//! Arbitrary inner products apply the GHASH coefficient-projection dual map.
//! They batch 128 packed coordinate claims into one Ligerito basis opening.
//!
//! # Interface
//!
//! - [`Pcs`] stores trusted Flock parameters and the expected bit length.
//! - [`Commitment`] contains the public Merkle root.
//! - [`ProverData`] retains the codeword and Merkle tree after commitment.
//! - [`OpeningQuery`] contains an MLE point or explicit inner-product weights.
//! - [`CommitScheme`] connects commitment, proving, and verification to project transcripts.
//!
//! The caller packs and retains the witness after [`CommitScheme::commit`].
//! [`CommitScheme::prove_lin`] dispatches both query variants.
//! It consumes the packed witness and borrows [`ProverData`].
//! The caller must use matching transcript session and instance labels.
//! The caller must also call `VerifierState::check_eof` after successful verification.
//!
//! # Example
//!
//! This example uses the zero polynomial, so its evaluation is zero at every point.
//!
//! ```
//! use common::Shape;
//! use field::F128;
//! use pcs::{
//!     CommitScheme, HashKind, LigeritoProfile, OpeningQuery, Pcs, StatementBinding,
//! };
//! use transcript::{build_prover, build_verifier};
//!
//! const M: usize = 22;
//! let shape = Shape::new(7, 15).unwrap();
//! let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
//! let packed_witness = vec![F128::default(); pcs.packed_len()];
//! let point = (0..M)
//!     .map(|coordinate| F128::from(coordinate as u64 + 2))
//!     .collect();
//! let query = OpeningQuery::Mle {
//!     point,
//!     target: F128::from(0u64),
//! };
//!
//! let (commitment, prover_data) = pcs.commit(&packed_witness).unwrap();
//! let mut prover = build_prover(b"pcs-example", b"zero-polynomial");
//! pcs.prove_lin(
//!     &prover_data,
//!     packed_witness,
//!     &query,
//!     StatementBinding::Bind,
//!     &mut prover,
//! )
//!     .unwrap();
//! let proof = prover.finish();
//!
//! let mut verifier = build_verifier(b"pcs-example", b"zero-polynomial", &proof);
//! pcs.verify_lin(
//!     &commitment,
//!     &query,
//!     StatementBinding::Bind,
//!     &mut verifier,
//! )
//!     .unwrap();
//! verifier.check_eof().unwrap();
//! ```

mod bridge;
mod challenger;
mod commitment;
mod inner_product;
mod ligerito;
mod mle;
mod ring_switch;
mod utils;
mod validation;

use field::F128;
use transcript::{ProverState, VerifierState};

pub use commitment::{Commitment, HashKind, Pcs, ProverData};
pub use flock_core::pcs::ligerito::LigeritoProfile;

/// A linear opening claim over the original committed bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpeningQuery {
    /// A multilinear evaluation claim.
    Mle {
        /// The evaluation point, in low-index-bit-first order.
        point: Vec<F128>,
        /// The claimed multilinear evaluation at `point`.
        target: F128,
    },
    /// An arbitrary `F128` inner-product claim.
    ///
    /// This opening requires [`LigeritoProfile::Secure`].
    InnerProduct {
        /// One weight for each original bit, in commitment index order.
        weights: Vec<F128>,
        /// The claimed inner product.
        target: F128,
    },
}

/// Controls statement binding for one opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementBinding {
    /// Binds the PCS parameters, commitment, query, and target.
    Bind,
    /// Uses a statement that the caller already bound.
    ///
    /// The caller must bind the same PCS parameters, commitment, query variant, fields, and target.
    AlreadyBound,
}

/// Errors from commitment and linear-query operations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommitError {
    /// The packed witness represents an unsupported bit length.
    InvalidBitLength,
    /// The evaluation point does not match the committed polynomial.
    PointLengthMismatch,
    /// The explicit weights do not match the committed bit vector.
    WeightLengthMismatch,
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
    /// The prover received an invalid linear claim.
    InvalidClaim,
    /// Arbitrary inner products require the unique-decoding profile.
    UnsupportedInnerProductProfile,
}

impl CommitError {
    pub(crate) fn invalid_configuration(description: impl Into<String>) -> Self {
        Self::InvalidConfiguration(description.into())
    }
}

/// A polynomial commitment scheme for linear claims over committed bit tables.
///
/// Let `q: {0,1}^m → F2` be the committed bit table. For `r ∈ F128^m`,
/// [`OpeningQuery::Mle`] proves
/// `q̂(r) = Σ_{b ∈ {0,1}^m} q(b) · eq(b, r) = target`, where
/// `eq(b, r) = ∏_i (b_i · r_i + (1 - b_i) · (1 - r_i))`.
/// [`OpeningQuery::InnerProduct`] accepts one arbitrary `F128` weight per bit.
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

    /// Consumes the exact packed witness and proves either opening query.
    ///
    /// [`OpeningQuery::InnerProduct`] requires [`LigeritoProfile::Secure`].
    fn prove_lin(
        &self,
        data: &Self::ProverData,
        packed_witness: Vec<F128>,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError>;

    /// Verifies either opening query against `commitment`.
    ///
    /// [`OpeningQuery::InnerProduct`] requires [`LigeritoProfile::Secure`].
    fn verify_lin(
        &self,
        commitment: &Self::Commitment,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
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
        data: &Self::ProverData,
        packed_witness: Vec<F128>,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut ProverState,
    ) -> Result<(), CommitError> {
        match query {
            OpeningQuery::Mle { point, target } => mle::open(
                self,
                data,
                packed_witness,
                point,
                *target,
                statement_binding,
                transcript,
            ),
            OpeningQuery::InnerProduct { weights, target } => inner_product::open(
                self,
                data,
                packed_witness,
                weights,
                *target,
                statement_binding,
                transcript,
            ),
        }
    }

    fn verify_lin(
        &self,
        commitment: &Self::Commitment,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), CommitError> {
        match query {
            OpeningQuery::Mle { point, target } => mle::verify(
                self,
                commitment,
                point,
                *target,
                statement_binding,
                transcript,
            ),
            OpeningQuery::InnerProduct { weights, target } => inner_product::verify(
                self,
                commitment,
                weights,
                *target,
                statement_binding,
                transcript,
            ),
        }
    }
}
