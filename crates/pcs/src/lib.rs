//! Binary polynomial commitments with MLE and factored inner-product openings.
//!
//! # Statement
//!
//! A vector of `2^m` bits defines a function `q: {0,1}^m → F2`.
//! Bit-vector indices and evaluation-point coordinates use low-bit-first order.
//! For `r ∈ F128^m`, the multilinear extension is
//! `q̂(r) = Σ_{b ∈ {0,1}^m} q(b) · eq(b, r)`, where
//! `eq(b, r) = ∏_i (b_i · r_i + (1 - b_i) · (1 - r_i))`.
//! [`OpeningQuery::Mle`] claims `q̂(point) = target`.
//! [`OpeningQuery::InnerProduct`] claims `Σ_{r,c} q(c * rows + r) · row_weights[r] · column_weights[c] = target`.
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
//! Ring-switching transposes `(s_v)` into `(s_u)` and samples `batching_point`.
//! It sets `packed_target = Σ_u eq(batching_point, u) · s_u`.
//! Recursive Ligerito proves `Σ_y B(y) · q_pkd(y) = packed_target` against the committed root.
//! Quadratic sumcheck reduces factored inner-product claims to MLE claims before this opening protocol.
//!
//! # Interface
//!
//! - [`Pcs`] stores trusted Flock parameters and the expected bit length.
//! - [`Root`] is the public Merkle root.
//! - [`ProverData`] retains the codeword and Merkle tree after commitment.
//! - [`VerifierData`] retains the root and OOD claim received before opening.
//! - [`OpeningQuery`] contains an MLE point and target, or a `common::LinearClaim<F128>`.
//! - [`CommitScheme`] connects commitment, proving, and verification to project transcripts.
//! - [`ConfigError`] reports configuration failures.
//! - [`CommitError`], [`ProveError`], and [`VerifyError`] report operation-specific failures.
//!
//! The caller packs and retains the witness after [`CommitScheme::commit`].
//! [`CommitScheme::prove_lin`] dispatches both query variants.
//! It consumes the packed witness and borrows [`ProverData`].
//! The caller must use matching transcript session and instance labels.
//! The caller must also call `VerifierState::check_eof` after successful verification.
//! Use [`Pcs::commit_with_ood`] and [`Pcs::receive_commitment`] before any
//! witness-dependent challenges to include the initial OOD claim. Proving batches
//! that retained claim automatically; verification uses [`Pcs::verify_lin_with_ood`].
//!
//! # Example
//!
//! This example uses the zero polynomial, so its evaluation is zero at every point.
//!
//! ```
//! use common::Shape;
//! use field::F128;
//! use num_traits::ConstZero;
//! use pcs::{
//!     CommitScheme, HashKind, LigeritoProfile, OpeningQuery, Pcs, StatementBinding,
//! };
//! use transcript::{build_prover, build_verifier};
//!
//! const M: usize = 22;
//! let shape = Shape::new(7, 15).unwrap();
//! let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
//! let packed_witness = vec![F128::ZERO; pcs.packed_len()];
//! let point = (0..M)
//!     .map(|coordinate| F128::from(coordinate as u64 + 2))
//!     .collect();
//! let query = OpeningQuery::Mle {
//!     point,
//!     target: F128::from(0u64),
//! };
//!
//! let mut prover = build_prover(b"pcs-example", b"zero-polynomial");
//! let (commitment, prover_data) = pcs.commit_with_ood(&packed_witness, &mut prover).unwrap();
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
//! let commitment = pcs.receive_commitment(commitment, &mut verifier).unwrap();
//! pcs.verify_lin_with_ood(
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
mod ligerito;
mod mle;
mod ood;
mod opening;
mod pow;
mod profiles;
mod sumcheck;
mod transpose;

#[cfg(test)]
#[path = "transpose/tests.rs"]
mod transpose_tests;

use field::F128;
use transcript::{ProverState, VerifierState};

pub use commitment::{CommitError, ConfigError, HashKind, Pcs, ProverData, VerifierData};
pub use common::{OpeningQuery, Root};
pub use flock_core::pcs::ligerito::LigeritoProfile;
pub use opening::{ProveError, VerifyError};

/// Controls statement binding for one opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementBinding {
    /// Binds the PCS parameters, commitment, query, and target.
    Bind,
    /// Uses a statement that the caller already bound.
    ///
    /// The caller must bind the same PCS parameters, commitment, query variant, fields, and target.
    /// For inner products, this covers both factor lengths, both factors, and the original target.
    /// The opening code still binds the MLE claim that sumcheck returns.
    AlreadyBound,
}

/// A polynomial commitment scheme for linear claims over committed bit tables.
///
/// Let `q: {0,1}^m → F2` be the committed bit table. For `r ∈ F128^m`,
/// [`OpeningQuery::Mle`] proves
/// `q̂(r) = Σ_{b ∈ {0,1}^m} q(b) · eq(b, r) = target`, where
/// `eq(b, r) = ∏_i (b_i · r_i + (1 - b_i) · (1 - r_i))`.
/// [`OpeningQuery::InnerProduct`] accepts row weights, column weights, and a target over `F128`.
/// Quadratic sumcheck reduces this claim to an MLE claim before the opening protocol.
pub trait CommitScheme {
    /// The public commitment.
    type Commitment;
    /// Private data retained by the prover after commitment.
    type ProverData;
    /// Commitment and OOD claim retained by the verifier before opening.
    type VerifierData;

    /// Commits the caller-owned packed witness to `Enc_C(q_pkd)`, where
    /// `q_pkd(y) = Σ_{v ∈ {0,1}^7} q(y, v) · basis[v]`.
    /// Bit `r` of element `i` must equal logical bit `128 * i + r`.
    fn commit(
        &self,
        packed_witness: &[F128],
    ) -> Result<(Self::Commitment, Self::ProverData), CommitError>;

    /// Commits and retains the initial OOD claim for subsequent openings.
    ///
    /// Call before witness-dependent challenges and continue with the same transcript.
    /// Profiles without initial OOD sampling omit that round.
    fn commit_with_ood(
        &self,
        packed_witness: &[F128],
        transcript: &mut ProverState,
    ) -> Result<(Self::Commitment, Self::ProverData), CommitError>;

    /// Receives the OOD claim for the public commitment before protocol challenges.
    ///
    /// Mirrors [`Self::commit_with_ood`]. The returned claim must be authenticated
    /// by [`Self::verify_lin_with_ood`] on the same transcript.
    fn receive_commitment(
        &self,
        commitment: Self::Commitment,
        transcript: &mut VerifierState<'_>,
    ) -> Result<Self::VerifierData, VerifyError>;

    /// Consumes the exact packed witness and proves either opening query.
    ///
    /// Inner-product claims first pass through quadratic sumcheck and then the MLE opening protocol.
    /// An OOD claim retained by [`Self::commit_with_ood`] is batched into the opening.
    fn prove_lin(
        &self,
        data: &Self::ProverData,
        packed_witness: Vec<F128>,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut ProverState,
    ) -> Result<(), ProveError>;

    /// Verifies either opening query against `commitment`.
    ///
    /// Inner-product claims first pass through quadratic sumcheck and then the MLE opening protocol.
    fn verify_lin(
        &self,
        commitment: &Self::Commitment,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), VerifyError>;

    /// Verifies the linear query batched with the retained OOD claim.
    ///
    /// Continue the transcript used by [`Self::receive_commitment`]. Borrowing
    /// the retained state permits multiple openings against the same commitment.
    fn verify_lin_with_ood(
        &self,
        commitment: &Self::VerifierData,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), VerifyError>;
}

impl CommitScheme for Pcs {
    type Commitment = Root;
    type ProverData = ProverData;
    type VerifierData = VerifierData;

    fn commit(
        &self,
        packed_witness: &[F128],
    ) -> Result<(Self::Commitment, Self::ProverData), CommitError> {
        Pcs::commit(self, packed_witness)
    }

    fn commit_with_ood(
        &self,
        packed_witness: &[F128],
        transcript: &mut ProverState,
    ) -> Result<(Self::Commitment, Self::ProverData), CommitError> {
        Pcs::commit_with_ood(self, packed_witness, transcript)
    }

    fn receive_commitment(
        &self,
        commitment: Self::Commitment,
        transcript: &mut VerifierState<'_>,
    ) -> Result<Self::VerifierData, VerifyError> {
        Pcs::receive_commitment(self, commitment, transcript)
    }

    fn prove_lin(
        &self,
        data: &Self::ProverData,
        packed_witness: Vec<F128>,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut ProverState,
    ) -> Result<(), ProveError> {
        opening::prove(
            self,
            data,
            packed_witness,
            query,
            statement_binding,
            transcript,
        )
    }

    fn verify_lin(
        &self,
        commitment: &Self::Commitment,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), VerifyError> {
        opening::verify(self, commitment, query, statement_binding, None, transcript)
    }

    fn verify_lin_with_ood(
        &self,
        commitment: &Self::VerifierData,
        query: &OpeningQuery,
        statement_binding: StatementBinding,
        transcript: &mut VerifierState<'_>,
    ) -> Result<(), VerifyError> {
        opening::verify_lin_with_ood(self, commitment, query, statement_binding, transcript)
    }
}
