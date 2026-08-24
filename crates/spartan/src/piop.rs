//! Composition of Spartan's outer and inner sumchecks.

use circuit::witgen::PackedWitness;
use field::FqDefault;
use poly::{
    DenseMultilinearExtension, MleClaimError, ScaledMleEvaluationClaim, make_equality_factors,
};
use transcript::{ProverState, VerifierState};

use crate::matrix::{
    PreparedConstraintMatrices, SpartanMatrixError, SpartanMatrixOperations, build_assignment_mle,
};
use crate::sumcheck::{
    FqChallengeSource, OuterSumcheckProof, R1csProductMles, SumcheckError, SumcheckProof,
    challenge_fq, prove_inner_sumcheck, prove_outer_sumcheck,
};

/// The two sumcheck proofs comprising the Spartan PIOP reduction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanPiopProof<F> {
    pub outer: OuterSumcheckProof<F>,
    pub inner: SumcheckProof<F, 3>,
}

/// Failures while composing or checking the Spartan PIOP.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpartanError {
    Matrix(SpartanMatrixError),
    Sumcheck(SumcheckError),
    MleClaim(MleClaimError),
    InvalidProductDimensions,
    InvalidAssignmentDimensions,
    InvalidMleClaim,
}

impl From<SpartanMatrixError> for SpartanError {
    fn from(error: SpartanMatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<SumcheckError> for SpartanError {
    fn from(error: SumcheckError) -> Self {
        Self::Sumcheck(error)
    }
}

impl From<MleClaimError> for SpartanError {
    fn from(error: MleClaimError) -> Self {
        Self::MleClaim(error)
    }
}

/// Runs the complete outer and inner sumcheck provers.
///
/// Transcript order:
///
/// 1. absorb the canonical matrix-statement digest;
/// 2. sample `tau`;
/// 3. run the outer sumcheck;
/// 4. absorb `[Ah(r_x), Bh(r_x), Ch(r_x)]`;
/// 5. sample `rho`;
/// 6. run the inner sumcheck.
pub fn prove_spartan_piop(
    transcript: &mut ProverState,
    matrices: &PreparedConstraintMatrices,
    products: R1csProductMles<FqDefault>,
    assignment: DenseMultilinearExtension<FqDefault>,
) -> Result<
    (
        SpartanPiopProof<FqDefault>,
        ScaledMleEvaluationClaim<FqDefault>,
    ),
    SpartanError,
> {
    let num_row_vars = matrices.num_row_vars();
    let num_column_vars = matrices.num_column_vars();
    if products.az.num_vars() != num_row_vars
        || products.bz.num_vars() != num_row_vars
        || products.cz.num_vars() != num_row_vars
    {
        return Err(SpartanError::InvalidProductDimensions);
    }
    if assignment.num_vars() != num_column_vars {
        return Err(SpartanError::InvalidAssignmentDimensions);
    }

    transcript.public_message(matrices.digest());
    let tau = challenge_point(transcript, num_row_vars);
    let equality_factors =
        make_equality_factors(&tau).map_err(|_| SpartanMatrixError::InvalidMleOperation)?;
    let outer = prove_outer_sumcheck(
        transcript,
        FqDefault::from(0u128),
        equality_factors,
        products,
    )?;

    // The outer prover absorbed these evaluations before returning.
    let rho = challenge_fq(transcript);
    let inner_initial_claim = outer.proof.az_mle_claim
        + rho * outer.proof.bz_mle_claim
        + rho * rho * outer.proof.cz_mle_claim;
    let batched_matrix = matrices.bind_and_batch(&outer.eval_points, rho)?;
    let inner = prove_inner_sumcheck(transcript, inner_initial_claim, batched_matrix, assignment)?;

    let mle_claim = ScaledMleEvaluationClaim::new(
        inner.sumcheck.eval_points.into_boxed_slice(),
        inner.batched_matrix_evaluation,
        inner.sumcheck.final_claim,
    );
    let proof = SpartanPiopProof {
        outer: outer.proof,
        inner: inner.sumcheck.proof,
    };

    Ok((proof, mle_claim))
}

/// Verifies both sumchecks and returns their terminal scaled assignment claim
/// `D(r_y) * h(r_y) = final_claim`.
pub fn verify_spartan_proof(
    transcript: &mut VerifierState<'_>,
    matrices: &PreparedConstraintMatrices,
    proof: &SpartanPiopProof<FqDefault>,
) -> Result<ScaledMleEvaluationClaim<FqDefault>, SpartanError> {
    let num_row_vars = matrices.num_row_vars();
    let num_column_vars = matrices.num_column_vars();
    transcript.public_message(matrices.digest());
    let tau = challenge_point(transcript, num_row_vars);
    let outer = proof
        .outer
        .verify(transcript, FqDefault::from(0u128), &tau)?;

    // The outer verifier absorbed the product evaluations before returning.
    let rho = challenge_fq(transcript);
    let inner_initial_claim =
        outer.az_mle_claim + rho * outer.bz_mle_claim + rho * rho * outer.cz_mle_claim;
    let (column_point, final_claim) =
        proof
            .inner
            .verify(transcript, inner_initial_claim, num_column_vars)?;
    let matrix_evaluation = matrices.evaluate_batched(&outer.eval_points, rho, &column_point)?;

    Ok(ScaledMleEvaluationClaim::new(
        column_point.into_boxed_slice(),
        matrix_evaluation,
        final_claim,
    ))
}

/// Verifies both sumchecks and checks their terminal claim against the complete
/// assignment.
///
/// `assignment` is the packed bit assignment `h = M(1 || f)`, not the raw
/// Boolean witness `f`. The eventual virtual opening protocol must enforce
/// that map from a commitment to `f`; this witness-aware path checks the
/// resulting assignment directly.
pub fn verify_spartan_with_mle_claim(
    transcript: &mut VerifierState<'_>,
    matrices: &PreparedConstraintMatrices,
    proof: &SpartanPiopProof<FqDefault>,
    mle_claim: &ScaledMleEvaluationClaim<FqDefault>,
    assignment: &PackedWitness,
) -> Result<(), SpartanError> {
    let assignment = build_assignment_mle(assignment, matrices.matrices().a.column_count())?;
    let expected_claim = verify_spartan_proof(transcript, matrices, proof)?;
    if mle_claim != &expected_claim {
        return Err(SpartanError::InvalidMleClaim);
    }
    mle_claim.nonsuccinct_verify(&assignment)?;

    Ok(())
}

fn challenge_point(transcript: &mut impl FqChallengeSource, num_vars: usize) -> Vec<FqDefault> {
    (0..num_vars).map(|_| challenge_fq(transcript)).collect()
}
