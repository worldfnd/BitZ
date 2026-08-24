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

#[cfg(test)]
mod tests {
    use super::*;
    use circuit::constraints::{ConstraintMatrices, SparseBoolMatrix, SparseMatrix};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use transcript::{build_prover, build_verifier};

    const SESSION: &[u8] = b"spartan/piop/random-r1cs/v1";
    const INSTANCE: &[u8] = b"five-rows-seven-columns";
    const ROWS: usize = 5;
    const COLUMNS: usize = 7;

    #[test]
    fn random_satisfying_r1cs_reduces_through_both_sumchecks() {
        let mut rng = Pcg64::seed_from_u64(0x5a17_c0de);
        let mut assignment_bits = Vec::with_capacity(COLUMNS);
        assignment_bits.push(true);
        assignment_bits.extend((1..COLUMNS).map(|_| rng.random::<bool>()));
        let assignment_values: Vec<_> = assignment_bits
            .iter()
            .copied()
            .map(FqDefault::from)
            .collect();

        let a = random_sparse_matrix(&mut rng);
        let b = random_sparse_matrix(&mut rng);
        let az = multiply(&a, &assignment_values);
        let bz = multiply(&b, &assignment_values);

        // Since h[0] = 1, setting C[i,0] = (A_i h)(B_i h) constructs a
        // satisfying row without constraining the remaining C coefficients.
        let c = SparseMatrix::try_from_rows(
            COLUMNS,
            az.iter()
                .zip(&bz)
                .map(|(&a, &b)| vec![(0, a * b)])
                .collect(),
        )
        .unwrap();
        let cz = multiply(&c, &assignment_values);

        for row in 0..ROWS {
            assert_eq!(az[row] * bz[row], cz[row]);
        }

        // The PIOP only consumes h and A/B/C. This identity M gives the fixture a
        // valid F2Z layout as well: h = M(1 || f).
        let m = SparseBoolMatrix::try_from_rows(
            COLUMNS,
            (0..COLUMNS).map(|column| vec![column]).collect(),
        )
        .unwrap();
        let matrices = PreparedConstraintMatrices::new(ConstraintMatrices { m, a, b, c }).unwrap();
        let products = R1csProductMles {
            az: padded_mle(az),
            bz: padded_mle(bz),
            cz: padded_mle(cz),
        };
        let packed_assignment = PackedWitness::from_bits(&assignment_bits);
        let assignment = build_assignment_mle(&packed_assignment, COLUMNS).unwrap();

        let mut prover = build_prover(SESSION, INSTANCE);
        let (proof, claim) =
            prove_spartan_piop(&mut prover, &matrices, products, assignment.clone()).unwrap();
        assert_eq!(proof.outer.sumcheck.round_polynomials.len(), 3);
        assert_eq!(proof.inner.round_polynomials.len(), 3);
        let transcript_proof = prover.finish();

        let mut verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        let derived_claim = verify_spartan_proof(&mut verifier, &matrices, &proof).unwrap();
        assert_eq!(derived_claim, claim);
        verifier.check_eof().unwrap();

        let mut complete_verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        verify_spartan_with_mle_claim(
            &mut complete_verifier,
            &matrices,
            &proof,
            &claim,
            &packed_assignment,
        )
        .unwrap();
        let assignment_evaluation = assignment.evaluate(claim.point()).unwrap();
        assert_eq!(claim.value(), claim.scale() * assignment_evaluation);
        complete_verifier.check_eof().unwrap();

        // Changing the terminal claim must fail even if D(r_y) happens to be zero.
        let invalid_claim = ScaledMleEvaluationClaim::new(
            claim.point().to_vec().into_boxed_slice(),
            claim.scale(),
            claim.value() + fq(1),
        );
        let mut invalid_claim_verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        assert_eq!(
            verify_spartan_with_mle_claim(
                &mut invalid_claim_verifier,
                &matrices,
                &proof,
                &invalid_claim,
                &packed_assignment,
            ),
            Err(SpartanError::InvalidMleClaim)
        );

        // A changed PIOP round is rejected by the composed verifier.
        let mut invalid_proof = proof.clone();
        invalid_proof.inner.round_polynomials[0][0] += fq(1);
        let mut invalid_proof_verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        assert!(
            verify_spartan_with_mle_claim(
                &mut invalid_proof_verifier,
                &matrices,
                &invalid_proof,
                &claim,
                &packed_assignment,
            )
            .is_err()
        );

        // The public matrix digest is absorbed before tau, so the same proof
        // cannot be replayed against a changed statement.
        let mut changed_m_rows: Vec<_> = (0..COLUMNS).map(|column| vec![column]).collect();
        changed_m_rows[1] = vec![0, 1];
        let changed_matrices = ConstraintMatrices {
            m: SparseBoolMatrix::try_from_rows(COLUMNS, changed_m_rows).unwrap(),
            a: matrices.matrices().a.clone(),
            b: matrices.matrices().b.clone(),
            c: matrices.matrices().c.clone(),
        };
        let changed_matrices = PreparedConstraintMatrices::new(changed_matrices).unwrap();
        let mut changed_statement_verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        assert!(
            verify_spartan_with_mle_claim(
                &mut changed_statement_verifier,
                &changed_matrices,
                &proof,
                &claim,
                &packed_assignment,
            )
            .is_err()
        );

        let mut wrong_instance_verifier =
            build_verifier(SESSION, b"different-instance", &transcript_proof);
        assert!(
            verify_spartan_with_mle_claim(
                &mut wrong_instance_verifier,
                &matrices,
                &proof,
                &claim,
                &packed_assignment,
            )
            .is_err()
        );

        let mut false_constant_bits = assignment_bits.clone();
        false_constant_bits[0] = false;
        assert_eq!(
            build_assignment_mle(&PackedWitness::from_bits(&false_constant_bits), COLUMNS),
            Err(SpartanMatrixError::InvalidAssignmentConstant)
        );

        // The temporary witness-aware endpoint also rejects a changed assignment
        // bit whose equality weight is nonzero.
        assert_ne!(claim.scale(), fq(0));
        let equality_weights = poly::eq_table(claim.point());
        let changed_index = equality_weights[..COLUMNS]
            .iter()
            .position(|weight| *weight != fq(0))
            .expect("the deterministic challenge addresses a logical assignment entry");
        let mut invalid_assignment_bits = assignment_bits;
        invalid_assignment_bits[changed_index] ^= true;
        let invalid_assignment = PackedWitness::from_bits(&invalid_assignment_bits);
        let mut invalid_assignment_verifier = build_verifier(SESSION, INSTANCE, &transcript_proof);
        assert!(
            verify_spartan_with_mle_claim(
                &mut invalid_assignment_verifier,
                &matrices,
                &proof,
                &claim,
                &invalid_assignment,
            )
            .is_err()
        );
    }

    fn random_sparse_matrix(rng: &mut Pcg64) -> SparseMatrix<FqDefault> {
        let rows = (0..ROWS)
            .map(|_| {
                // A nonzero constant-column entry keeps each row evaluation
                // nonzero for this small positive fixture.
                let mut entries = vec![(0, fq(u128::from(rng.random::<u64>() % 9 + 1)))];
                entries.extend((1..COLUMNS).filter_map(|column| {
                    if rng.random::<bool>() {
                        Some((column, fq(u128::from(rng.random::<u64>() % 9 + 1))))
                    } else {
                        None
                    }
                }));
                entries
            })
            .collect();
        SparseMatrix::try_from_rows(COLUMNS, rows).unwrap()
    }

    fn multiply(matrix: &SparseMatrix<FqDefault>, assignment: &[FqDefault]) -> Vec<FqDefault> {
        matrix
            .rows()
            .iter()
            .map(|row| {
                row.entries()
                    .iter()
                    .fold(fq(0), |sum, &(column, coefficient)| {
                        sum + coefficient * assignment[column]
                    })
            })
            .collect()
    }

    fn padded_mle(values: Vec<FqDefault>) -> DenseMultilinearExtension<FqDefault> {
        let padded_len = values.len().max(1).next_power_of_two();
        let num_vars = padded_len.ilog2() as usize;
        let mut evaluations = values;
        evaluations.resize(padded_len, fq(0));
        DenseMultilinearExtension::from_evaluations(num_vars, evaluations).unwrap()
    }

    fn fq(value: u128) -> FqDefault {
        FqDefault::from(value)
    }
}
