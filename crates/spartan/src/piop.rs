//! Composition of Spartan's outer and inner sumchecks.

use circuit::witgen::PackedWitness;
use crypto_primitives::ConstField;
use poly::{
    DenseMultilinearExtension, MleClaimError, ScaledMleEvaluationClaim, make_equality_factors,
};
use transcript::{Encoding, ProverState, TranscriptChallenge, VerifierState};

use crate::matrix::{PreparedConstraintMatrices, SpartanMatrixError, build_assignment_mle};
use crate::sumcheck::{
    OuterSumcheckProof, R1csProductMles, SumcheckError, SumcheckProof, prove_inner_sumcheck,
    prove_outer_sumcheck,
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
///
/// When a protocol supports more than one choice of `F`, its transcript
/// session or instance must bind that choice so proofs from different fields
/// occupy distinct Fiat--Shamir domains.
#[tracing::instrument(name = "Prove Spartan", skip_all)]
pub fn prove_spartan_piop<F>(
    transcript: &mut ProverState,
    matrices: &PreparedConstraintMatrices<F>,
    products: &R1csProductMles<F>,
    assignment: &DenseMultilinearExtension<F>,
) -> Result<(SpartanPiopProof<F>, ScaledMleEvaluationClaim<F>), SpartanError>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
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
    let tau = (0..num_row_vars)
        .map(|_| transcript.squeeze::<F>())
        .collect::<Vec<_>>();
    let equality_factors =
        make_equality_factors(&tau).map_err(|_| SpartanMatrixError::InvalidMleOperation)?;
    let outer = prove_outer_sumcheck(transcript, F::ZERO, equality_factors, products)?;

    // The outer prover absorbed these evaluations before returning.
    let rho = transcript.squeeze::<F>();
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
#[tracing::instrument(name = "Verify Spartan", skip_all)]
pub fn verify_spartan_proof<F>(
    transcript: &mut VerifierState<'_>,
    matrices: &PreparedConstraintMatrices<F>,
    proof: &SpartanPiopProof<F>,
) -> Result<ScaledMleEvaluationClaim<F>, SpartanError>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
    let num_row_vars = matrices.num_row_vars();
    let num_column_vars = matrices.num_column_vars();
    transcript.public_message(matrices.digest());
    let tau = (0..num_row_vars)
        .map(|_| transcript.squeeze::<F>())
        .collect::<Vec<_>>();
    let outer = proof.outer.verify(transcript, F::ZERO, &tau)?;

    // The outer verifier absorbed the product evaluations before returning.
    let rho = transcript.squeeze::<F>();
    let inner_initial_claim =
        outer.az_mle_claim + rho * outer.bz_mle_claim + rho * rho * outer.cz_mle_claim;
    let (column_point, final_claim) = {
        let _span = tracing::info_span!("Verify inner sumcheck").entered();
        proof
            .inner
            .verify(transcript, inner_initial_claim, num_column_vars)?
    };
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
pub fn verify_spartan_with_mle_claim<F>(
    transcript: &mut VerifierState<'_>,
    matrices: &PreparedConstraintMatrices<F>,
    proof: &SpartanPiopProof<F>,
    mle_claim: &ScaledMleEvaluationClaim<F>,
    assignment: &PackedWitness,
) -> Result<(), SpartanError>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
    let assignment = build_assignment_mle::<F>(assignment, matrices.matrices().a.column_count())?;
    let expected_claim = verify_spartan_proof(transcript, matrices, proof)?;
    if mle_claim != &expected_claim {
        return Err(SpartanError::InvalidMleClaim);
    }
    mle_claim.nonsuccinct_verify(&assignment)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use circuit::{
        constraints::{ConstraintMatrices, SparseBoolMatrix, SparseMatrix},
        witgen::PackedWitness,
    };
    use crypto_primitives::ConstField;
    use field::{F128, FqDefault};
    use poly::DenseMultilinearExtension;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use transcript::{Encoding, TranscriptChallenge, build_prover, build_verifier};

    use super::{prove_spartan_piop, verify_spartan_with_mle_claim};
    use crate::matrix::{PreparedConstraintMatrices, build_assignment_mle};
    use crate::sumcheck::R1csProductMles;

    const FQ_SESSION: &[u8] = b"spartan/piop/random-r1cs/fq/v1";
    const F128_SESSION: &[u8] = b"spartan/piop/random-r1cs/f128/v1";
    const INSTANCE: &[u8] = b"five-rows-seven-columns";
    const ROWS: usize = 5;
    const COLUMNS: usize = 7;

    #[test]
    fn random_satisfying_r1cs_reduces_through_both_sumchecks() {
        check_random_satisfying_r1cs::<FqDefault>(FQ_SESSION);
        check_random_satisfying_r1cs::<F128>(F128_SESSION);
    }

    #[test]
    fn verifier_rejects_proof_with_unsatisfied_witness() {
        check_verifier_rejects_proof_with_unsatisfied_witness::<FqDefault>(FQ_SESSION);
        check_verifier_rejects_proof_with_unsatisfied_witness::<F128>(F128_SESSION);
    }

    fn check_verifier_rejects_proof_with_unsatisfied_witness<F>(session: &[u8])
    where
        F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
    {
        const SIZE: usize = 2;
        const UNSATISFIED_INSTANCE: &[u8] = b"unsatisfied-r1cs";

        let one = F::ONE;
        let zero = F::ZERO;
        let satisfying_assignment = [one, zero];
        let unsatisfied_assignment = [one, one];
        let a = SparseMatrix::try_from_rows(SIZE, vec![vec![(1, one)], vec![(1, one)]]).unwrap();
        let b = a.clone();
        let c = SparseMatrix::try_from_rows(SIZE, vec![Vec::new(), Vec::new()]).unwrap();
        let az = multiply(&a, &satisfying_assignment);
        let bz = multiply(&b, &satisfying_assignment);
        let cz = multiply(&c, &satisfying_assignment);
        let unsatisfied_az = multiply(&a, &unsatisfied_assignment);
        let unsatisfied_bz = multiply(&b, &unsatisfied_assignment);
        let unsatisfied_cz = multiply(&c, &unsatisfied_assignment);
        assert_ne!(unsatisfied_az[0] * unsatisfied_bz[0], unsatisfied_cz[0]);

        let m = SparseBoolMatrix::try_from_rows(SIZE, vec![vec![0], vec![1]]).unwrap();
        let matrices = PreparedConstraintMatrices::new(ConstraintMatrices { m, a, b, c }).unwrap();
        let products = R1csProductMles {
            az: padded_mle(az),
            bz: padded_mle(bz),
            cz: padded_mle(cz),
        };
        let satisfying_witness = PackedWitness::from_bits(&[true, false]);
        let assignment = build_assignment_mle::<F>(&satisfying_witness, SIZE).unwrap();

        let mut prover = build_prover(session, UNSATISFIED_INSTANCE);
        let (proof, claim) =
            prove_spartan_piop(&mut prover, &matrices, &products, &assignment).unwrap();
        let transcript_proof = prover.finish();

        let unsatisfied_witness = PackedWitness::from_bits(&[true, true]);
        let mut verifier = build_verifier(session, UNSATISFIED_INSTANCE, &transcript_proof);
        assert!(
            verify_spartan_with_mle_claim(
                &mut verifier,
                &matrices,
                &proof,
                &claim,
                &unsatisfied_witness,
            )
            .is_err()
        );
    }

    fn check_random_satisfying_r1cs<F>(session: &[u8])
    where
        F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
    {
        let mut rng = Pcg64::seed_from_u64(0x5a17_c0de);
        let mut assignment_bits = Vec::with_capacity(COLUMNS);
        assignment_bits.push(true);
        assignment_bits.extend((1..COLUMNS).map(|_| rng.random::<bool>()));
        let assignment_values: Vec<_> = assignment_bits
            .iter()
            .copied()
            .map(|bit| if bit { F::ONE } else { F::ZERO })
            .collect();

        let a = random_sparse_matrix::<F>(&mut rng);
        let b = random_sparse_matrix::<F>(&mut rng);
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
        // valid BitZ layout as well: h = M(1 || f).
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
        let assignment = build_assignment_mle::<F>(&packed_assignment, COLUMNS).unwrap();

        let mut prover = build_prover(session, INSTANCE);
        let (proof, claim) =
            prove_spartan_piop(&mut prover, &matrices, &products, &assignment).unwrap();
        assert_eq!(proof.outer.sumcheck.round_polynomials.len(), 3);
        assert_eq!(proof.inner.round_polynomials.len(), 3);
        let transcript_proof = prover.finish();

        let mut complete_verifier = build_verifier(session, INSTANCE, &transcript_proof);
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
    }

    fn random_sparse_matrix<F: ConstField + Copy>(rng: &mut Pcg64) -> SparseMatrix<F> {
        let rows = (0..ROWS)
            .map(|_| {
                // A nonzero constant-column entry keeps each row evaluation
                // nonzero for this small positive fixture.
                let mut entries = vec![(0, F::from(u128::from(rng.random::<u64>() % 9 + 1)))];
                entries.extend((1..COLUMNS).filter_map(|column| {
                    if rng.random::<bool>() {
                        Some((column, F::from(u128::from(rng.random::<u64>() % 9 + 1))))
                    } else {
                        None
                    }
                }));
                entries
            })
            .collect();
        SparseMatrix::try_from_rows(COLUMNS, rows).unwrap()
    }

    fn multiply<F: ConstField + Copy>(matrix: &SparseMatrix<F>, assignment: &[F]) -> Vec<F> {
        matrix
            .rows()
            .iter()
            .map(|row| {
                row.entries()
                    .iter()
                    .fold(F::ZERO, |sum, &(column, coefficient)| {
                        sum + coefficient * assignment[column]
                    })
            })
            .collect()
    }

    fn padded_mle<F: ConstField + Copy>(values: Vec<F>) -> DenseMultilinearExtension<F> {
        let padded_len = values.len().max(1).next_power_of_two();
        let num_vars = padded_len.ilog2() as usize;
        let mut evaluations = values;
        evaluations.resize(padded_len, F::ZERO);
        DenseMultilinearExtension::from_evaluations(num_vars, evaluations).unwrap()
    }
}
