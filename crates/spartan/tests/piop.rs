use circuit::{
    constraints::{ConstraintMatrices, SparseBoolMatrix, SparseMatrix},
    witgen::PackedWitness,
};
use field::FqDefault;
use poly::{DenseMultilinearExtension, ScaledMleEvaluationClaim};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use spartan::{
    PreparedConstraintMatrices, R1csProductMles, SpartanError, SpartanMatrixError,
    build_assignment_mle, prove_spartan_piop, verify_spartan_proof, verify_spartan_with_mle_claim,
};
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
    let m =
        SparseBoolMatrix::try_from_rows(COLUMNS, (0..COLUMNS).map(|column| vec![column]).collect())
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
