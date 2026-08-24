//! R1CS matrix preparation and the sparse kernels used by Spartan.

use circuit::constraints::{ConstraintMatrices, SparseMatrix};
use circuit::matrix_products::{IntegerProducts, ModularVector, RuntimeModulus};
use circuit::witgen::PackedWitness;
use field::{FqDefault, Q100};
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, ToPrimitive};
use poly::DenseMultilinearExtension;
use sha2::{Digest, Sha256};
use transcript::Encoding;

use crate::sumcheck::R1csProductMles;

/// Failures while preparing or evaluating Spartan's R1CS matrices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpartanMatrixError {
    InvalidR1csShape,
    InvalidProductLength { expected: usize, actual: usize },
    InvalidAssignmentLength { expected: usize, actual: usize },
    InvalidAssignmentConstant,
    InvalidRowPointLength { expected: usize, actual: usize },
    InvalidColumnPointLength { expected: usize, actual: usize },
    DomainTooLarge,
    InvalidModulus,
    InvalidMleOperation,
}

/// Immutable field-valued constraint matrices prepared for repeated Spartan
/// proofs and verification.
///
/// Shape validation, Boolean-domain sizing, and canonical statement hashing
/// are performed once during construction rather than inside the prover or
/// verifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedConstraintMatrices {
    matrices: ConstraintMatrices<FqDefault>,
    digest: [u8; 32],
    num_row_vars: usize,
    num_column_vars: usize,
}

impl PreparedConstraintMatrices {
    pub fn new(matrices: ConstraintMatrices<FqDefault>) -> Result<Self, SpartanMatrixError> {
        let (num_row_vars, num_column_vars) = r1cs_num_vars(&matrices)?;
        let digest = constraint_matrix_digest(&matrices)?;

        Ok(Self {
            matrices,
            digest,
            num_row_vars,
            num_column_vars,
        })
    }

    pub fn matrices(&self) -> &ConstraintMatrices<FqDefault> {
        &self.matrices
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub const fn num_row_vars(&self) -> usize {
        self.num_row_vars
    }

    pub const fn num_column_vars(&self) -> usize {
        self.num_column_vars
    }
}

/// Spartan-specific operations on the field-valued constraint matrices.
pub trait SpartanMatrixOperations {
    /// Constructs
    ///
    /// `D(j) = sum_i eq(i,r_x) (A[i,j] + rho B[i,j] + rho^2 C[i,j])`.
    fn bind_and_batch(
        &self,
        row_point: &[FqDefault],
        rho: FqDefault,
    ) -> Result<DenseMultilinearExtension<FqDefault>, SpartanMatrixError>;

    /// Directly evaluates
    ///
    /// `D(r_y) = A(r_x,r_y) + rho B(r_x,r_y) + rho^2 C(r_x,r_y)`.
    ///
    /// This deliberately does not call [`Self::bind_and_batch`], keeping the
    /// verifier path independent from the prover's dense-table construction.
    fn evaluate_batched(
        &self,
        row_point: &[FqDefault],
        rho: FqDefault,
        column_point: &[FqDefault],
    ) -> Result<FqDefault, SpartanMatrixError>;
}

/// Reduces a signed integer canonically modulo Q100.
pub fn bigint_to_fq(value: &BigInt) -> FqDefault {
    let modulus = BigInt::from(Q100);
    let mut reduced = value % &modulus;
    if reduced.is_negative() {
        reduced += modulus;
    }
    FqDefault::from(
        reduced
            .to_u128()
            .expect("a canonical Q100 residue always fits a u128"),
    )
}

/// Reduces exact `Ah`, `Bh`, and `Ch` values modulo Q100 and pads their row
/// tables with trailing zeros to the next power of two.
pub fn build_product_mles(
    products: &IntegerProducts,
    expected_rows: usize,
) -> Result<R1csProductMles<FqDefault>, SpartanMatrixError> {
    for actual in [
        products.a_mw.len(),
        products.b_mw.len(),
        products.c_mw.len(),
    ] {
        if actual != expected_rows {
            return Err(SpartanMatrixError::InvalidProductLength {
                expected: expected_rows,
                actual,
            });
        }
    }

    let modulus = RuntimeModulus::<2>::new(BigUint::from(Q100))
        .map_err(|_| SpartanMatrixError::InvalidModulus)?;
    let reduced = products.reduce_parallel(&modulus);
    let num_vars = padded_num_vars(expected_rows)?;

    Ok(R1csProductMles {
        az: modular_vector_mle(&reduced.a_mw, num_vars)?,
        bz: modular_vector_mle(&reduced.b_mw, num_vars)?,
        cz: modular_vector_mle(&reduced.c_mw, num_vars)?,
    })
}

/// Converts the packed assignment `h = M(1 || f)` into Q100 and pads it with
/// trailing zeros to the next power-of-two column domain. The first bit must
/// be the R1CS constant one.
pub fn build_assignment_mle(
    assignment: &PackedWitness,
    expected_columns: usize,
) -> Result<DenseMultilinearExtension<FqDefault>, SpartanMatrixError> {
    if assignment.bit_len() != expected_columns {
        return Err(SpartanMatrixError::InvalidAssignmentLength {
            expected: expected_columns,
            actual: assignment.bit_len(),
        });
    }
    if expected_columns == 0 || !assignment.bit(0) {
        return Err(SpartanMatrixError::InvalidAssignmentConstant);
    }

    let num_vars = padded_num_vars(expected_columns)?;
    let padded_len = 1usize << num_vars;
    let zero = FqDefault::from(0u128);
    let mut evaluations = Vec::with_capacity(padded_len);
    evaluations
        .extend((0..assignment.bit_len()).map(|index| FqDefault::from(assignment.bit(index))));
    evaluations.resize(padded_len, zero);

    DenseMultilinearExtension::from_evaluations(num_vars, evaluations)
        .map_err(|_| SpartanMatrixError::InvalidMleOperation)
}

impl SpartanMatrixOperations for ConstraintMatrices<FqDefault> {
    fn bind_and_batch(
        &self,
        row_point: &[FqDefault],
        rho: FqDefault,
    ) -> Result<DenseMultilinearExtension<FqDefault>, SpartanMatrixError> {
        let (num_row_vars, num_column_vars) = r1cs_num_vars(self)?;
        bind_and_batch_with_num_vars(self, row_point, rho, num_row_vars, num_column_vars)
    }

    fn evaluate_batched(
        &self,
        row_point: &[FqDefault],
        rho: FqDefault,
        column_point: &[FqDefault],
    ) -> Result<FqDefault, SpartanMatrixError> {
        let (num_row_vars, num_column_vars) = r1cs_num_vars(self)?;
        evaluate_batched_with_num_vars(
            self,
            row_point,
            rho,
            column_point,
            num_row_vars,
            num_column_vars,
        )
    }
}

impl SpartanMatrixOperations for PreparedConstraintMatrices {
    fn bind_and_batch(
        &self,
        row_point: &[FqDefault],
        rho: FqDefault,
    ) -> Result<DenseMultilinearExtension<FqDefault>, SpartanMatrixError> {
        bind_and_batch_with_num_vars(
            &self.matrices,
            row_point,
            rho,
            self.num_row_vars,
            self.num_column_vars,
        )
    }

    fn evaluate_batched(
        &self,
        row_point: &[FqDefault],
        rho: FqDefault,
        column_point: &[FqDefault],
    ) -> Result<FqDefault, SpartanMatrixError> {
        evaluate_batched_with_num_vars(
            &self.matrices,
            row_point,
            rho,
            column_point,
            self.num_row_vars,
            self.num_column_vars,
        )
    }
}

fn bind_and_batch_with_num_vars(
    matrices: &ConstraintMatrices<FqDefault>,
    row_point: &[FqDefault],
    rho: FqDefault,
    num_row_vars: usize,
    num_column_vars: usize,
) -> Result<DenseMultilinearExtension<FqDefault>, SpartanMatrixError> {
    if row_point.len() != num_row_vars {
        return Err(SpartanMatrixError::InvalidRowPointLength {
            expected: num_row_vars,
            actual: row_point.len(),
        });
    }

    let zero = FqDefault::from(0u128);
    let one = FqDefault::from(1u128);
    let row_weights = poly::eq_table(row_point);
    let mut evaluations = vec![zero; 1usize << num_column_vars];

    for (matrix, batch_scale) in [
        (&matrices.a, one),
        (&matrices.b, rho),
        (&matrices.c, rho * rho),
    ] {
        for (row_index, row) in matrix.rows().iter().enumerate() {
            let row_scale = row_weights[row_index] * batch_scale;
            for &(column, coefficient) in row.entries() {
                evaluations[column] += row_scale * coefficient;
            }
        }
    }

    DenseMultilinearExtension::from_evaluations(num_column_vars, evaluations)
        .map_err(|_| SpartanMatrixError::InvalidMleOperation)
}

fn evaluate_batched_with_num_vars(
    matrices: &ConstraintMatrices<FqDefault>,
    row_point: &[FqDefault],
    rho: FqDefault,
    column_point: &[FqDefault],
    num_row_vars: usize,
    num_column_vars: usize,
) -> Result<FqDefault, SpartanMatrixError> {
    if row_point.len() != num_row_vars {
        return Err(SpartanMatrixError::InvalidRowPointLength {
            expected: num_row_vars,
            actual: row_point.len(),
        });
    }
    if column_point.len() != num_column_vars {
        return Err(SpartanMatrixError::InvalidColumnPointLength {
            expected: num_column_vars,
            actual: column_point.len(),
        });
    }

    let zero = FqDefault::from(0u128);
    let one = FqDefault::from(1u128);
    let row_weights = poly::eq_table(row_point);
    let column_weights = poly::eq_table(column_point);
    let mut evaluation = zero;

    for (matrix, batch_scale) in [
        (&matrices.a, one),
        (&matrices.b, rho),
        (&matrices.c, rho * rho),
    ] {
        for (row_index, row) in matrix.rows().iter().enumerate() {
            for &(column, coefficient) in row.entries() {
                evaluation +=
                    batch_scale * row_weights[row_index] * column_weights[column] * coefficient;
            }
        }
    }

    Ok(evaluation)
}

pub(crate) fn r1cs_num_vars(
    matrices: &ConstraintMatrices<FqDefault>,
) -> Result<(usize, usize), SpartanMatrixError> {
    matrices
        .validate_shape()
        .map_err(|_| SpartanMatrixError::InvalidR1csShape)?;
    let rows = matrices.a.row_count();
    let columns = matrices.a.column_count();

    Ok((padded_num_vars(rows)?, padded_num_vars(columns)?))
}

/// Canonically commits the complete public matrix statement before any
/// Fiat--Shamir challenge is sampled.
pub(crate) fn constraint_matrix_digest(
    matrices: &ConstraintMatrices<FqDefault>,
) -> Result<[u8; 32], SpartanMatrixError> {
    matrices
        .validate_shape()
        .map_err(|_| SpartanMatrixError::InvalidR1csShape)?;

    let mut hash = Sha256::new();
    hash.update(b"f2z/spartan/q100-constraint-matrices/v1");

    hash.update(b"M");
    hash_usize(&mut hash, matrices.m.row_count())?;
    hash_usize(&mut hash, matrices.m.column_count())?;
    for row in matrices.m.rows() {
        hash_usize(&mut hash, row.positions().len())?;
        for &column in row.positions() {
            hash_usize(&mut hash, column)?;
        }
    }

    for (label, matrix) in [
        (b"A", &matrices.a),
        (b"B", &matrices.b),
        (b"C", &matrices.c),
    ] {
        hash.update(label);
        hash_sparse_matrix(&mut hash, matrix)?;
    }

    Ok(hash.finalize().into())
}

fn hash_sparse_matrix(
    hash: &mut Sha256,
    matrix: &SparseMatrix<FqDefault>,
) -> Result<(), SpartanMatrixError> {
    hash_usize(hash, matrix.row_count())?;
    hash_usize(hash, matrix.column_count())?;
    for row in matrix.rows() {
        hash_usize(hash, row.entries().len())?;
        for &(column, coefficient) in row.entries() {
            hash_usize(hash, column)?;
            hash.update(coefficient.encode().as_ref());
        }
    }
    Ok(())
}

fn hash_usize(hash: &mut Sha256, value: usize) -> Result<(), SpartanMatrixError> {
    let value = u64::try_from(value).map_err(|_| SpartanMatrixError::DomainTooLarge)?;
    hash.update(value.to_le_bytes());
    Ok(())
}

pub(crate) fn padded_num_vars(logical_len: usize) -> Result<usize, SpartanMatrixError> {
    logical_len
        .max(1)
        .checked_next_power_of_two()
        .map(|len| len.ilog2() as usize)
        .ok_or(SpartanMatrixError::DomainTooLarge)
}

fn modular_vector_mle(
    values: &ModularVector<2>,
    num_vars: usize,
) -> Result<DenseMultilinearExtension<FqDefault>, SpartanMatrixError> {
    let zero = FqDefault::from(0u128);
    let padded_len = 1usize << num_vars;
    let mut evaluations: Vec<_> = values
        .values()
        .iter()
        .map(|&[low, high]| FqDefault::from(u128::from(low) | (u128::from(high) << 64)))
        .collect();
    evaluations.resize(padded_len, zero);

    DenseMultilinearExtension::from_evaluations(num_vars, evaluations)
        .map_err(|_| SpartanMatrixError::InvalidMleOperation)
}
