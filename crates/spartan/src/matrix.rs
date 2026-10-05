//! R1CS matrix preparation and the sparse kernels used by Spartan.

use circuit::constraints::{ConstraintMatrices, SparseBoolMatrix, SparseMatrix};
use circuit::matrix_products::{IntegerProducts, ModularVector, RuntimeModulus};
use circuit::witgen::PackedWitness;
use circuit::{BitWidth, IntoWords};
use common::{BitzClaimField, BitzField};
use poly::DenseMultilinearExtension;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::Arc;
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

/// Constraint matrices over their coefficient ring, prepared once for every
/// proof: shape validated, Boolean domains sized, nonzeros chunked by column.
/// None of that depends on a modulus, so [`Self::project`] shares it and
/// only reduces the coefficients.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedIntegerMatrices<R> {
    matrices: ConstraintMatrices<R>,
    /// Nonzeros of `a`, `b` and `c` grouped by column chunk.
    column_chunks: Arc<[ColumnChunkIndex; 3]>,
    num_row_vars: usize,
    num_column_vars: usize,
}

/// Immutable field-valued constraint matrices prepared for repeated Spartan
/// proofs and verification: `A`, `B` and `C` over the field, their column
/// chunking, the Boolean domain sizes and the canonical statement digest.
///
/// Built by [`PreparedIntegerMatrices::project`] under a drawn modulus, or by
/// [`Self::new`] from matrices already over the field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedConstraintMatrices<F> {
    a: SparseMatrix<F>,
    b: SparseMatrix<F>,
    c: SparseMatrix<F>,
    column_chunks: Arc<[ColumnChunkIndex; 3]>,
    digest: [u8; 32],
    num_row_vars: usize,
    num_column_vars: usize,
}

/// `bind_and_batch` splits the column domain into chunks of `2^16` columns and
/// accumulates each chunk on its own thread.
/// A chunk's slice of the dense table fits in cache.
const BIND_CHUNK_COLUMN_VARS: usize = 16;

/// The nonzeros of one sparse matrix grouped by column chunk.
///
/// Chunk `c` owns columns `[c * chunk_len, (c + 1) * chunk_len)` and lists
/// the entries of every row that fall into it, in row order. The thread
/// accumulating chunk `c` reads only these entries and writes only these
/// columns, so threads never share a column.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ColumnChunkIndex {
    chunk_len: usize,
    spans: Vec<Vec<RowSpan>>,
}

/// The entries `[start, end)` of row `row`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowSpan {
    row: usize,
    start: usize,
    end: usize,
}

impl ColumnChunkIndex {
    /// Groups `matrix` into `chunk_count` chunks of `chunk_len` columns.
    fn new<C>(matrix: &SparseMatrix<C>, chunk_len: usize, chunk_count: usize) -> Self {
        debug_assert!(chunk_len.is_power_of_two());
        debug_assert!(chunk_len * chunk_count >= matrix.column_count());

        let mut spans = vec![Vec::new(); chunk_count];
        for (row, entries) in matrix.rows().iter().enumerate() {
            let entries = entries.entries();
            let mut start = 0;
            while start < entries.len() {
                // Columns increase within a row, so a chunk's entries are a
                // contiguous run.
                let chunk = entries[start].0 / chunk_len;
                let end = start
                    + entries[start..].partition_point(|(column, _)| column / chunk_len == chunk);
                spans[chunk].push(RowSpan { row, start, end });
                start = end;
            }
        }

        Self { chunk_len, spans }
    }
}

/// Groups `a`, `b` and `c` by column chunk over the padded column domain.
fn column_chunks<C>(
    matrices: &ConstraintMatrices<C>,
    num_column_vars: usize,
) -> [ColumnChunkIndex; 3] {
    let num_columns = 1_usize << num_column_vars;
    let chunk_len = num_columns.min(1_usize << BIND_CHUNK_COLUMN_VARS);
    let chunk_count = num_columns / chunk_len;
    [&matrices.a, &matrices.b, &matrices.c]
        .map(|matrix| ColumnChunkIndex::new(matrix, chunk_len, chunk_count))
}

impl<R: Send + Sync> PreparedIntegerMatrices<R> {
    pub fn new(matrices: ConstraintMatrices<R>) -> Result<Self, SpartanMatrixError> {
        let (num_row_vars, num_column_vars) = r1cs_num_vars(&matrices)?;
        let column_chunks = Arc::new(column_chunks(&matrices, num_column_vars));
        Ok(Self {
            matrices,
            column_chunks,
            num_row_vars,
            num_column_vars,
        })
    }

    pub fn matrices(&self) -> &ConstraintMatrices<R> {
        &self.matrices
    }

    /// The matrices with `map` applied to every coefficient of `A`, `B` and
    /// `C`: the per-proof work. The chunking is shared; the digest is over the
    /// projected coefficients, as in [`PreparedConstraintMatrices::new`].
    pub fn project<F, M>(&self, map: M) -> Result<PreparedConstraintMatrices<F>, SpartanMatrixError>
    where
        F: BitzField,
        M: Fn(&R) -> F + Copy + Send + Sync,
    {
        let [a, b, c] = [&self.matrices.a, &self.matrices.b, &self.matrices.c]
            .map(|matrix| matrix.map_values_ref(map));
        let digest = constraint_matrix_digest(&self.matrices.m, [&a, &b, &c])?;
        Ok(PreparedConstraintMatrices {
            a,
            b,
            c,
            column_chunks: Arc::clone(&self.column_chunks),
            digest,
            num_row_vars: self.num_row_vars,
            num_column_vars: self.num_column_vars,
        })
    }
}

impl<F: BitzField> PreparedConstraintMatrices<F> {
    /// Prepares matrices already over the field, without copying them.
    pub fn new(matrices: ConstraintMatrices<F>) -> Result<Self, SpartanMatrixError> {
        let (num_row_vars, num_column_vars) = r1cs_num_vars(&matrices)?;
        let column_chunks = Arc::new(column_chunks(&matrices, num_column_vars));
        let ConstraintMatrices { m, a, b, c } = matrices;
        let digest = constraint_matrix_digest(&m, [&a, &b, &c])?;
        Ok(Self {
            a,
            b,
            c,
            column_chunks,
            digest,
            num_row_vars,
            num_column_vars,
        })
    }

    /// Returns human-readable info about R1CS matrices and their nonzero entries.
    pub fn short_debug_info(&self) -> String {
        let nonzeros: usize = self
            .matrices()
            .iter()
            .flat_map(|matrix| matrix.rows())
            .map(|row| row.entries().len())
            .sum();
        format!(
            "{} r1cs rows -> 2^{}, {} h entries -> 2^{}, {nonzeros} nonzeros",
            self.row_count(),
            self.num_row_vars(),
            self.column_count(),
            self.num_column_vars(),
        )
    }

    /// `A`, `B` and `C`.
    pub fn matrices(&self) -> [&SparseMatrix<F>; 3] {
        [&self.a, &self.b, &self.c]
    }

    /// Number of R1CS rows before padding.
    pub fn row_count(&self) -> usize {
        self.a.row_count()
    }

    /// Number of assignment entries before padding, the constant included.
    pub const fn column_count(&self) -> usize {
        self.a.column_count()
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

const PRIME_LIMBS: usize = 2;

/// Reduces exact `Ah`, `Bh`, and `Ch` values modulo `F::modulus` and pads their row
/// tables with trailing zeros to the next power of two.
pub fn build_product_mles<F>(
    products: &IntegerProducts,
    expected_rows: usize,
) -> Result<R1csProductMles<F>, SpartanMatrixError>
where
    F: BitzClaimField,
    F::Integer: BitWidth + IntoWords,
    Vec<F>: for<'a> From<&'a ModularVector<PRIME_LIMBS>>,
{
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

    // We cannot define RuntimeModulus<F::NUM_WORDS> in stable Rust, so
    // use Vec<F>: for<'a> From<&'a ModularVector<PRIME_LIMBS>> as a workaround
    let modulus = RuntimeModulus::<PRIME_LIMBS>::new(F::modulus())
        .map_err(|_| SpartanMatrixError::InvalidModulus)?;
    let reduced = products.reduce_parallel(&modulus);
    let num_vars = padded_num_vars(expected_rows)?;

    Ok(R1csProductMles {
        az: modular_vector_mle(&reduced.a_mw, num_vars)?,
        bz: modular_vector_mle(&reduced.b_mw, num_vars)?,
        cz: modular_vector_mle(&reduced.c_mw, num_vars)?,
    })
}

fn modular_vector_mle<F, const PRIME_LIMBS: usize>(
    values: &ModularVector<PRIME_LIMBS>,
    num_vars: usize,
) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError>
where
    F: BitzClaimField,
    Vec<F>: for<'a> From<&'a ModularVector<PRIME_LIMBS>>,
{
    let zero = F::zero();
    let padded_len = 1usize << num_vars;
    let mut evaluations: Vec<F> = values.into();
    evaluations.resize(padded_len, zero);

    DenseMultilinearExtension::from_evaluations(num_vars, evaluations)
        .map_err(|_| SpartanMatrixError::InvalidMleOperation)
}

/// Converts the packed assignment `h = M(1 || f)` into the selected field and
/// pads it with trailing zeros to the next power-of-two column domain. The
/// first bit must be the R1CS constant one.
pub fn build_assignment_mle<F: BitzField>(
    assignment: &PackedWitness,
    expected_columns: usize,
) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError> {
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
    let mut evaluations = Vec::with_capacity(padded_len);
    evaluations.extend((0..assignment.bit_len()).map(|index| {
        if assignment.bit(index) {
            F::one()
        } else {
            F::zero()
        }
    }));
    evaluations.resize(padded_len, F::zero());

    DenseMultilinearExtension::from_evaluations(num_vars, evaluations)
        .map_err(|_| SpartanMatrixError::InvalidMleOperation)
}

impl<F: BitzField> PreparedConstraintMatrices<F> {
    /// Constructs
    ///
    /// `D(j) = sum_i eq(i,r_x) (A[i,j] + rho B[i,j] + rho^2 C[i,j])`
    ///
    /// as a dense table over the column domain, one column chunk per task.
    #[tracing::instrument(name = "Bind Spartan matrices", skip_all)]
    pub fn bind_and_batch(
        &self,
        row_point: &[F],
        rho: F,
    ) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError> {
        bind_and_batch_with_num_vars(
            self.matrices(),
            &self.column_chunks,
            row_point,
            rho,
            self.num_row_vars,
            self.num_column_vars,
        )
    }

    /// Directly evaluates
    ///
    /// `D(r_y) = A(r_x,r_y) + rho B(r_x,r_y) + rho^2 C(r_x,r_y)`.
    ///
    /// This deliberately does not call [`Self::bind_and_batch`], keeping the
    /// verifier path independent from the prover's dense-table construction.
    #[tracing::instrument(name = "Evaluate Spartan matrices", level = "debug", skip_all)]
    pub fn evaluate_batched(
        &self,
        row_point: &[F],
        rho: F,
        column_point: &[F],
    ) -> Result<F, SpartanMatrixError> {
        evaluate_batched_with_num_vars(
            self.matrices(),
            &self.column_chunks,
            row_point,
            rho,
            column_point,
            self.num_row_vars,
            self.num_column_vars,
        )
    }
}

fn bind_and_batch_with_num_vars<F: BitzField>(
    matrices: [&SparseMatrix<F>; 3],
    column_chunks: &[ColumnChunkIndex; 3],
    row_point: &[F],
    rho: F,
    num_row_vars: usize,
    num_column_vars: usize,
) -> Result<DenseMultilinearExtension<F>, SpartanMatrixError> {
    if row_point.len() != num_row_vars {
        return Err(SpartanMatrixError::InvalidRowPointLength {
            expected: num_row_vars,
            actual: row_point.len(),
        });
    }

    let row_weights = poly::eq_table(row_point);
    let batched = [
        (matrices[0], &column_chunks[0], F::one()),
        (matrices[1], &column_chunks[1], rho),
        (matrices[2], &column_chunks[2], rho * rho),
    ];
    let chunk_len = column_chunks[0].chunk_len;

    // Every task owns one column chunk of the table and touches only the
    // nonzeros landing in it: no two tasks write the same column, and each
    // task's writes stay within a cache-sized slice.
    let mut evaluations: Vec<F> =
        rayon::iter::repeat_n(F::zero(), 1usize << num_column_vars).collect();
    evaluations
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(chunk, output)| {
            let base = chunk * chunk_len;
            for (matrix, index, batch_scale) in batched {
                for span in &index.spans[chunk] {
                    let row_scale = row_weights[span.row] * batch_scale;
                    let entries = &matrix.rows()[span.row].entries()[span.start..span.end];
                    for &(column, coefficient) in entries {
                        output[column - base] += row_scale * coefficient;
                    }
                }
            }
        });

    DenseMultilinearExtension::from_evaluations(num_column_vars, evaluations)
        .map_err(|_| SpartanMatrixError::InvalidMleOperation)
}

fn evaluate_batched_with_num_vars<F: BitzField>(
    matrices: [&SparseMatrix<F>; 3],
    column_chunks: &[ColumnChunkIndex; 3],
    row_point: &[F],
    rho: F,
    column_point: &[F],
    num_row_vars: usize,
    num_column_vars: usize,
) -> Result<F, SpartanMatrixError> {
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

    let row_weights = poly::eq_table(row_point);
    let batched = [
        (matrices[0], &column_chunks[0], F::one()),
        (matrices[1], &column_chunks[1], rho),
        (matrices[2], &column_chunks[2], rho * rho),
    ];

    // All tasks share one cache-sized `low` table and a per-chunk factor.
    // Each task reduces one column chunk of nonzeros to a single field element.
    let chunk_len = column_chunks[0].chunk_len;
    let (low_point, high_point) = column_point.split_at(chunk_len.ilog2() as usize);
    let low_weights = poly::eq_table(low_point);
    let high_weights = poly::eq_table(high_point);
    debug_assert_eq!(high_weights.len(), column_chunks[0].spans.len());

    let evaluation = high_weights
        .par_iter()
        .enumerate()
        .map(|(chunk, &chunk_weight)| {
            let base = chunk * chunk_len;
            let mut chunk_sum = F::zero();
            for (matrix, index, batch_scale) in batched {
                let mut matrix_sum = F::zero();
                for span in &index.spans[chunk] {
                    let entries = &matrix.rows()[span.row].entries()[span.start..span.end];
                    let span_sum = entries
                        .iter()
                        .fold(F::zero(), |sum, &(column, coefficient)| {
                            sum + low_weights[column - base] * coefficient
                        });
                    matrix_sum += row_weights[span.row] * span_sum;
                }
                chunk_sum += batch_scale * matrix_sum;
            }
            chunk_weight * chunk_sum
        })
        .reduce(|| F::zero(), |left, right| left + right);

    Ok(evaluation)
}

pub(crate) fn r1cs_num_vars<R: Send + Sync>(
    matrices: &ConstraintMatrices<R>,
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
///
/// The digest domain is intentionally field-neutral. A protocol that supports
/// more than one field must bind the field choice in its transcript session or
/// instance; canonical coefficient encodings need not identify their field.
pub(crate) fn constraint_matrix_digest<F: BitzField>(
    m: &SparseBoolMatrix,
    matrices: [&SparseMatrix<F>; 3],
) -> Result<[u8; 32], SpartanMatrixError> {
    let mut hash = Sha256::new();
    hash.update(b"bitz/spartan/constraint-matrices/v1");

    hash.update(b"M");
    hash_usize(&mut hash, m.row_count())?;
    hash_usize(&mut hash, m.column_count())?;
    for row in m.rows() {
        hash_usize(&mut hash, row.positions().len())?;
        for &column in row.positions() {
            hash_usize(&mut hash, column)?;
        }
    }

    for (label, matrix) in [b"A", b"B", b"C"].into_iter().zip(matrices) {
        hash.update(label);
        hash_sparse_matrix(&mut hash, matrix)?;
    }

    Ok(hash.finalize().into())
}

fn hash_sparse_matrix<F>(
    hash: &mut Sha256,
    matrix: &SparseMatrix<F>,
) -> Result<(), SpartanMatrixError>
where
    F: Encoding<[u8]>,
{
    hash_usize(hash, matrix.row_count())?;
    hash_usize(hash, matrix.column_count())?;
    for row in matrix.rows() {
        hash_usize(hash, row.entries().len())?;
        for (column, coefficient) in row.entries() {
            hash_usize(hash, *column)?;
            let encoding = coefficient.encode();
            let bytes = encoding.as_ref();
            hash_usize(hash, bytes.len())?;
            hash.update(bytes);
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

#[cfg(test)]
mod tests {
    use super::*;
    use circuit::constraints::{ConstraintMatrices, SparseBoolMatrix, SparseMatrix};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    type F = field::FqDefault;

    /// Three chunks of columns plus one chunk of padding, so rows straddle
    /// chunk boundaries and the last chunk holds no nonzeros.
    const COLUMNS: usize = 3 << BIND_CHUNK_COLUMN_VARS;
    const ROWS: usize = 37;
    const ENTRIES_PER_ROW: usize = 24;

    fn random_sparse_matrix(rng: &mut Pcg64) -> SparseMatrix<F> {
        let rows = (0..ROWS)
            .map(|_| {
                let mut columns: Vec<usize> = (0..ENTRIES_PER_ROW)
                    .map(|_| rng.random_range(0..COLUMNS))
                    .collect();
                columns.sort_unstable();
                columns.dedup();
                columns
                    .into_iter()
                    .map(|column| (column, F::from(u128::from(rng.random::<u64>()))))
                    .collect()
            })
            .collect();
        SparseMatrix::try_from_rows(COLUMNS, rows).unwrap()
    }

    fn random_matrices(rng: &mut Pcg64) -> ConstraintMatrices<F> {
        let m = SparseBoolMatrix::try_from_rows(1, vec![Vec::new(); COLUMNS]).unwrap();
        let a = random_sparse_matrix(rng);
        let b = random_sparse_matrix(rng);
        let c = random_sparse_matrix(rng);
        ConstraintMatrices { m, a, b, c }
    }

    fn random_prepared_matrices(rng: &mut Pcg64) -> PreparedConstraintMatrices<F> {
        PreparedConstraintMatrices::new(random_matrices(rng)).unwrap()
    }

    fn random_point(rng: &mut Pcg64, len: usize) -> Vec<F> {
        (0..len)
            .map(|_| F::from(u128::from(rng.random::<u64>())))
            .collect()
    }

    /// `D(r_y)` by the direct triple loop over every nonzero.
    fn reference_evaluation(
        matrices: [&SparseMatrix<F>; 3],
        row_point: &[F],
        rho: F,
        column_point: &[F],
    ) -> F {
        let row_weights = poly::eq_table(row_point);
        let column_weights = poly::eq_table(column_point);
        let mut evaluation = F::from(0u128);
        for (matrix, batch_scale) in matrices.into_iter().zip([F::from(1u128), rho, rho * rho]) {
            for (row, entries) in matrix.rows().iter().enumerate() {
                for &(column, coefficient) in entries.entries() {
                    evaluation +=
                        batch_scale * row_weights[row] * column_weights[column] * coefficient;
                }
            }
        }
        evaluation
    }

    #[test]
    fn column_chunks_partition_every_row() {
        let mut rng = Pcg64::seed_from_u64(7);
        let prepared = random_prepared_matrices(&mut rng);

        for (matrix, index) in prepared
            .matrices()
            .into_iter()
            .zip(prepared.column_chunks.iter())
        {
            let mut spans: Vec<_> = index
                .spans
                .iter()
                .enumerate()
                .flat_map(|(chunk, spans)| spans.iter().map(move |span| (chunk, *span)))
                .collect();
            spans.sort_by_key(|(_, span)| (span.row, span.start));

            let mut spans = spans.into_iter().peekable();
            for (row, entries) in matrix.rows().iter().enumerate() {
                let mut next_start = 0;
                while let Some((chunk, span)) = spans.next_if(|(_, span)| span.row == row) {
                    assert_eq!(span.start, next_start);
                    assert!(span.end > span.start);
                    for &(column, _) in &entries.entries()[span.start..span.end] {
                        assert_eq!(column / index.chunk_len, chunk);
                    }
                    next_start = span.end;
                }
                assert_eq!(next_start, entries.entries().len());
            }
            assert!(spans.next().is_none());
        }
    }

    #[test]
    fn evaluate_batched_matches_reference_and_bound_table() {
        let mut rng = Pcg64::seed_from_u64(11);
        let prepared = random_prepared_matrices(&mut rng);
        let row_point = random_point(&mut rng, prepared.num_row_vars());
        let column_point = random_point(&mut rng, prepared.num_column_vars());
        let rho = F::from(u128::from(rng.random::<u64>()));

        let expected = reference_evaluation(prepared.matrices(), &row_point, rho, &column_point);
        let evaluation = prepared
            .evaluate_batched(&row_point, rho, &column_point)
            .unwrap();
        assert_eq!(evaluation, expected);

        let bound = prepared.bind_and_batch(&row_point, rho).unwrap();
        assert_eq!(bound.evaluate(&column_point).unwrap(), expected);
    }

    /// Projecting integer matrices prepared once gives exactly what preparing
    /// the projected matrices gives, and the chunking is shared, not rebuilt.
    #[test]
    fn projecting_prepared_integer_matrices_matches_preparing_the_projection() {
        let mut rng = Pcg64::seed_from_u64(13);
        let mut random_integer_matrix = || {
            let rows = (0..ROWS)
                .map(|_| {
                    let mut columns: Vec<usize> = (0..ENTRIES_PER_ROW)
                        .map(|_| rng.random_range(0..COLUMNS))
                        .collect();
                    columns.sort_unstable();
                    columns.dedup();
                    columns
                        .into_iter()
                        .map(|column| (column, i128::from(rng.random::<i64>())))
                        .collect()
                })
                .collect();
            SparseMatrix::try_from_rows(COLUMNS, rows).unwrap()
        };
        let project = |coefficient: &i128| {
            let magnitude = F::from(coefficient.unsigned_abs());
            if *coefficient < 0 {
                -magnitude
            } else {
                magnitude
            }
        };

        let integer_matrices = ConstraintMatrices {
            m: SparseBoolMatrix::try_from_rows(1, vec![Vec::new(); COLUMNS]).unwrap(),
            a: random_integer_matrix(),
            b: random_integer_matrix(),
            c: random_integer_matrix(),
        };
        let prepared = PreparedIntegerMatrices::new(integer_matrices.clone()).unwrap();
        let projected = prepared.project(project).unwrap();
        let direct =
            PreparedConstraintMatrices::new(integer_matrices.map_coefficients(|c| project(&c)))
                .unwrap();
        assert_eq!(projected, direct);
        assert!(Arc::ptr_eq(
            &projected.column_chunks,
            &prepared.project(project).unwrap().column_chunks
        ));
        assert_eq!(prepared.matrices().a.row_count(), projected.row_count());
        assert_eq!(
            prepared.matrices().a.column_count(),
            projected.column_count()
        );
    }
}
