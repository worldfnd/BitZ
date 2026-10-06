//! R1CS matrix preparation and the sparse kernels used by Spartan.

use circuit::constraints::{ConstraintMatrices, SparseBoolMatrix, SparseMatrix};
use circuit::matrix_products::{IntegerProducts, ModularVector, RuntimeModulus, StoredInteger};
use circuit::witgen::PackedWitness;
use circuit::{BitWidth, IntoWords};
use common::{BitzClaimField, BitzField};
use num_traits::ToPrimitive;
use poly::DenseMultilinearExtension;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::sync::Arc;

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
/// proof: shape validated, Boolean domains sized, nonzeros laid out row by
/// row and chunked by column, statement digested. None of that depends on a
/// modulus, so [`Self::project`] shares it and only reduces the coefficients.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedIntegerMatrices<R> {
    m: SparseBoolMatrix,
    layout: Arc<Layout>,
    /// The coefficients of `A`, `B` and `C`, each in its layout's order.
    values: [Vec<R>; 3],
    /// `integer_matrix_digest`: over the integers, so it exists before a
    /// modulus does.
    digest: [u8; 32],
}

/// Immutable field-valued constraint matrices prepared for repeated Spartan
/// proofs and verification: the coefficients of `A`, `B` and `C` over the
/// field, their shared layout and the canonical statement digest.
///
/// Built by [`PreparedIntegerMatrices::project`] under a drawn modulus, or by
/// [`Self::new`] from matrices already over the field. The digest is stamped
/// by whichever built it: `project` carries the `integer_matrix_digest` its
/// source computed at setup, hashing nothing itself, while `new` hashes the
/// field coefficients it is given (`constraint_matrix_digest`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedConstraintMatrices<F> {
    layout: Arc<Layout>,
    /// The coefficients of `A`, `B` and `C`, each in its layout's order.
    values: [Vec<F>; 3],
    digest: [u8; 32],
}

/// Everything about `A`, `B` and `C` but their coefficients: positions in
/// compressed-row form, the column chunking and the Boolean domain sizes.
/// Modulus-free, so one layout serves the integer matrices and every
/// projection of them.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Layout {
    positions: [Positions; 3],
    column_chunks: [ColumnChunkIndex; 3],
    /// R1CS rows and assignment entries before padding.
    row_count: usize,
    column_count: usize,
    num_row_vars: usize,
    num_column_vars: usize,
}

/// Where one matrix's nonzeros are: row `i` owns
/// `columns[row_starts[i]..row_starts[i + 1]]` and the same range of the
/// matrix's values.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Positions {
    row_starts: Vec<usize>,
    columns: Vec<u32>,
}

impl Positions {
    /// Lays `matrix` out row by row, returning its coefficients in that order.
    fn new<C>(matrix: SparseMatrix<C>) -> Result<(Self, Vec<C>), SpartanMatrixError> {
        let rows = matrix.into_rows();
        let nonzeros = rows.iter().map(|row| row.entries().len()).sum();
        let mut row_starts = Vec::with_capacity(rows.len() + 1);
        let mut columns = Vec::with_capacity(nonzeros);
        let mut values = Vec::with_capacity(nonzeros);
        row_starts.push(0);
        for row in rows {
            for (column, coefficient) in row.into_entries() {
                let column =
                    u32::try_from(column).map_err(|_| SpartanMatrixError::DomainTooLarge)?;
                columns.push(column);
                values.push(coefficient);
            }
            row_starts.push(columns.len());
        }
        Ok((
            Self {
                row_starts,
                columns,
            },
            values,
        ))
    }

    fn row_count(&self) -> usize {
        self.row_starts.len() - 1
    }

    /// The positions of row `row`.
    fn row(&self, row: usize) -> std::ops::Range<usize> {
        self.row_starts[row]..self.row_starts[row + 1]
    }
}

/// `bind_and_batch` splits the column domain into chunks of `2^16` columns and
/// accumulates each chunk on its own thread.
/// A chunk's slice of the dense table fits in cache.
const BIND_CHUNK_COLUMN_VARS: usize = 16;

/// The nonzeros of one sparse matrix grouped by column chunk.
///
/// Chunk `c` owns columns `[c * chunk_len, (c + 1) * chunk_len)` and lists
/// the positions of every row that fall into it, in row order. The thread
/// accumulating chunk `c` reads only these positions and writes only these
/// columns, so threads never share a column.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ColumnChunkIndex {
    chunk_len: usize,
    spans: Vec<Vec<RowSpan>>,
}

/// The positions `[start, end)` of row `row`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowSpan {
    row: usize,
    start: usize,
    end: usize,
}

impl ColumnChunkIndex {
    /// Groups `positions` into `chunk_count` chunks of `chunk_len` columns.
    fn new(positions: &Positions, chunk_len: usize, chunk_count: usize) -> Self {
        debug_assert!(chunk_len.is_power_of_two());

        let mut spans = vec![Vec::new(); chunk_count];
        for row in 0..positions.row_count() {
            let range = positions.row(row);
            let mut start = range.start;
            while start < range.end {
                // Columns increase within a row, so a chunk's entries are a
                // contiguous run.
                let chunk = positions.columns[start] as usize / chunk_len;
                let end = start
                    + positions.columns[start..range.end]
                        .partition_point(|&column| column as usize / chunk_len == chunk);
                spans[chunk].push(RowSpan { row, start, end });
                start = end;
            }
        }

        Self { chunk_len, spans }
    }
}

impl Layout {
    /// Validates the shape, lays `a`, `b` and `c` out and chunks them, and
    /// returns `m` and the coefficients in layout order.
    #[allow(clippy::type_complexity)]
    fn new<C: Send + Sync>(
        matrices: ConstraintMatrices<C>,
    ) -> Result<(SparseBoolMatrix, Arc<Self>, [Vec<C>; 3]), SpartanMatrixError> {
        let (num_row_vars, num_column_vars) = r1cs_num_vars(&matrices)?;
        let row_count = matrices.a.row_count();
        let column_count = matrices.a.column_count();
        let ConstraintMatrices { m, a, b, c } = matrices;
        let (a, a_values) = Positions::new(a)?;
        let (b, b_values) = Positions::new(b)?;
        let (c, c_values) = Positions::new(c)?;
        let positions = [a, b, c];

        let num_columns = 1_usize << num_column_vars;
        let chunk_len = num_columns.min(1_usize << BIND_CHUNK_COLUMN_VARS);
        let chunk_count = num_columns / chunk_len;
        let column_chunks = [&positions[0], &positions[1], &positions[2]]
            .map(|positions| ColumnChunkIndex::new(positions, chunk_len, chunk_count));

        let layout = Self {
            positions,
            column_chunks,
            row_count,
            column_count,
            num_row_vars,
            num_column_vars,
        };
        Ok((m, Arc::new(layout), [a_values, b_values, c_values]))
    }
}

impl<R> PreparedIntegerMatrices<R>
where
    R: Send + Sync + ToPrimitive,
    for<'a> StoredInteger: From<&'a R>,
{
    pub fn new(matrices: ConstraintMatrices<R>) -> Result<Self, SpartanMatrixError> {
        let (m, layout, values) = Layout::new(matrices)?;
        let digest = integer_matrix_digest(&m, &layout, [&values[0], &values[1], &values[2]])?;
        Ok(Self {
            m,
            layout,
            values,
            digest,
        })
    }

    /// The Boolean witness matrix.
    pub fn m(&self) -> &SparseBoolMatrix {
        &self.m
    }

    /// Number of R1CS rows before padding.
    pub fn row_count(&self) -> usize {
        self.layout.row_count
    }

    /// Number of assignment entries before padding, the constant included.
    pub fn column_count(&self) -> usize {
        self.layout.column_count
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The matrices with `map` applied to every coefficient of `A`, `B` and
    /// `C`, each into one contiguous table: the per-proof work. The layout
    /// and the digest are shared.
    pub fn project<F, M>(&self, map: M) -> PreparedConstraintMatrices<F>
    where
        F: BitzField,
        M: Fn(&R) -> F + Sync,
    {
        let values = [&self.values[0], &self.values[1], &self.values[2]]
            .map(|values| values.par_iter().map(&map).collect());
        PreparedConstraintMatrices {
            layout: Arc::clone(&self.layout),
            values,
            digest: self.digest,
        }
    }
}

impl<F: BitzField> PreparedConstraintMatrices<F> {
    /// Prepares matrices already over the field; the digest is
    /// `constraint_matrix_digest`.
    pub fn new(matrices: ConstraintMatrices<F>) -> Result<Self, SpartanMatrixError> {
        let (m, layout, values) = Layout::new(matrices)?;
        let digest = constraint_matrix_digest(&m, &layout, [&values[0], &values[1], &values[2]])?;
        Ok(Self {
            layout,
            values,
            digest,
        })
    }

    /// Returns human-readable info about R1CS matrices and their nonzero entries.
    pub fn short_debug_info(&self) -> String {
        let nonzeros: usize = self.values.iter().map(Vec::len).sum();
        format!(
            "{} r1cs rows -> 2^{}, {} h entries -> 2^{}, {nonzeros} nonzeros",
            self.row_count(),
            self.num_row_vars(),
            self.column_count(),
            self.num_column_vars(),
        )
    }

    /// Number of R1CS rows before padding.
    pub fn row_count(&self) -> usize {
        self.layout.row_count
    }

    /// Number of assignment entries before padding, the constant included.
    pub fn column_count(&self) -> usize {
        self.layout.column_count
    }

    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn num_row_vars(&self) -> usize {
        self.layout.num_row_vars
    }

    pub fn num_column_vars(&self) -> usize {
        self.layout.num_column_vars
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
        let layout = &*self.layout;
        if row_point.len() != layout.num_row_vars {
            return Err(SpartanMatrixError::InvalidRowPointLength {
                expected: layout.num_row_vars,
                actual: row_point.len(),
            });
        }

        let row_weights = poly::eq_table(row_point);
        let batched = self.batched(rho);
        let chunk_len = layout.column_chunks[0].chunk_len;

        // Every task owns one column chunk of the table and touches only the
        // nonzeros landing in it: no two tasks write the same column, and each
        // task's writes stay within a cache-sized slice.
        let mut evaluations: Vec<F> =
            rayon::iter::repeat_n(F::zero(), 1usize << layout.num_column_vars).collect();
        evaluations
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk, output)| {
                let base = chunk * chunk_len;
                for (positions, index, values, batch_scale) in batched {
                    for span in &index.spans[chunk] {
                        let row_scale = row_weights[span.row] * batch_scale;
                        for k in span.start..span.end {
                            output[positions.columns[k] as usize - base] += row_scale * values[k];
                        }
                    }
                }
            });

        DenseMultilinearExtension::from_evaluations(layout.num_column_vars, evaluations)
            .map_err(|_| SpartanMatrixError::InvalidMleOperation)
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
        let layout = &*self.layout;
        if row_point.len() != layout.num_row_vars {
            return Err(SpartanMatrixError::InvalidRowPointLength {
                expected: layout.num_row_vars,
                actual: row_point.len(),
            });
        }
        if column_point.len() != layout.num_column_vars {
            return Err(SpartanMatrixError::InvalidColumnPointLength {
                expected: layout.num_column_vars,
                actual: column_point.len(),
            });
        }

        let row_weights = poly::eq_table(row_point);
        let batched = self.batched(rho);

        // All tasks share one cache-sized `low` table and a per-chunk factor.
        // Each task reduces one column chunk of nonzeros to a single field element.
        let chunk_len = layout.column_chunks[0].chunk_len;
        let (low_point, high_point) = column_point.split_at(chunk_len.ilog2() as usize);
        let low_weights = poly::eq_table(low_point);
        let high_weights = poly::eq_table(high_point);
        debug_assert_eq!(high_weights.len(), layout.column_chunks[0].spans.len());

        let evaluation = high_weights
            .par_iter()
            .enumerate()
            .map(|(chunk, &chunk_weight)| {
                let base = chunk * chunk_len;
                let mut chunk_sum = F::zero();
                for (positions, index, values, batch_scale) in batched {
                    let mut matrix_sum = F::zero();
                    for span in &index.spans[chunk] {
                        let span_sum = (span.start..span.end).fold(F::zero(), |sum, k| {
                            sum + low_weights[positions.columns[k] as usize - base] * values[k]
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

    /// `A`, `B` and `C`, each with its positions, chunking and batching
    /// scale: `1`, `rho` and `rho^2`.
    fn batched(&self, rho: F) -> [(&Positions, &ColumnChunkIndex, &[F], F); 3] {
        let layout = &*self.layout;
        let scales = [F::one(), rho, rho * rho];
        std::array::from_fn(|i| {
            (
                &layout.positions[i],
                &layout.column_chunks[i],
                self.values[i].as_slice(),
                scales[i],
            )
        })
    }
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
/// Fiat--Shamir challenge is sampled: `domain`, the positions of `M`, then
/// `A`, `B` and `C` row by row with every coefficient written by `encode`.
fn matrix_digest<C>(
    domain: &[u8],
    m: &SparseBoolMatrix,
    layout: &Layout,
    values: [&[C]; 3],
    encode: impl Fn(&mut Sha256, &C) -> Result<(), SpartanMatrixError>,
) -> Result<[u8; 32], SpartanMatrixError> {
    let mut hash = Sha256::new();
    hash.update(domain);

    hash.update(b"M");
    hash_usize(&mut hash, m.row_count())?;
    hash_usize(&mut hash, m.column_count())?;
    for row in m.rows() {
        hash_usize(&mut hash, row.positions().len())?;
        for &column in row.positions() {
            hash_usize(&mut hash, column)?;
        }
    }

    for ((label, positions), values) in [b"A", b"B", b"C"]
        .into_iter()
        .zip(&layout.positions)
        .zip(values)
    {
        hash.update(label);
        hash_usize(&mut hash, layout.row_count)?;
        hash_usize(&mut hash, layout.column_count)?;
        for row in 0..layout.row_count {
            let range = positions.row(row);
            hash_usize(&mut hash, range.len())?;
            for k in range {
                hash_usize(&mut hash, positions.columns[k] as usize)?;
                encode(&mut hash, &values[k])?;
            }
        }
    }

    Ok(hash.finalize().into())
}

/// The statement digest over coefficients already in the field, encoded as
/// the transcript encodes them.
///
/// The digest domain is intentionally field-neutral. A protocol that supports
/// more than one field must bind the field choice in its transcript session or
/// instance; canonical coefficient encodings need not identify their field.
fn constraint_matrix_digest<F: BitzField>(
    m: &SparseBoolMatrix,
    layout: &Layout,
    values: [&[F]; 3],
) -> Result<[u8; 32], SpartanMatrixError> {
    matrix_digest(
        b"bitz/spartan/constraint-matrices/v1",
        m,
        layout,
        values,
        |hash, coefficient| {
            let encoding = transcript::Encoding::encode(coefficient);
            let bytes = encoding.as_ref();
            hash_usize(hash, bytes.len())?;
            hash.update(bytes);
            Ok(())
        },
    )
}

/// The statement digest over integer coefficients, each in its normalised
/// two's-complement words, so it is fixed before any modulus is drawn and
/// shared by every projection. Its own domain: it never equals the digest of
/// the projected matrices. A protocol that projects must bind the modulus in
/// its transcript before Spartan reads this digest, as the e2e does by
/// absorbing its parameters right after the draw.
fn integer_matrix_digest<R>(
    m: &SparseBoolMatrix,
    layout: &Layout,
    values: [&[R]; 3],
) -> Result<[u8; 32], SpartanMatrixError>
where
    R: ToPrimitive,
    for<'a> StoredInteger: From<&'a R>,
{
    matrix_digest(
        b"bitz/spartan/integer-constraint-matrices/v1",
        m,
        layout,
        values,
        |hash, coefficient| {
            // Coefficients are small in practice; spare them the heap.
            if let Some(value) = coefficient.to_i128() {
                let (words, len) = canonical_words(value);
                hash_words(hash, &words[..len])
            } else {
                hash_words(hash, StoredInteger::from(coefficient).words())
            }
        },
    )
}

/// The words [`StoredInteger`] holds for `value`, and how many: little-endian
/// two's complement with a redundant sign word dropped, none for zero.
fn canonical_words(value: i128) -> ([u64; 2], usize) {
    let (low, high) = (value as u64, (value >> 64) as u64);
    let sign_extended = (high == 0 && low >> 63 == 0) || (high == u64::MAX && low >> 63 == 1);
    let len = match (value == 0, sign_extended) {
        (true, _) => 0,
        (false, true) => 1,
        (false, false) => 2,
    };
    ([low, high], len)
}

fn hash_words(hash: &mut Sha256, words: &[u64]) -> Result<(), SpartanMatrixError> {
    hash_usize(hash, words.len())?;
    for word in words {
        hash.update(word.to_le_bytes());
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
    use circuit::constraints::{ConstraintMatrices, SparseBoolMatrix, SparseMatrix};
    use circuit::matrix_products::StoredInteger;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use std::sync::Arc;

    type F = field::FqDefault;

    use super::{BIND_CHUNK_COLUMN_VARS, PreparedConstraintMatrices, PreparedIntegerMatrices};

    /// Three chunks of columns plus one chunk of padding, so rows straddle
    /// chunk boundaries and the last chunk holds no nonzeros.
    const COLUMNS: usize = 3 << BIND_CHUNK_COLUMN_VARS;
    const ROWS: usize = 37;
    const ENTRIES_PER_ROW: usize = 24;

    fn random_sparse_matrix<C>(
        rng: &mut Pcg64,
        mut coefficient: impl FnMut(&mut Pcg64) -> C,
    ) -> SparseMatrix<C> {
        let rows = (0..ROWS)
            .map(|_| {
                let mut columns: Vec<usize> = (0..ENTRIES_PER_ROW)
                    .map(|_| rng.random_range(0..COLUMNS))
                    .collect();
                columns.sort_unstable();
                columns.dedup();
                columns
                    .into_iter()
                    .map(|column| (column, coefficient(rng)))
                    .collect()
            })
            .collect();
        SparseMatrix::try_from_rows(COLUMNS, rows).unwrap()
    }

    fn empty_m() -> SparseBoolMatrix {
        SparseBoolMatrix::try_from_rows(1, vec![Vec::new(); COLUMNS]).unwrap()
    }

    fn random_matrices(rng: &mut Pcg64) -> ConstraintMatrices<F> {
        let mut field = |rng: &mut Pcg64| F::from(u128::from(rng.random::<u64>()));
        ConstraintMatrices {
            m: empty_m(),
            a: random_sparse_matrix(rng, &mut field),
            b: random_sparse_matrix(rng, &mut field),
            c: random_sparse_matrix(rng, &mut field),
        }
    }

    fn random_integer_matrices(rng: &mut Pcg64) -> ConstraintMatrices<i128> {
        let mut integer = |rng: &mut Pcg64| i128::from(rng.random::<i64>());
        ConstraintMatrices {
            m: empty_m(),
            a: random_sparse_matrix(rng, &mut integer),
            b: random_sparse_matrix(rng, &mut integer),
            c: random_sparse_matrix(rng, &mut integer),
        }
    }

    fn random_point(rng: &mut Pcg64, len: usize) -> Vec<F> {
        (0..len)
            .map(|_| F::from(u128::from(rng.random::<u64>())))
            .collect()
    }

    /// `D(r_y)` by the direct triple loop over every nonzero.
    fn reference_evaluation(
        matrices: &ConstraintMatrices<F>,
        row_point: &[F],
        rho: F,
        column_point: &[F],
    ) -> F {
        let row_weights = poly::eq_table(row_point);
        let column_weights = poly::eq_table(column_point);
        let mut evaluation = F::from(0u128);
        for (matrix, batch_scale) in [
            (&matrices.a, F::from(1u128)),
            (&matrices.b, rho),
            (&matrices.c, rho * rho),
        ] {
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
    fn the_layout_keeps_every_nonzero_in_row_order() {
        let mut rng = Pcg64::seed_from_u64(5);
        let matrices = random_matrices(&mut rng);
        let prepared = PreparedConstraintMatrices::new(matrices.clone()).unwrap();
        assert_eq!(prepared.row_count(), ROWS);
        assert_eq!(prepared.column_count(), COLUMNS);

        for ((matrix, positions), values) in [&matrices.a, &matrices.b, &matrices.c]
            .into_iter()
            .zip(&prepared.layout.positions)
            .zip(&prepared.values)
        {
            assert_eq!(positions.row_count(), matrix.row_count());
            let mut k = 0;
            for (row, entries) in matrix.rows().iter().enumerate() {
                assert_eq!(positions.row(row), k..k + entries.entries().len());
                for &(column, coefficient) in entries.entries() {
                    assert_eq!(positions.columns[k] as usize, column);
                    assert_eq!(values[k], coefficient);
                    k += 1;
                }
            }
            assert_eq!(k, values.len());
        }
    }

    #[test]
    fn column_chunks_partition_every_row() {
        let mut rng = Pcg64::seed_from_u64(7);
        let prepared = PreparedConstraintMatrices::new(random_matrices(&mut rng)).unwrap();

        for (positions, index) in prepared
            .layout
            .positions
            .iter()
            .zip(&prepared.layout.column_chunks)
        {
            let mut spans: Vec<_> = index
                .spans
                .iter()
                .enumerate()
                .flat_map(|(chunk, spans)| spans.iter().map(move |span| (chunk, *span)))
                .collect();
            spans.sort_by_key(|(_, span)| (span.row, span.start));

            let mut spans = spans.into_iter().peekable();
            for row in 0..positions.row_count() {
                let range = positions.row(row);
                let mut next_start = range.start;
                while let Some((chunk, span)) = spans.next_if(|(_, span)| span.row == row) {
                    assert_eq!(span.start, next_start);
                    assert!(span.end > span.start);
                    for &column in &positions.columns[span.start..span.end] {
                        assert_eq!(column as usize / index.chunk_len, chunk);
                    }
                    next_start = span.end;
                }
                assert_eq!(next_start, range.end);
            }
            assert!(spans.next().is_none());
        }
    }

    #[test]
    fn evaluate_batched_matches_reference_and_bound_table() {
        let mut rng = Pcg64::seed_from_u64(11);
        let matrices = random_matrices(&mut rng);
        let prepared = PreparedConstraintMatrices::new(matrices.clone()).unwrap();
        let row_point = random_point(&mut rng, prepared.num_row_vars());
        let column_point = random_point(&mut rng, prepared.num_column_vars());
        let rho = F::from(u128::from(rng.random::<u64>()));

        let expected = reference_evaluation(&matrices, &row_point, rho, &column_point);
        let evaluation = prepared
            .evaluate_batched(&row_point, rho, &column_point)
            .unwrap();
        assert_eq!(evaluation, expected);

        let bound = prepared.bind_and_batch(&row_point, rho).unwrap();
        assert_eq!(bound.evaluate(&column_point).unwrap(), expected);
    }

    #[test]
    fn canonical_words_are_the_stored_integers() {
        for value in [
            0_i128,
            1,
            -1,
            i128::from(i64::MAX),
            i128::from(i64::MIN),
            i128::from(i64::MAX) + 1,
            i128::from(i64::MIN) - 1,
            i128::from(u64::MAX),
            -i128::from(u64::MAX),
            i128::MAX,
            i128::MIN,
        ] {
            let (words, len) = super::canonical_words(value);
            assert_eq!(
                &words[..len],
                StoredInteger::from(&value).words(),
                "{value}"
            );
        }
    }

    /// Projecting integer matrices prepared once gives what preparing the
    /// projected matrices gives, the layout shared rather than rebuilt; only
    /// the digest differs, each over its own coefficients and domain.
    #[test]
    fn projection_matches_direct_preparation_up_to_the_digest_domain() {
        let mut rng = Pcg64::seed_from_u64(13);
        let project = |coefficient: &i128| {
            let magnitude = F::from(coefficient.unsigned_abs());
            if *coefficient < 0 {
                -magnitude
            } else {
                magnitude
            }
        };

        let integer_matrices = random_integer_matrices(&mut rng);
        let prepared = PreparedIntegerMatrices::new(integer_matrices.clone()).unwrap();
        let projected = prepared.project(project);
        let direct = PreparedConstraintMatrices::new(
            integer_matrices.clone().map_coefficients(|c| project(&c)),
        )
        .unwrap();
        assert_eq!(projected.values, direct.values);
        assert_eq!(projected.layout, direct.layout);
        assert_ne!(
            projected.digest(),
            direct.digest(),
            "the integer digest is its own domain"
        );
        assert_eq!(projected.digest(), prepared.digest());
        assert!(Arc::ptr_eq(
            &projected.layout,
            &prepared.project(project).layout
        ));
        assert_eq!(prepared.row_count(), projected.row_count());
        assert_eq!(prepared.column_count(), projected.column_count());

        // The integer digest is the integer matrices': the same under every
        // projection, different as soon as a coefficient or a position is.
        type Wide = field::Fq<{ (1 << 114) - 11 }>;
        let wide = |coefficient: &i128| Wide::from(coefficient.unsigned_abs());
        assert_eq!(prepared.project(wide).digest(), prepared.digest());
        let mut negated = integer_matrices.clone();
        negated.a = negated.a.map_values_ref(|c| -c);
        assert_ne!(
            PreparedIntegerMatrices::new(negated).unwrap().digest(),
            prepared.digest()
        );
        let mut moved = integer_matrices;
        moved.m = SparseBoolMatrix::try_from_rows(2, vec![vec![1]; COLUMNS]).unwrap();
        assert_ne!(
            PreparedIntegerMatrices::new(moved).unwrap().digest(),
            prepared.digest()
        );
    }
}
