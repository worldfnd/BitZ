//! Materialization and multiplication of the transposed Boolean matrix.
//!
//! [`MTransposeGenerator`] builds a compact row-major representation of `M^T`
//! from circuit structure alone. [`MaterializedMTranspose`] computes `r * M`
//! as parallel, disjoint column gathers over the GHASH field.

use std::error::Error;
use std::fmt::{self, Display};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use common::{TransposedWeights, VirtualMap, VirtualMapError};
use field::F128;
use poly::eq_table;
use rayon::prelude::*;

use crate::witgen::Z;
use crate::{BoolWitness, Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

const PARALLEL_MATRIX_NNZ_THRESHOLD: usize = 1 << 15;

const MATRIX_INLINE_SUPPORT: usize = 4;
const MATRIX_ARENA_SUPPORT: u8 = u8::MAX;
static NEXT_MATRIX_ID: AtomicU64 = AtomicU64::new(1);

/// A canonical sparse F2 expression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixBit {
    constant_term: bool,
    support_len: u8,
    support: [u32; MATRIX_INLINE_SUPPORT],
    matrix_id: u32,
}

impl MatrixBit {
    fn constant(value: bool) -> Self {
        let mut support = [0; MATRIX_INLINE_SUPPORT];
        support[0] = 0;
        Self {
            constant_term: value,
            support_len: u8::from(value),
            support,
            matrix_id: 0,
        }
    }

    fn inline(support: &[u32], matrix_id: u32) -> Self {
        debug_assert!(support.len() <= MATRIX_INLINE_SUPPORT);
        let mut inline = [0; MATRIX_INLINE_SUPPORT];
        inline[..support.len()].copy_from_slice(support);
        Self {
            constant_term: support.first() == Some(&0),
            support_len: support.len() as u8,
            support: inline,
            matrix_id,
        }
    }

    const fn constant_term(&self) -> bool {
        self.constant_term
    }
}

impl From<bool> for MatrixBit {
    fn from(value: bool) -> Self {
        Self::constant(value)
    }
}

impl BoolWitness for MatrixBit {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
}

fn symmetric_difference(left: &[u32], right: &[u32], output: &mut [u32]) -> usize {
    let (mut left_index, mut right_index, mut output_index) = (0, 0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => {
                output[output_index] = left[left_index];
                left_index += 1;
                output_index += 1;
            }
            std::cmp::Ordering::Greater => {
                output[output_index] = right[right_index];
                right_index += 1;
                output_index += 1;
            }
            std::cmp::Ordering::Equal => {
                left_index += 1;
                right_index += 1;
            }
        }
    }
    for &entry in &left[left_index..] {
        output[output_index] = entry;
        output_index += 1;
    }
    for &entry in &right[right_index..] {
        output[output_index] = entry;
        output_index += 1;
    }
    output_index
}

/// Row recorder finalized into CSC for `M` (row-major `M^T`).
#[derive(Debug)]
pub(crate) struct MTransposeRecorder {
    id: u32,
    witness_count: usize,
    expression_offsets: Vec<u32>,
    expression_entries: Vec<u32>,
    row_offsets: Vec<u32>,
    column_indices: Vec<u32>,
}

impl MTransposeRecorder {
    pub(crate) fn with_witnesses(witness_count: usize) -> Self {
        assert!(
            witness_count < u32::MAX as usize,
            "too many Boolean witnesses for M"
        );
        let id = u32::try_from(NEXT_MATRIX_ID.fetch_add(1, Ordering::Relaxed))
            .expect("matrix identifier overflow");
        assert_ne!(id, 0, "matrix identifier overflow");
        Self {
            id,
            witness_count,
            expression_offsets: vec![0],
            expression_entries: Vec::new(),
            row_offsets: vec![0, 1],
            column_indices: vec![0],
        }
    }

    pub(crate) fn witness(&self, index: usize) -> MatrixBit {
        assert!(
            index < self.witness_count,
            "Boolean witness is not allocated"
        );
        MatrixBit {
            constant_term: false,
            support_len: 1,
            support: [
                u32::try_from(index + 1).expect("too many Boolean witnesses for M"),
                0,
                0,
                0,
            ],
            matrix_id: self.id,
        }
    }

    pub(crate) fn allocate_witness(&mut self) -> MatrixBit {
        let index = self.witness_count;
        self.witness_count = self
            .witness_count
            .checked_add(1)
            .filter(|count| *count < u32::MAX as usize)
            .expect("too many Boolean witnesses for M");
        self.witness(index)
    }

    fn check_owner(&self, value: MatrixBit) {
        assert!(
            value.matrix_id == 0 || value.matrix_id == self.id,
            "Boolean expression belongs to another matrix recorder"
        );
    }

    fn support<'a>(&'a self, value: &'a MatrixBit) -> &'a [u32] {
        if value.support_len == MATRIX_ARENA_SUPPORT {
            let index = value.support[0] as usize;
            let start = self.expression_offsets[index] as usize;
            let end = self.expression_offsets[index + 1] as usize;
            &self.expression_entries[start..end]
        } else {
            &value.support[..usize::from(value.support_len)]
        }
    }

    fn bit_from_support(&mut self, support: &[u32]) -> MatrixBit {
        if support.len() <= MATRIX_INLINE_SUPPORT {
            return MatrixBit::inline(support, self.id);
        }
        let expression = self.expression_offsets.len() - 1;
        self.expression_entries.extend_from_slice(support);
        self.expression_offsets.push(
            u32::try_from(self.expression_entries.len())
                .expect("too many Boolean-expression entries for M"),
        );
        MatrixBit {
            constant_term: support.first() == Some(&0),
            support_len: MATRIX_ARENA_SUPPORT,
            support: [
                u32::try_from(expression).expect("too many Boolean expressions for M"),
                0,
                0,
                0,
            ],
            matrix_id: self.id,
        }
    }

    pub(crate) fn xor(&mut self, lhs: MatrixBit, rhs: MatrixBit) -> MatrixBit {
        self.check_owner(lhs);
        self.check_owner(rhs);
        if lhs.support_len == 0 {
            return rhs;
        }
        if rhs.support_len == 0 {
            return lhs;
        }

        let left = self.support(&lhs);
        let right = self.support(&rhs);
        if left.len() + right.len() <= 2 * MATRIX_INLINE_SUPPORT {
            let mut merged = [0; 2 * MATRIX_INLINE_SUPPORT];
            let len = symmetric_difference(left, right, &mut merged);
            return self.bit_from_support(&merged[..len]);
        }

        let mut merged = vec![0; left.len() + right.len()];
        let len = symmetric_difference(left, right, &mut merged);
        merged.truncate(len);
        self.bit_from_support(&merged)
    }

    pub(crate) fn push_row(&mut self, value: &MatrixBit) {
        self.check_owner(*value);
        if value.support_len == MATRIX_ARENA_SUPPORT {
            let index = value.support[0] as usize;
            let start = self.expression_offsets[index] as usize;
            let end = self.expression_offsets[index + 1] as usize;
            self.column_indices
                .extend_from_slice(&self.expression_entries[start..end]);
        } else {
            self.column_indices
                .extend_from_slice(&value.support[..usize::from(value.support_len)]);
        }
        self.row_offsets
            .push(u32::try_from(self.column_indices.len()).expect("too many nonzeros in M"));
    }

    pub(crate) fn finish(self) -> MaterializedMTranspose {
        let column_count = self.witness_count + 1;
        let row_count = self.row_offsets.len() - 1;
        let mut column_offsets = vec![0_u32; column_count + 1];
        for &column in &self.column_indices {
            column_offsets[column as usize + 1] = column_offsets[column as usize + 1]
                .checked_add(1)
                .expect("too many nonzeros in one M column");
        }
        for column in 0..column_count {
            column_offsets[column + 1] = column_offsets[column + 1]
                .checked_add(column_offsets[column])
                .expect("too many nonzeros in M");
        }

        let mut cursors = column_offsets[..column_count].to_vec();
        let mut row_indices = vec![0_u32; self.column_indices.len()];
        for row in 0..row_count {
            let start = self.row_offsets[row] as usize;
            let end = self.row_offsets[row + 1] as usize;
            for &column in &self.column_indices[start..end] {
                let cursor = &mut cursors[column as usize];
                row_indices[*cursor as usize] = u32::try_from(row).expect("too many rows in M");
                *cursor += 1;
            }
        }

        MaterializedMTranspose {
            row_count,
            column_offsets: column_offsets.into_boxed_slice(),
            row_indices: row_indices.into_boxed_slice(),
        }
    }
}

/// A compact row-major representation of `M^T` over F2 (equivalently, CSC for `M`).
#[derive(Debug)]
pub struct MaterializedMTranspose {
    row_count: usize,
    column_offsets: Box<[u32]>,
    row_indices: Box<[u32]>,
}

impl MaterializedMTranspose {
    /// Number of rows in `M`, including its implicit constant row.
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of columns in `M`, including its implicit constant column.
    pub const fn column_count(&self) -> usize {
        self.column_offsets.len() - 1
    }

    /// Number of nonzero entries in `M`.
    pub const fn nonzero_count(&self) -> usize {
        self.row_indices.len()
    }

    /// Bytes occupied by the fixed-width CSC payload.
    pub const fn payload_bytes(&self) -> usize {
        self.column_offsets.len() * size_of::<u32>() + self.row_indices.len() * size_of::<u32>()
    }

    /// Computes `r * M` and allocates the result vector.
    pub fn apply(&self, challenges: &[F128]) -> Result<Vec<F128>, MatrixApplyError> {
        let mut output = Vec::new();
        self.apply_into(challenges, &mut output)?;
        Ok(output)
    }

    /// Computes `r * M` into a reusable result allocation.
    pub fn apply_into(
        &self,
        challenges: &[F128],
        output: &mut Vec<F128>,
    ) -> Result<(), MatrixApplyError> {
        self.apply_inner(challenges, output, None)
    }

    fn apply_inner(
        &self,
        challenges: &[F128],
        output: &mut Vec<F128>,
        force_parallel: Option<bool>,
    ) -> Result<(), MatrixApplyError> {
        if challenges.len() != self.row_count {
            return Err(MatrixApplyError {
                expected: self.row_count,
                actual: challenges.len(),
            });
        }
        output.resize(self.column_count(), F128::new(0, 0));

        let evaluate = |column: usize| {
            let start = self.column_offsets[column] as usize;
            let end = self.column_offsets[column + 1] as usize;
            let mut lo = 0_u64;
            let mut hi = 0_u64;
            for &row in &self.row_indices[start..end] {
                let challenge = challenges[row as usize];
                lo ^= challenge.lo;
                hi ^= challenge.hi;
            }
            F128::new(lo, hi)
        };
        let parallel = force_parallel.unwrap_or_else(|| {
            rayon::current_num_threads() > 1
                && self.row_indices.len() >= PARALLEL_MATRIX_NNZ_THRESHOLD
        });
        if parallel {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(column, value)| *value = evaluate(column));
        } else {
            output
                .iter_mut()
                .enumerate()
                .for_each(|(column, value)| *value = evaluate(column));
        }
        Ok(())
    }
}

/// Domain separation for the digest.
const DIGEST_DOMAIN: &[u8] = b"BitZ/virtual-map/csc/v1";

const DIGEST_CHUNK: usize = 1 << 12;

impl VirtualMap for MaterializedMTranspose {
    /// `h` is zero-padded to a power of two, so truncating the equality table
    /// to `row_count` drops only weights that multiply a zero.
    fn transpose_eq(&self, point: &[F128]) -> Result<TransposedWeights, VirtualMapError> {
        if point.len() < self.row_count.next_power_of_two().trailing_zeros() as usize {
            return Err(VirtualMapError::PointLengthMismatch);
        }

        let mut weights = eq_table(point);
        weights.truncate(self.row_count);
        let mut transposed = self
            .apply(&weights)
            .map_err(|_| VirtualMapError::PointLengthMismatch)?;

        // Column zero is `M`'s constant column.
        let on_bits = transposed.split_off(1);
        Ok(TransposedWeights::new(on_bits, transposed[0]))
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(DIGEST_DOMAIN);
        for count in [self.row_count, self.column_count(), self.nonzero_count()] {
            hasher.update(&(count as u64).to_le_bytes());
        }
        // Hash the offsets too: two matrices can share every row index and
        // still split the columns differently.
        for indices in [&self.column_offsets, &self.row_indices] {
            let mut bytes = Vec::with_capacity(DIGEST_CHUNK * size_of::<u32>());
            for chunk in indices.chunks(DIGEST_CHUNK) {
                bytes.clear();
                bytes.extend(chunk.iter().flat_map(|index| index.to_le_bytes()));
                hasher.update(&bytes);
            }
        }
        *hasher.finalize().as_bytes()
    }

    fn h_len(&self) -> usize {
        self.row_count
    }

    fn f_len(&self) -> usize {
        self.column_count()
    }
}

/// A challenge-vector dimension mismatch while applying materialized `M^T`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixApplyError {
    pub expected: usize,
    pub actual: usize,
}

impl Display for MatrixApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "challenge vector has length {}, expected {}",
            self.actual, self.expected
        )
    }
}

impl Error for MatrixApplyError {}

/// Materializes compact `M^T` from a circuit's static Boolean structure.
#[derive(Debug)]
pub struct MTransposeGenerator {
    recorder: MTransposeRecorder,
    inputs: Box<[MatrixBit]>,
}

impl MTransposeGenerator {
    /// Creates a matrix generator with preallocated Boolean input columns.
    pub fn new(input_count: usize) -> Self {
        let recorder = MTransposeRecorder::with_witnesses(input_count);
        let inputs = (0..input_count)
            .map(|index| recorder.witness(index))
            .collect();
        Self { recorder, inputs }
    }

    /// Moves every input into a fixed-size boxed array without cloning.
    pub fn take_boxed_inputs<const N: usize>(&mut self) -> Box<[MatrixBit; N]> {
        assert_eq!(N, self.inputs.len(), "input witness count mismatch");
        std::mem::take(&mut self.inputs)
            .try_into()
            .unwrap_or_else(|_| unreachable!("input length was checked"))
    }

    /// Moves dynamically sized input handles out without cloning.
    pub fn take_inputs(&mut self) -> Box<[MatrixBit]> {
        std::mem::take(&mut self.inputs)
    }

    /// Finishes the compact transposed Boolean matrix.
    pub fn finish(self) -> MaterializedMTranspose {
        self.recorder.finish()
    }
}

impl Circuit for MTransposeGenerator {
    type Bool = MatrixBit;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = Z<LIMBS>;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Z<LIMBS> {
        Z::from_le_words(words)
    }

    fn xor(&mut self, lhs: MatrixBit, rhs: MatrixBit) -> MatrixBit {
        self.recorder.xor(lhs, rhs)
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<MatrixBit, N>
    where
        H: Fn(&dyn WitnessContext<Z<LIMBS>, MatrixBit, Z<LIMBS>>) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        ScalarBits(std::array::from_fn(|_| self.recorder.allocate_witness()))
    }

    fn bitz<const LIMBS: usize>(&mut self, value: MatrixBit) -> Z<LIMBS> {
        self.recorder.push_row(&value);
        Z::from(u64::from(value.constant_term()))
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        bits_le: &<MatrixBit as BoolWitness>::Repr<N, M>,
    ) -> (Z<LIMBS>, Z<LIMBS>) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        let values: [bool; N] = std::array::from_fn(|index| {
            let bit = &bits_le.0[index];
            self.recorder.push_row(bit);
            bit.constant_term()
        });
        (Z::from_le_bits(&values), Z::from_le_bits(&values[..LOW]))
    }

    fn assert_r1c<const LIMBS: usize>(&mut self, _: Z<LIMBS>, _: Z<LIMBS>, _: Z<LIMBS>) {}

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: Z<FROM_LIMBS>,
    ) -> Z<TO_LIMBS> {
        value.sign_extend()
    }
}

#[cfg(test)]
mod tests {
    use crate::constraints::ConstraintGenerator;
    use crate::sha256::{COMPRESSION_HINT_BITS, COMPRESSION_INPUT_BITS, compression_circuit};
    use crate::witgen::Witgen;
    use crate::{BoolRepresentation, BoolWitness, Circuit};

    use super::*;

    fn example_circuit<CS: Circuit>(circuit: &mut CS, inputs: &[CS::Bool; 3]) {
        let xy = circuit.xor(inputs[0].clone(), inputs[1].clone());
        let not_xy = circuit.xor(xy.clone(), CS::Bool::from(true));
        let _ = circuit.bitz::<1>(xy);
        let _ = circuit.bitz::<1>(not_xy);

        let captured = inputs[2].clone();
        let hinted = circuit.hint::<1, 2, 1, _>(move |context| {
            let value = context.eval_bool(&captured);
            Ok(crate::PackedBits::from_array([value, !value]))
        });
        let hinted_zero =
            <<<CS as Circuit>::Bool as BoolWitness>::Repr<2, 1> as BoolRepresentation<
                CS::Bool,
                2,
                1,
            >>::bit(&hinted, 0);
        let mixed = circuit.xor(hinted_zero, inputs[0].clone());
        let _ = circuit.bitz::<1>(mixed);
        let _: (CS::Z<1>, CS::Z<1>) = circuit.bitz_unsigned::<1, 2, 1, 1>(&hinted);
    }

    fn challenges(count: usize) -> Vec<F128> {
        (0..count)
            .map(|index| {
                F128::new(
                    (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03) ^ 0xa5a5,
                )
            })
            .collect()
    }

    /// A point over the padded `h`.
    fn point(rows: usize) -> Vec<F128> {
        challenges(rows.next_power_of_two().trailing_zeros() as usize)
    }

    /// `<M^T eq, (1 || f)>` must equal `<eq, h>`.
    #[test]
    fn the_transposed_claim_matches_the_claim_on_the_integer_witness() {
        let inputs = [true, false, true];

        let mut materializer = MTransposeGenerator::new(inputs.len());
        let matrix_inputs = materializer.take_boxed_inputs();
        example_circuit(&mut materializer, &matrix_inputs);
        let map = materializer.finish();

        let mut witgen = Witgen::with_inputs(&inputs);
        example_circuit(&mut witgen, &inputs);
        let (f, h) = witgen.into_witnesses();

        assert_eq!(map.h_len(), h.bit_len(), "one M row per integer witness");
        assert_eq!(map.f_len(), f.bit_len() + 1, "M's columns are 1 || f");
        assert!(h.bit(0), "M's first row is the constant one");

        let point = point(map.h_len());
        let weights = eq_table(&point);
        let direct = (0..h.bit_len())
            .filter(|index| h.bit(*index))
            .fold(F128::new(0, 0), |sum, index| sum + weights[index]);

        let transposed = map.transpose_eq(&point).unwrap();
        let through_f = (0..f.bit_len())
            .filter(|index| f.bit(*index))
            .fold(transposed.constant_weight(), |sum, index| {
                sum + transposed.weights()[index]
            });

        assert_eq!(direct, through_f);
    }

    #[test]
    fn a_point_that_does_not_index_the_padded_rows_is_rejected() {
        let mut materializer = MTransposeGenerator::new(3);
        let inputs = materializer.take_boxed_inputs();
        example_circuit(&mut materializer, &inputs);
        let map = materializer.finish();

        let short = point(map.h_len()).len() - 1;
        assert_eq!(
            map.transpose_eq(&challenges(short)),
            Err(VirtualMapError::PointLengthMismatch)
        );
    }

    #[test]
    fn a_point_that_indexes_zero_padding_is_accepted() {
        let mut materializer = MTransposeGenerator::new(3);
        let inputs = materializer.take_boxed_inputs();
        example_circuit(&mut materializer, &inputs);
        let map = materializer.finish();
        let wide = challenges(point(map.h_len()).len() + 1);

        assert!(map.transpose_eq(&wide).is_ok());
    }

    #[test]
    fn the_digest_separates_different_matrices() {
        let mut first = MTransposeGenerator::new(3);
        let inputs = first.take_boxed_inputs();
        example_circuit(&mut first, &inputs);

        let mut second = MTransposeGenerator::new(3);
        let other: Box<[MatrixBit; 3]> = second.take_boxed_inputs();
        example_circuit(&mut second, &other);
        let extra = second.xor(other[0], other[2]);
        let _ = second.bitz::<1>(extra);

        assert_ne!(first.finish().digest(), second.finish().digest());
    }

    #[test]
    fn materialized_transpose_matches_constraint_generation() {
        let mut materializer = MTransposeGenerator::new(3);
        let inputs = materializer.take_boxed_inputs();
        example_circuit(&mut materializer, &inputs);
        let transpose = materializer.finish();

        let mut generator = ConstraintGenerator::new(3);
        let symbolic = generator.inputs();
        example_circuit(&mut generator, &symbolic);
        let matrices = generator.into_matrices();
        let r = challenges(matrices.m.row_count());

        let mut expected = vec![F128::new(0, 0); matrices.m.column_count()];
        for (row, challenge) in matrices.m.rows().iter().zip(&r) {
            for &column in row.positions() {
                expected[column] += *challenge;
            }
        }

        assert_eq!(transpose.row_count(), matrices.m.row_count());
        assert_eq!(transpose.column_count(), matrices.m.column_count());
        assert_eq!(transpose.apply(&r).unwrap(), expected);
    }

    #[test]
    fn materializes_a_full_sha256_compression_from_structure() {
        let mut materializer = MTransposeGenerator::new(COMPRESSION_INPUT_BITS);
        let matrix_inputs = materializer.take_boxed_inputs();
        let _ = compression_circuit(&mut materializer, &matrix_inputs);

        let transpose = materializer.finish();
        assert_eq!(
            transpose.column_count(),
            COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS + 1
        );
        assert_eq!(transpose.row_count(), 20_457);
        assert_eq!(transpose.nonzero_count(), 42_361);
        assert!(transpose.payload_bytes() < 194 * 1024);
    }

    #[test]
    fn parallel_and_sequential_matrix_gathers_agree() {
        const WIDTH: usize = 4096;
        let mut recorder = MTransposeRecorder::with_witnesses(WIDTH);
        for index in 0..WIDTH / 2 {
            let left = recorder.witness(index);
            let right = recorder.witness(index + WIDTH / 2);
            let root = recorder.xor(left, right);
            recorder.push_row(&root);
        }
        let transpose = recorder.finish();
        let r = challenges(transpose.row_count());
        let mut sequential = Vec::new();
        let mut parallel = Vec::new();
        transpose
            .apply_inner(&r, &mut sequential, Some(false))
            .unwrap();
        transpose
            .apply_inner(&r, &mut parallel, Some(true))
            .unwrap();
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn materialized_supports_can_exceed_the_inline_capacity() {
        let mut recorder = MTransposeRecorder::with_witnesses(6);
        let mut expression = MatrixBit::from(false);
        for index in 0..6 {
            expression = recorder.xor(expression, recorder.witness(index));
        }
        recorder.push_row(&expression);
        let transpose = recorder.finish();
        let challenge = F128::new(7, 11);
        let product = transpose.apply(&[F128::new(3, 5), challenge]).unwrap();

        assert_eq!(transpose.nonzero_count(), 7);
        assert_eq!(product[0], F128::new(3, 5));
        assert!(product[1..].iter().all(|value| *value == challenge));
    }

    #[test]
    fn apply_rejects_the_wrong_challenge_length() {
        let transpose = MTransposeRecorder::with_witnesses(0).finish();
        assert_eq!(
            transpose.apply(&[]),
            Err(MatrixApplyError {
                expected: 1,
                actual: 0,
            })
        );
    }
}
