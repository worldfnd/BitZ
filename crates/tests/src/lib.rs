//! Round-trip coverage for the prover and the verifier.
//!
//! This crate exists only so the two sides can be paired without either
//! depending on the other: `verifier` is written from the specification, and a
//! dev-dependency on `prover` would let a check drift into agreeing with how
//! the prover happens to compute something.
//!
//!
//! The fixtures live here rather than under `tests/` so they compile once
//! rather than once per test binary.

use circuit::matrix_transpose::{MTransposeGenerator, MaterializedMTranspose};
use circuit::sha256::{COMPRESSION_HINT_BITS, COMPRESSION_INPUT_BITS, compression_circuit};
use circuit::witgen::Witgen;
use common::{
    BitTable, BitZParams, LinearClaim, Root, Shape, TransposedWeights, VirtualMap, VirtualMapError,
    VirtualStatement, shape::PACK_BITS,
};
use crypto_primitives::LiftElement;
use field::{F128, Fq, gf128::smallest_generator};
use num_traits::{ConstOne, ConstZero};
use pcs::{HashKind, LigeritoProfile, Pcs, ProverData};
use poly::eq_table;
use prover::VirtualWitness;
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};
use rayon::prelude::*;
use transcript::{
    Proof, ProverState, VerificationError, VerificationResult, VerifierState, build_prover,
    build_verifier,
};

/// The specification's fixed modulus, `2^100 − 15`. Under it the fold bound
/// admits every row width up to `t = 27`, so the reference split
/// ([`Shape::for_log_bits`]) is admissible at every size in the window.
pub const Q: u128 = field::Q100;

/// Comb window. `FixedBasePow` always covers the full 128-bit exponent range;
/// this only trades table size against multiplies per call.
pub const WINDOW: u32 = 8;

/// Builds a packed witness and advances the RNG past its words.
pub fn packed_witness(shape: Shape, rng: &mut impl Rng) -> Vec<F128> {
    (0..1usize << shape.log_packed_len())
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect()
}

/// An instance whose claim actually holds, committed under a real scheme.
pub struct Instance {
    pub params: BitZParams<Q>,
    pub prover: prover::BitZProver<Q>,
    pub verifier: verifier::BitZVerifier<Q>,
    pub claim: LinearClaim<field::Fq<Q>>,
    pub pcs: Pcs,
    pub com: Root,
    pub data: ProverData,
    pub packed: Vec<F128>,
}

impl Instance {
    /// Builds a random witness and the target its own fold produces, so the
    /// claim is true by construction rather than by asserting the code agrees
    /// with itself.
    ///
    /// `mu` comes straight from the definition — `sum_j v^(2)_j pi_q(eta_j)`
    /// with `eta_j` read bit by bit — not from the reconstruction the verifier
    /// runs.
    pub fn honest(shape: Shape, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let params = BitZParams::<Q>::new(shape, smallest_generator()).unwrap();

        let packed = packed_witness(shape, &mut rng);
        let row_weights: Vec<Fq<Q>> = (0..shape.rows())
            .map(|_| Fq::from(sample_below_q(&mut rng)))
            .collect();
        let column_weights: Vec<Fq<Q>> = (0..shape.columns())
            .map(|_| Fq::from(sample_below_q(&mut rng)))
            .collect();

        let table = params.table(&packed).unwrap();
        let exponents: Vec<u128> = row_weights.iter().map(|weight| weight.lift()).collect();
        let target: Fq<Q> = (0..shape.columns())
            .map(|column| {
                let fold: u128 = (0..shape.rows())
                    .filter(|&row| table.bit(column, row))
                    .map(|row| exponents[row])
                    .sum();
                column_weights[column] * Fq::from(fold)
            })
            .sum();

        let claim = LinearClaim::new(&params, row_weights, column_weights, target).unwrap();

        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let (com, data) = pcs.commit(&packed).unwrap();

        Self {
            params,
            prover: prover::BitZProver::new(params, WINDOW),
            verifier: verifier::BitZVerifier::new(params, WINDOW),
            claim,
            pcs,
            com,
            data,
            packed,
        }
    }

    pub fn table(&self) -> BitTable<'_> {
        self.params.table(&self.packed).unwrap()
    }

    /// The same instance under a different claimed value.
    pub fn with_target(&self, target: Fq<Q>) -> LinearClaim<field::Fq<Q>> {
        LinearClaim::new(
            &self.params,
            self.claim.row_weights().to_vec(),
            self.claim.column_weights().to_vec(),
            target,
        )
        .unwrap()
    }
}

/// A uniform integer in `[0, q)`, rejection sampled so the weights are not
/// biased toward the low end of the range.
fn sample_below_q(rng: &mut ChaCha8Rng) -> u128 {
    loop {
        let candidate =
            (u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) & ((1u128 << 100) - 1);
        if candidate < Q {
            return candidate;
        }
    }
}

/// `m = 22` at the narrowest admissible row count: 128 rows, 32768 columns.
pub fn narrow_shape() -> Shape {
    Shape::new(7, 15).unwrap()
}

/// `m = 22` with wide rows: 8192 rows over 64 packed elements per column.
pub fn wide_shape() -> Shape {
    Shape::new(13, 9).unwrap()
}

/// `m = 28` at the reference split, `(t, s) = (17, 11)`: 131072 rows over
/// 2048 columns, a 32 MiB witness. Every other fixture sits at the floor, so
/// this is the only one whose cost scales the way a real instance does.
pub fn large_shape() -> Shape {
    reference_shape(28)
}

/// The reference split for `log_bits` bits, [`Shape::for_log_bits`].
pub fn reference_shape(log_bits: usize) -> Shape {
    Shape::for_log_bits(log_bits).unwrap()
}

const SESSION: &str = "bitz-tests";
const INSTANCE: &str = "fold-round-trip";

pub fn prover_transcript() -> ProverState {
    build_prover(SESSION, INSTANCE)
}

pub fn verifier_transcript(proof: &Proof) -> VerifierState<'_> {
    build_verifier(SESSION, INSTANCE, proof)
}

/// A batch of independent SHA-256 compressions as the two tables the virtual
/// pipeline works on (2.4. "Virtual F_2-linear transforms, NP-complete
/// multi-domain linear relations, and hybrid proof systems").
///
/// Compression `j` is column `j` of both tables. Its 20456 assignment cells
/// `h_j = M_0 (1, f_j)` fill the assignment column's first rows, with the
/// circuit's constant cell left out: it is public, and the transposition
/// accounts for it. The source column holds the compression's 7144
/// committed bits, block and state both inputs. Only power-of-two batches,
/// so every column is live.
#[derive(Debug, Clone)]
pub struct Sha256Batch {
    pub log_compressions: usize,
    /// `f`, column major, `2^SOURCE_LOG_ROWS` rows.
    pub source: Vec<F128>,
    /// `h`, column major, `2^ASSIGNMENT_LOG_ROWS` rows.
    pub assignment: Vec<F128>,
}

/// `7144 <= 2^13` committed bits per compression.
pub const SOURCE_LOG_ROWS: usize = 13;
/// `20456 <= 2^15` assignment cells per compression.
pub const ASSIGNMENT_LOG_ROWS: usize = 15;

/// Committed bits per independent compression: the 768 input bits and the
/// hint bits.
pub const SOURCE_BITS: usize = COMPRESSION_INPUT_BITS + COMPRESSION_HINT_BITS;

impl Sha256Batch {
    /// Runs `2^log_compressions` compressions on random blocks and states.
    pub fn generate(log_compressions: usize, seed: u64) -> Self {
        let columns: Vec<(Vec<F128>, Vec<F128>)> = (0..1u64 << log_compressions)
            .into_par_iter()
            .map(|compression| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed ^ compression.rotate_left(32));
                let inputs: [bool; COMPRESSION_INPUT_BITS] =
                    std::array::from_fn(|_| rng.next_u32() & 1 == 1);
                let mut witgen = Witgen::with_inputs_and_capacity(&inputs, SOURCE_BITS);
                let _ = compression_circuit(&mut witgen, &inputs);
                let (source, assignment) = witgen.into_witnesses();
                debug_assert_eq!(source.bit_len(), SOURCE_BITS);
                debug_assert_eq!(assignment.bit_len(), ASSIGNMENT_CELLS + 1);
                (
                    pack_column(source.words(), SOURCE_LOG_ROWS),
                    pack_column(&drop_constant_cell(assignment.words()), ASSIGNMENT_LOG_ROWS),
                )
            })
            .collect();

        let mut source = Vec::with_capacity(columns.len() << (SOURCE_LOG_ROWS - 7));
        let mut assignment = Vec::with_capacity(columns.len() << (ASSIGNMENT_LOG_ROWS - 7));
        for (f, h) in columns {
            source.extend(f);
            assignment.extend(h);
        }
        Self {
            log_compressions,
            source,
            assignment,
        }
    }

    pub fn source_shape(&self) -> Shape {
        Shape::new(SOURCE_LOG_ROWS, self.log_compressions).unwrap()
    }

    pub fn assignment_shape(&self) -> Shape {
        Shape::new(ASSIGNMENT_LOG_ROWS, self.log_compressions).unwrap()
    }
}

/// Assignment cells per compression, the constant cell excluded.
const ASSIGNMENT_CELLS: usize = 20_456;

/// One column of `2^log_rows` bits from little-endian words, zero padded.
fn pack_column(words: &[u64], log_rows: usize) -> Vec<F128> {
    (0..1usize << (log_rows - 7))
        .map(|element| {
            let word = |index: usize| words.get(index).copied().unwrap_or(0);
            F128::new(word(2 * element), word(2 * element + 1))
        })
        .collect()
}

/// The integer witness without its leading constant cell: every bit moved
/// down one place.
fn drop_constant_cell(words: &[u64]) -> Vec<u64> {
    (0..words.len())
        .map(|index| (words[index] >> 1) | words.get(index + 1).map_or(0, |next| next << 63))
        .collect()
}

/// `M_0^T` for one compression: `M_0` has a row per assignment cell (the
/// constant cell first) and a column per committed bit (the constant first).
fn compression_transpose() -> MaterializedMTranspose {
    let mut generator = MTransposeGenerator::new(COMPRESSION_INPUT_BITS);
    let inputs = generator.take_boxed_inputs::<COMPRESSION_INPUT_BITS>();
    let _ = compression_circuit(&mut generator, &inputs);
    generator.finish()
}

/// The batch's map over `2^log_compressions` independent compressions.
pub fn compression_map(log_compressions: usize) -> CompressionMap {
    CompressionMap {
        compression: compression_transpose(),
        log_compressions,
    }
}

/// The batch's map: `Id (x) M_0`, one compression per column.
///
/// The transposition goes column by column: compression `c`'s cells are
/// entries `c 2^15 ..` of `h` and its bits entries `c 2^13 ..` of `f`, so
/// `M_0^T` moves the weights on the one onto the other. The constant cell is
/// not in the table, so its row weighs nothing; the constant column's
/// weights add up across the compressions and leave the target.
#[derive(Debug)]
pub struct CompressionMap {
    pub compression: MaterializedMTranspose,
    pub log_compressions: usize,
}

impl VirtualMap for CompressionMap {
    fn transpose(&self, weights: &[F128]) -> Result<TransposedWeights, VirtualMapError> {
        if weights.len() < self.h_len() {
            return Err(VirtualMapError::WeightCountMismatch);
        }
        let cells = self.compression.row_count() - 1;
        let bits = self.compression.column_count() - 1;
        let columns: Vec<Vec<F128>> = (0..1usize << self.log_compressions)
            .into_par_iter()
            .map(|compression| {
                let mut challenges = Vec::with_capacity(cells + 1);
                challenges.push(F128::ZERO);
                challenges
                    .extend_from_slice(&weights[compression << ASSIGNMENT_LOG_ROWS..][..cells]);
                self.compression.apply(&challenges).unwrap()
            })
            .collect();
        let constant_weight = columns.iter().map(|column| column[0]).sum();
        let mut on_bits = vec![F128::ZERO; self.f_len() - 1];
        for (compression, column) in columns.iter().enumerate() {
            on_bits[compression << SOURCE_LOG_ROWS..][..bits].copy_from_slice(&column[1..]);
        }
        Ok(TransposedWeights::new(on_bits, constant_weight))
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"bitz/tests/compression-map/v1");
        hasher.update(&(self.log_compressions as u64).to_le_bytes());
        hasher.update(&self.compression.digest());
        *hasher.finalize().as_bytes()
    }

    /// Through the last compression's cells, in `h`'s layout.
    fn h_len(&self) -> usize {
        let last = (1usize << self.log_compressions) - 1;
        (last << ASSIGNMENT_LOG_ROWS) + self.compression.row_count() - 1
    }

    /// Through the last compression's bits, in `f`'s layout, plus the
    /// constant.
    fn f_len(&self) -> usize {
        let last = (1usize << self.log_compressions) - 1;
        1 + (last << SOURCE_LOG_ROWS) + self.compression.column_count() - 1
    }
}

/// Stands in for the PIOP (Step 3 of 5. "An end-to-end F2Z-based SNARK over
/// any finitely generated ring").
///
/// A PIOP ends on an evaluation claim `MLE[h](r) = y` on the assignment. The
/// mock squeezes `r` and has the prover compute `y` from `h` and send it,
/// where the PIOP would leave the verifier holding it.
#[derive(Debug, Clone, Copy)]
pub struct MockSpartan;

impl MockSpartan {
    /// Squeezes `r`, evaluates `MLE[h](r)` over `table`, and sends it.
    pub fn claim_prover<const Q: u128>(
        params: &BitZParams<Q>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
    ) -> LinearClaim<Fq<Q>> {
        let shape = params.shape();
        let row_point: Vec<Fq<Q>> = (0..shape.log_rows())
            .map(|_| sample_fq(transcript.verifier_message()))
            .collect();
        let column_point: Vec<Fq<Q>> = (0..shape.log_columns())
            .map(|_| sample_fq(transcript.verifier_message()))
            .collect();
        let row_weights = eq_table(&row_point);
        let column_weights = eq_table(&column_point);
        let target = evaluate_fq(table, &row_weights, &column_weights);
        transcript.prover_message(&target);

        LinearClaim::new(params, row_weights, column_weights, target).unwrap()
    }

    /// Squeezes the same `r` and reads `y`.
    pub fn claim_verifier<const Q: u128>(
        params: &BitZParams<Q>,
        transcript: &mut VerifierState<'_>,
    ) -> VerificationResult<LinearClaim<Fq<Q>>> {
        let shape = params.shape();
        let row_point: Vec<Fq<Q>> = (0..shape.log_rows())
            .map(|_| sample_fq(transcript.verifier_message()))
            .collect();
        let column_point: Vec<Fq<Q>> = (0..shape.log_columns())
            .map(|_| sample_fq(transcript.verifier_message()))
            .collect();
        let target = transcript.prover_message::<Fq<Q>>()?;

        LinearClaim::new(
            params,
            eq_table(&row_point),
            eq_table(&column_point),
            target,
        )
        .map_err(|_| VerificationError)
    }
}

/// A field element from 256 squeezed bits: the bias is below `2^-150`.
fn sample_fq<const Q: u128>(bytes: [u8; 32]) -> Fq<Q> {
    let (low, high) = bytes.split_at(16);
    let low = u128::from_le_bytes(low.try_into().unwrap());
    let high = u128::from_le_bytes(high.try_into().unwrap());
    let shift = Fq::from(u128::MAX) + Fq::ONE;
    Fq::from(high) * shift + Fq::from(low)
}

/// `<v^(1) (x) v^(2), table>` over `F_q`: each set bit adds its row weight,
/// each column is then scaled by its weight.
fn evaluate_fq<const Q: u128>(
    table: &BitTable<'_>,
    row_weights: &[Fq<Q>],
    column_weights: &[Fq<Q>],
) -> Fq<Q> {
    (0..table.shape().columns())
        .into_par_iter()
        .map(|column| {
            let mut sum = Fq::ZERO;
            for (index, element) in table.column(column).iter().enumerate() {
                let base = index << PACK_BITS;
                for (half, mut remaining) in [(0, element.lo), (64, element.hi)] {
                    while remaining != 0 {
                        sum += row_weights[base + half + remaining.trailing_zeros() as usize];
                        remaining &= remaining - 1;
                    }
                }
            }
            column_weights[column] * sum
        })
        .sum()
}

/// A committed batch with everything both sides hold: the assignment's
/// parameters, the source's opening scheme, and the map from the source to
/// the assignment.
pub struct Sha256Instance<const Q: u128> {
    pub batch: Sha256Batch,
    /// Shaped to the assignment.
    pub params: BitZParams<Q>,
    pub prover: prover::BitZProver<Q>,
    pub verifier: verifier::BitZVerifier<Q>,
    pub pcs: Pcs,
    pub com: Root,
    pub data: ProverData,
    pub map: CompressionMap,
}

/// A rejected SHA-256 proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sha256VerifyError {
    /// The PIOP mock's claimed value is missing or malformed.
    Claim,
    Verify(verifier::VerifyError),
}

impl<const Q: u128> Sha256Instance<Q> {
    /// Generates and commits `2^log_compressions` independent compressions.
    pub fn new(log_compressions: usize, profile: LigeritoProfile, seed: u64) -> Self {
        Self::commit(
            Sha256Batch::generate(log_compressions, seed),
            profile,
            compression_map(log_compressions),
        )
    }

    /// Commits a generated batch under `profile`, with `map` from its source
    /// to its assignment.
    pub fn commit(batch: Sha256Batch, profile: LigeritoProfile, map: CompressionMap) -> Self {
        let params = BitZParams::<Q>::new(batch.assignment_shape(), smallest_generator()).unwrap();
        let pcs = Pcs::new(&batch.source_shape(), profile, HashKind::Blake3).unwrap();
        let (com, data) = pcs.commit(&batch.source).unwrap();
        Self {
            batch,
            params,
            prover: prover::BitZProver::new(params, WINDOW),
            verifier: verifier::BitZVerifier::new(params, WINDOW),
            pcs,
            com,
            data,
            map,
        }
    }

    /// The statement both sides bind: the assignment's parameters, the
    /// source's shape, the map, and the mocked PIOP's claim.
    pub fn statement<'a>(
        &'a self,
        claim: &'a LinearClaim<Fq<Q>>,
    ) -> VirtualStatement<'a, Q, CompressionMap> {
        VirtualStatement::new(self.params, self.batch.source_shape(), &self.map, claim)
            .expect("the map fits both shapes")
    }

    /// The mocked PIOP's claim, then `prove_virtual` over the assignment,
    /// opened against the source.
    pub fn prove(&self) -> Proof {
        let mut transcript = prover_transcript();
        let table = self.params.table(&self.batch.assignment).unwrap();
        let claim = MockSpartan::claim_prover(&self.params, &table, &mut transcript);
        self.prover
            .prove_virtual(
                &self.statement(&claim),
                &self.pcs,
                &self.data,
                VirtualWitness {
                    committed_bits: self.batch.source.clone(),
                    virtual_bits: &self.batch.assignment,
                },
                &mut transcript,
            )
            .expect("honest batch");
        transcript.finish()
    }

    pub fn verify(&self, proof: &Proof) -> Result<(), Sha256VerifyError> {
        let mut transcript = verifier_transcript(proof);
        let claim = MockSpartan::claim_verifier(&self.params, &mut transcript)
            .map_err(|_| Sha256VerifyError::Claim)?;
        self.verifier
            .verify_virtual(&self.statement(&claim), &self.pcs, self.com, transcript)
            .map_err(Sha256VerifyError::Verify)
    }
}
