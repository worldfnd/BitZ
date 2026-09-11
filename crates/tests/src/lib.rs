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

use common::{
    BitTable, F2ZParams, LinearClaim, OpeningQuery, ReductionInput, Root, Shape, shape::PACK_BITS,
};
use crypto_primitives::LiftElement;
use field::{F128, Fq, gf128::smallest_generator};
use pcs::{HashKind, LigeritoProfile, Pcs, ProverData};
use poly::eq_table;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use transcript::{Proof, ProverState, VerifierState, build_prover, build_verifier};

/// The largest prime below `2^114`, the top of the paper's sampling range.
pub const Q: u128 = (1 << 114) - 11;

/// Comb window. `FixedBasePow` always covers the full 128-bit exponent range;
/// this only trades table size against multiplies per call.
pub const WINDOW: u32 = 8;

/// An instance whose claim actually holds, committed under a real scheme.
pub struct Instance {
    pub params: F2ZParams<Q>,
    pub prover: prover::F2ZProver<Q>,
    pub verifier: verifier::F2ZVerifier<Q>,
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
        let params = F2ZParams::<Q>::new(shape, smallest_generator()).unwrap();

        let packed: Vec<F128> = (0..(1 << shape.log_bits()) / 128)
            .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
            .collect();
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
            prover: prover::F2ZProver::new(params, WINDOW),
            verifier: verifier::F2ZVerifier::new(params, WINDOW),
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
            (u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64())) & ((1u128 << 114) - 1);
        if candidate < Q {
            return candidate;
        }
    }
}

/// `m = 22` at the narrowest admissible row count: 128 rows, 32768 columns.
pub fn narrow_shape() -> Shape {
    Shape::new(7, 15).unwrap()
}

/// `m = 22` at the widest row count this `q` admits: 8192 rows over 64 packed
/// elements per column, with folds pressed against the u128 ceiling.
pub fn wide_shape() -> Shape {
    Shape::new(13, 9).unwrap()
}

/// `m = 28`, the smallest shape past the profile's floor: 8192 rows over
/// 32768 columns, a 32 MiB witness. Every other fixture sits at the floor, so
/// this is the only one whose cost scales the way a real instance does.
pub fn large_shape() -> Shape {
    Shape::new(13, 15).unwrap()
}

const SESSION: &str = "f2z-tests";
const INSTANCE: &str = "fold-round-trip";

pub fn prover_transcript() -> ProverState {
    build_prover(SESSION, INSTANCE)
}

pub fn verifier_transcript(proof: &Proof) -> VerifierState<'_> {
    build_verifier(SESSION, INSTANCE, proof)
}

/// Stands in for #8.
///
/// Squeezes the point the opening expects and hands over an evaluation that is
/// actually true, so step 6 runs against the real witness rather than against a
/// claim it would reject out of hand.
///
/// The target crosses on the transcript because the verifier has no witness to
/// derive it from. That is exactly what #8 replaces: the real reduction leaves
/// the verifier computing `mu'` from the sumcheck, so nothing has to be
/// trusted. Sending it makes the round trip exercise the wiring, not the
/// argument -- a prover that lies here is caught by its own opening, not by the
/// verifier.
#[derive(Debug)]
pub struct HonestStub;

impl prover::Reduction<Q> for HonestStub {
    type Error = ();

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
    ) -> Result<OpeningQuery, Self::Error> {
        let point: Vec<F128> = (0..input.params.shape().log_bits())
            .map(|_| transcript.verifier_message())
            .collect();
        let target = evaluate(table, &point);
        transcript.prover_message(&target);

        Ok(OpeningQuery::Mle { point, target })
    }
}

impl verifier::Reduction<Q> for HonestStub {
    type Error = ();

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningQuery, Self::Error> {
        let point = (0..input.params.shape().log_bits())
            .map(|_| transcript.verifier_message())
            .collect();
        let target = transcript.prover_message::<F128>().map_err(|_| ())?;

        Ok(OpeningQuery::Mle { point, target })
    }
}

/// The multilinear extension of the committed bits at `point`.
///
/// Splitting `point` at the pack width is what keeps this affordable: the
/// equality table over the high coordinates has one entry per packed element
/// rather than one per bit, and each set bit adds its element's weight to the
/// low coordinate it sits at. Materialising `eq` over all `m` coordinates would
/// be `2^m` field elements.
fn evaluate(table: &BitTable<'_>, point: &[F128]) -> F128 {
    let shape = table.shape();
    let (low, high) = point.split_at(PACK_BITS as usize);
    let eq_low = eq_table(low);
    let eq_high = eq_table(high);
    let per_column = shape.rows() >> PACK_BITS;

    let mut sums = vec![F128::default(); 1usize << PACK_BITS];
    for column in 0..shape.columns() {
        for (index, element) in table.column(column).iter().enumerate() {
            let weight = eq_high[column * per_column + index];
            for (base, mut remaining) in [(0u32, element.lo), (64, element.hi)] {
                while remaining != 0 {
                    sums[(base + remaining.trailing_zeros()) as usize] += weight;
                    remaining &= remaining - 1;
                }
            }
        }
    }

    eq_low
        .iter()
        .zip(&sums)
        .fold(F128::default(), |total, (&weight, &sum)| {
            total + weight * sum
        })
}
