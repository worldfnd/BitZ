//! Honest instances the round-trip tests run against.
//!
//! Cargo compiles this module separately into every integration test binary,
//! so a helper only one of them uses is dead code in the others.
#![allow(dead_code)]

use common::{BitTable, CoreStatement, Fold, OpeningClaim, ReductionInput, Root, Shape};
use crypto_primitives::LiftElement;
use field::{F128, FixedBasePow, Fq, gf128::smallest_generator};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use transcript::{Proof, ProverState, VerifierState, build_prover, build_verifier};

/// The largest prime below `2^114`, the top of the paper's sampling range.
pub const Q: u128 = (1 << 114) - 11;

/// Comb window. `FixedBasePow` always covers the full 128-bit exponent range;
/// this only trades table size against multiplies per call.
pub const WINDOW: u32 = 8;

/// An instance whose claim actually holds.
pub struct Instance {
    pub statement: CoreStatement<Q>,
    pub com: Root,
    pub words: Vec<u64>,
    pub generator: FixedBasePow,
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

        let words: Vec<u64> = (0..(1 << shape.m()) / 64).map(|_| rng.next_u64()).collect();
        let row_weights: Vec<Fq<Q>> = (0..shape.rows())
            .map(|_| Fq::from(sample_below_q(&mut rng)))
            .collect();
        let column_weights: Vec<Fq<Q>> = (0..shape.columns())
            .map(|_| Fq::from(sample_below_q(&mut rng)))
            .collect();

        let table = BitTable::new(shape, &words).unwrap();
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

        let statement = CoreStatement::new(
            shape,
            smallest_generator(),
            row_weights,
            column_weights,
            target,
        )
        .unwrap();

        Self {
            statement,
            com: Root([9u8; 32]),
            words,
            generator: FixedBasePow::new(smallest_generator(), WINDOW),
        }
    }

    pub fn table(&self) -> BitTable<'_> {
        BitTable::new(*self.statement.shape(), &self.words).unwrap()
    }

    /// The same instance under a different claimed value.
    pub fn with_target(&self, target: Fq<Q>) -> CoreStatement<Q> {
        CoreStatement::new(
            *self.statement.shape(),
            self.statement.generator(),
            self.statement.row_weights().to_vec(),
            self.statement.column_weights().to_vec(),
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

/// `m = 22` at the widest row count this `q` admits: 16384 rows over 256 words
/// per column, with folds pressed against the u128 ceiling.
pub fn wide_shape() -> Shape {
    Shape::new(14, 8).unwrap()
}

const SESSION: &str = "f2z-tests";
const INSTANCE: &str = "fold-round-trip";

pub fn prover_transcript() -> ProverState {
    build_prover(SESSION, INSTANCE)
}

pub fn verifier_transcript(proof: &Proof) -> VerifierState<'_> {
    build_verifier(SESSION, INSTANCE, proof)
}

/// Stands in for #8. Squeezes a point of the width the opening expects and
/// moves one value across, so the transcript order around it is exercised and
/// its argument is not.
pub struct EchoReduction;

impl prover::Reduction<Q> for EchoReduction {
    type Error = ();

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        _table: &BitTable<'_>,
        transcript: &mut ProverState,
    ) -> Result<OpeningClaim, Self::Error> {
        let point = (0..input.statement.shape().m())
            .map(|_| transcript.verifier_message())
            .collect();
        Ok(claim(point, input.fold))
    }
}

impl verifier::Reduction<Q> for EchoReduction {
    type Error = ();

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        transcript: &mut VerifierState<'_>,
    ) -> Result<OpeningClaim, Self::Error> {
        let point = (0..input.statement.shape().m())
            .map(|_| transcript.verifier_message())
            .collect();
        Ok(claim(point, input.fold))
    }
}

fn claim(point: Vec<F128>, fold: &Fold) -> OpeningClaim {
    OpeningClaim {
        point,
        target: fold.e0,
    }
}
