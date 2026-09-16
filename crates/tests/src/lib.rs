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

use common::{BitTable, BitZParams, LinearClaim, Root, Shape};
use crypto_primitives::LiftElement;
use field::{F128, Fq, gf128::smallest_generator};
use pcs::{HashKind, LigeritoProfile, Pcs, ProverData};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use transcript::{Proof, ProverState, VerifierState, build_prover, build_verifier};

/// The specification's fixed modulus, `2^100 − 15`. Under it the fold bound
/// admits every row width up to `t = 27`, so the reference split
/// ([`Shape::for_log_bits`]) is admissible at every size in the window.
pub const Q: u128 = field::Q100;

/// Comb window. `FixedBasePow` always covers the full 128-bit exponent range;
/// this only trades table size against multiplies per call.
pub const WINDOW: u32 = 8;

/// Builds a packed witness and advances the RNG past its words.
pub fn packed_witness(shape: Shape, rng: &mut impl RngCore) -> Vec<F128> {
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
