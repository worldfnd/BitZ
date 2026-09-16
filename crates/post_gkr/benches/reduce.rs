//! The post-GKR sumcheck on its own, prover and verifier.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p post_gkr --bench reduce`.
//! Without the flag the field falls back to its portable kernel, which is
//! the whole difference on every multiplication.
//!
//! The claim holds by construction over a pseudo-random witness, so the
//! prover's own check passes and the rounds run. One weight per bit, the
//! shape a virtual map leaves: `m = 22`, the commitment floor, and
//! `m = 25`, the SHA-256 batch as the step bench commits it.

use std::sync::OnceLock;

use common::{LinearClaim, Shape};
use divan::Bencher;
use divan::counter::ItemsCount;
use field::F128;
use num_traits::ConstZero;
use rand_core::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use transcript::{Proof, build_prover, build_verifier};

const LOG_BITS: &[usize] = &[22, 25];

const SEED: u64 = 0x_5245_4455_4345_0000;

fn main() {
    divan::main();
}

/// One witness and claim per size, built on first use.
struct Fixture {
    packed: Vec<F128>,
    claim: LinearClaim<F128>,
}

static FIXTURES: OnceLock<Vec<(usize, Fixture)>> = OnceLock::new();

fn fixture(log_bits: usize) -> &'static Fixture {
    let fixtures = FIXTURES.get_or_init(|| {
        LOG_BITS
            .iter()
            .map(|&log_bits| (log_bits, Fixture::new(log_bits)))
            .collect()
    });
    &fixtures
        .iter()
        .find(|(candidate, _)| *candidate == log_bits)
        .expect("every size is built up front")
        .1
}

fn random(rng: &mut Pcg64, count: usize) -> Vec<F128> {
    (0..count)
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect()
}

impl Fixture {
    fn new(log_bits: usize) -> Self {
        let mut rng = Pcg64::seed_from_u64(SEED ^ log_bits as u64);
        let packed = random(&mut rng, 1 << (log_bits - 7));
        let weights = random(&mut rng, 1 << log_bits);
        let mut target = F128::ZERO;
        for (index, element) in packed.iter().enumerate() {
            let mut bits = u128::from(element.lo) | (u128::from(element.hi) << 64);
            while bits != 0 {
                target += weights[(index << 7) | bits.trailing_zeros() as usize];
                bits &= bits - 1;
            }
        }
        let shape = Shape::new(log_bits, 0).unwrap();
        let claim =
            LinearClaim::from_shape(&shape, weights, vec![F128::from(1u64)], target).unwrap();
        Self { packed, claim }
    }

    fn proof(&self) -> Proof {
        let mut transcript = build_prover("post_gkr-bench", "reduce");
        post_gkr::prove(&self.claim, &self.packed, &mut transcript).unwrap();
        transcript.finish()
    }
}

fn bits(log_bits: usize) -> ItemsCount {
    ItemsCount::new(1usize << log_bits)
}

#[divan::bench(args = LOG_BITS, sample_count = 5, sample_size = 1)]
fn prove(bencher: Bencher, log_bits: usize) {
    let fixture = fixture(log_bits);
    bencher.counter(bits(log_bits)).bench_local(|| {
        let mut transcript = build_prover("post_gkr-bench", "reduce");
        post_gkr::prove(&fixture.claim, &fixture.packed, &mut transcript).unwrap()
    });
}

#[divan::bench(args = LOG_BITS, sample_count = 5, sample_size = 1)]
fn verify(bencher: Bencher, log_bits: usize) {
    let fixture = fixture(log_bits);
    let proof = fixture.proof();
    bencher.counter(bits(log_bits)).bench_local(|| {
        let mut transcript = build_verifier("post_gkr-bench", "reduce", &proof);
        post_gkr::verify(&fixture.claim, &mut transcript).unwrap()
    });
}
