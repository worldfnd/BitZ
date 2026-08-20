use std::fmt;

use divan::{Bencher, black_box};
use field::F128 as F2zF128;
use flock_core::challenger::FsChallenger;
use flock_core::field::F128 as FlockF128;
use flock_core::hash::HashKind as FlockHashKind;
use flock_core::pcs::ligerito::{LigeritoProfile as FlockProfile, ProverConfig};
use flock_core::pcs::{
    Commitment as FlockCommitment, PcsParams, ProverData as FlockProverData, commit,
    open_batch_mixed_ligerito_with_precomputed_s_hat_v,
};
use flock_core::zerocheck::PaddingSpec;
use pcs::{
    CommitScheme, HashKind, LigeritoProfile, OpeningQuery, Pcs, ProverData, ScopedOpeningQuery,
    StatementBinding,
};
use transcript::build_prover;

const SESSION: &[u8] = b"pcs-open-comparison-v1";
const INSTANCE: &[u8] = b"full-opening";
// Bit i is one exactly when the six-bit integer i has odd parity.
// Bits 0..7 are 0,1,1,0,1,0,0,1, so their little-endian byte is 0x96.
// Bit 6 flips parity in packed positions 64..127, so the high word is `!lo`.
// Odd parity in the packed-row index also complements the low-word table.
const PARITY_BITS: u64 = 0x6996_9669_9669_6996;
const CASES: &[Case] = &[
    Case { m: 22, claims: 1 },
    Case { m: 22, claims: 2 },
    Case { m: 26, claims: 2 },
    Case { m: 26, claims: 4 },
    Case { m: 30, claims: 2 },
];

#[derive(Clone, Copy, Debug)]
struct Case {
    m: usize,
    claims: usize,
}

impl Case {
    fn packed_len(self) -> usize {
        1 << (self.m - flock_core::pcs::LOG_PACKING)
    }
}

impl fmt::Display for Case {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "m{}_q{}", self.m, self.claims)
    }
}

struct F2zFixture {
    pcs: Pcs,
    data: ProverData,
    witness: Vec<F2zF128>,
    queries: Vec<OpeningQuery>,
}

struct FlockFixture {
    commitment: FlockCommitment,
    data: FlockProverData,
    witness: Vec<FlockF128>,
    x_outers: Vec<Vec<FlockF128>>,
    padding: PaddingSpec,
    config: ProverConfig,
}

fn main() {
    let automatic_workers = flock_core::init_perf_thread_pool();
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    println!("Available execution units: {available}");
    println!("Rayon workers: {}", rayon::current_num_threads());
    if let Some(workers) = automatic_workers {
        println!("Flock performance-core default: {workers}");
    } else if let Ok(workers) = std::env::var("RAYON_NUM_THREADS") {
        println!("RAYON_NUM_THREADS: {workers}");
    }
    divan::main();
}

#[divan::bench(args = CASES, sample_count = 5, sample_size = 1)]
fn f2z_full_open(bencher: Bencher, case: Case) {
    bencher
        .with_inputs(|| f2z_fixture(case))
        .bench_local_values(|fixture| {
            let F2zFixture {
                pcs,
                data,
                witness,
                queries,
            } = fixture;
            let scoped_queries = queries
                .iter()
                .enumerate()
                .map(|(scope, query)| ScopedOpeningQuery::new(scope as u32, query))
                .collect::<Vec<_>>();
            let mut transcript = build_prover(SESSION, INSTANCE);
            pcs.prove_lin_batch(
                &data,
                witness,
                &scoped_queries,
                StatementBinding::Bind,
                &mut transcript,
            )
            .expect("F2Z benchmark opening must succeed");
            black_box(transcript.finish())
        });
}

#[divan::bench(args = CASES, sample_count = 5, sample_size = 1)]
fn flock_full_open(bencher: Bencher, case: Case) {
    bencher
        .with_inputs(|| flock_fixture(case))
        .bench_local_values(|fixture| {
            let FlockFixture {
                commitment,
                data,
                witness,
                x_outers,
                padding,
                config,
            } = fixture;
            let x_outer_refs = x_outers.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let mut challenger = FsChallenger::with_hash(SESSION, FlockHashKind::Blake3);
            black_box(open_batch_mixed_ligerito_with_precomputed_s_hat_v(
                witness,
                &data,
                &commitment,
                &x_outer_refs,
                &[],
                &[],
                &padding,
                &config,
                &mut challenger,
            ))
        });
}

fn f2z_fixture(case: Case) -> F2zFixture {
    let pcs = Pcs::new(case.m, LigeritoProfile::Fast, HashKind::Blake3)
        .expect("F2Z benchmark configuration must be valid");
    // Boolean parity has a dense packed witness and evaluates to the sum of the point.
    let witness = (0..case.packed_len())
        .map(|index| {
            let lo = if index.count_ones() & 1 == 0 {
                PARITY_BITS
            } else {
                !PARITY_BITS
            };
            F2zF128::new(lo, !lo)
        })
        .collect::<Vec<_>>();
    let queries = (0..case.claims)
        .map(|claim| {
            let point = (0..case.m)
                .map(|coordinate| F2zF128::from((claim * case.m + coordinate + 1) as u64))
                .collect::<Vec<_>>();
            let target = point
                .iter()
                .copied()
                .fold(F2zF128::default(), |sum, value| sum + value);
            OpeningQuery { point, target }
        })
        .collect::<Vec<_>>();
    let (_, data) = pcs
        .commit(&witness)
        .expect("F2Z benchmark commitment must succeed");
    F2zFixture {
        pcs,
        data,
        witness,
        queries,
    }
}

fn flock_fixture(case: Case) -> FlockFixture {
    let profile = FlockProfile::Fast;
    let params = PcsParams {
        m: case.m,
        log_inv_rate: profile.log_inv_rate(),
        log_batch_size: 6,
        profile,
        merkle_hash: FlockHashKind::Blake3,
    };
    let witness = (0..case.packed_len())
        .map(|index| {
            let lo = if index.count_ones() & 1 == 0 {
                PARITY_BITS
            } else {
                !PARITY_BITS
            };
            FlockF128::new(lo, !lo)
        })
        .collect::<Vec<_>>();
    let x_outers = (0..case.claims)
        .map(|claim| {
            (6..case.m)
                .map(|coordinate| FlockF128::new((claim * case.m + coordinate + 1) as u64, 0))
                .collect()
        })
        .collect();
    let config = params
        .ligerito_prover_config()
        .expect("Flock benchmark configuration must be valid");
    let (commitment, data) = commit(&witness, &params);
    FlockFixture {
        commitment,
        data,
        witness,
        x_outers,
        padding: PaddingSpec::dense(case.m),
        config,
    }
}
