//! Independent SHA-256 compressions end to end, per batch size.
//!
//! Ported from f2z-pcs's `benches/sha256_compressions.rs`. The witness is the
//! circuit crate's; the committed bits `f` and the assignment `h = M f` are
//! laid out one compression per column (2.2. "Virtual F_2-linear transforms
//! in F2Z and NP-complete dually linear relations"). The PIOP
//! ([`MockSpartan`]) is mocked; the commitment, the fold, the grand product
//! (the `gkr` crate), the transposition onto `f`, the
//! post-GKR sumcheck and the opening are real. `sha256_steps` times the same
//! pipeline one step at a time.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p tests --bench sha256`.
//! Knobs:
//! - `BITZ_BENCH_SHAPES`: log2 of the compression counts, space or comma
//!   separated; default `9 10 11 12`. The commitment floor puts the minimum
//!   at 9; larger batches only cost time and memory.
//! - `BITZ_LIG_PROFILE`: `fast` (default), `slim` or `secure`.
//! - Divan's `DIVAN_SAMPLE_COUNT`, `DIVAN_SAMPLE_SIZE`, ...; the rows default
//!   to three samples of one run each, the source's repetition count.
//!
//! As in the source, `witness` is outside `prove`; unlike it, `commit` is its
//! own row rather than part of `prove`. Any other `BITZ_*` variable aborts the
//! run. The header lists each batch's table sizes and its proof size after
//! verifying that proof; the prover is deterministic, so the timed proofs are
//! the same bytes. The allocation columns count the benchmarking thread only.
//!
//! [`MockSpartan`]: tests::MockSpartan

use std::fmt::{self, Display};
use std::sync::OnceLock;

use divan::counter::ItemsCount;
use divan::{AllocProfiler, Bencher};
use field::Q100;
use host::wire_proof;
use pcs::LigeritoProfile;
use tests::{Sha256Batch, Sha256Instance};
use transcript::Proof;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

/// `q = 2^100 - 15`; the assignment table's `2^15` rows are well within what
/// it admits.
const Q: u128 = Q100;

const KNOWN_ENV: &[&str] = &["BITZ_BENCH_SHAPES", "BITZ_LIG_PROFILE"];

/// log2 of the compression counts: source tables of `2^22` to `2^25` bits.
const DEFAULT_SHAPES: &[usize] = &[9, 10, 11, 12];

const SEED: u64 = 0x_5348_4132_5600_0000;

fn main() {
    enforce_known_env();
    let _ = flock_core::init_perf_thread_pool();

    println!(
        "profile {:?}, {} threads",
        profile(),
        rayon::current_num_threads()
    );
    for shape in shapes() {
        let fixture = fixture(shape);
        let batch = &fixture.instance.batch;
        println!(
            "{shape}: {} compressions, f 2^{} bits, h 2^{} bits, proof {} B (narg {} B + hints {} B), verified",
            1usize << shape.log_compressions,
            batch.source_shape().log_bits(),
            batch.assignment_shape().log_bits(),
            fixture.bytes.len(),
            fixture.proof.narg_string.len(),
            fixture.proof.hints.len(),
        );
    }

    divan::main();
}

/// Aborts on any exported `BITZ_*` variable this bench does not know.
fn enforce_known_env() {
    let mut unknown: Vec<String> = std::env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .filter(|key| key.starts_with("BITZ_") && !KNOWN_ENV.contains(&key.as_str()))
        .collect();
    if unknown.is_empty() {
        return;
    }
    unknown.sort();
    eprintln!(
        "error: unknown BITZ_* variable(s): {}; known: {}",
        unknown.join(", "),
        KNOWN_ENV.join(", ")
    );
    std::process::exit(2);
}

fn profile() -> LigeritoProfile {
    match std::env::var("BITZ_LIG_PROFILE").as_deref() {
        Err(_) | Ok("fast") => LigeritoProfile::Fast,
        Ok("slim") => LigeritoProfile::Slim,
        Ok("secure") => LigeritoProfile::Secure,
        Ok(other) => panic!("BITZ_LIG_PROFILE: unknown profile `{other}` (fast | slim | secure)"),
    }
}

/// `2^log_compressions` compressions; the row of the divan table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BenchShape {
    log_compressions: usize,
}

impl Display for BenchShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "2^{}", self.log_compressions)
    }
}

fn shapes() -> Vec<BenchShape> {
    let exponents: Vec<usize> = match std::env::var("BITZ_BENCH_SHAPES") {
        Ok(list) => list
            .split([',', ' '])
            .filter(|token| !token.is_empty())
            .map(|token| {
                token.parse().unwrap_or_else(|_| {
                    panic!("BITZ_BENCH_SHAPES: `{token}` is not a log2 compression count")
                })
            })
            .collect(),
        Err(_) => DEFAULT_SHAPES.to_vec(),
    };
    exponents
        .into_iter()
        .map(|log_compressions| BenchShape { log_compressions })
        .collect()
}

/// One committed batch with the proof the timed rows reproduce.
struct Fixture {
    instance: Sha256Instance<Q>,
    proof: Proof,
    bytes: Vec<u8>,
}

impl Fixture {
    fn new(shape: BenchShape, profile: LigeritoProfile, seed: u64) -> Self {
        let instance = Sha256Instance::<Q>::new(shape.log_compressions, profile, seed);
        let proof = instance.prove();
        let bytes = wire_proof::encode(&proof);
        instance
            .verify(&proof)
            .expect("the fixture's own proof verifies");
        Self {
            instance,
            proof,
            bytes,
        }
    }
}

static FIXTURES: OnceLock<Vec<(BenchShape, Fixture)>> = OnceLock::new();

/// Every shape's fixture is built on the first call, so the rows share one
/// batch per shape.
fn fixture(shape: BenchShape) -> &'static Fixture {
    let fixtures = FIXTURES.get_or_init(|| {
        let profile = profile();
        shapes()
            .into_iter()
            .map(|shape| {
                let seed = SEED ^ shape.log_compressions as u64;
                (shape, Fixture::new(shape, profile, seed))
            })
            .collect()
    });
    &fixtures
        .iter()
        .find(|(candidate, _)| *candidate == shape)
        .expect("every shape is built up front")
        .1
}

/// The circuit crate's witness generation, into the two tables. Outside
/// `prove`, as in the source.
#[divan::bench(args = shapes(), sample_count = 3, sample_size = 1)]
fn witness(bencher: Bencher, shape: BenchShape) {
    bencher
        .counter(ItemsCount::new(1usize << shape.log_compressions))
        .bench_local(|| Sha256Batch::generate(shape.log_compressions, SEED));
}

/// Step 1: the commitment to `f`.
#[divan::bench(args = shapes(), sample_count = 3, sample_size = 1)]
fn commit(bencher: Bencher, shape: BenchShape) {
    let instance = &fixture(shape).instance;
    bencher
        .counter(ItemsCount::new(1usize << shape.log_compressions))
        .bench_local(|| instance.pcs.commit(&instance.batch.source).unwrap());
}

/// Steps 3 to 6 on the committed batch: the mocked PIOP's claim, the fold,
/// the grand product, the transposition, the sumcheck, the opening.
#[divan::bench(args = shapes(), sample_count = 3, sample_size = 1)]
fn prove(bencher: Bencher, shape: BenchShape) {
    let instance = &fixture(shape).instance;
    bencher
        .counter(ItemsCount::new(1usize << shape.log_compressions))
        .bench_local(|| instance.prove());
}

#[divan::bench(args = shapes(), sample_count = 3, sample_size = 1)]
fn verify(bencher: Bencher, shape: BenchShape) {
    let fixture = fixture(shape);
    bencher
        .counter(ItemsCount::new(1usize << shape.log_compressions))
        .bench_local(|| fixture.instance.verify(&fixture.proof).unwrap());
}
