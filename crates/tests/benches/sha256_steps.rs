//! The SHA-256 pipeline one step at a time, prover and verifier.
//!
//! The same pipeline as `sha256`, driven through the crates' public steps
//! with a clock around each: the commitment (`pcs`), the mocked PIOP's claim,
//! the binding and the fold (`prover`/`verifier`), the grand product's leaf
//! claim (the `gkr` crate), its transposition onto the committed bits, and
//! the opening (`pcs`), which runs the post-GKR sumcheck (`post_gkr`) before
//! the ring switch. The sumcheck is also clocked alone, on a scratch
//! transcript, since the opening does not expose it as a step. Medians over
//! the repetitions are printed per step. As in the source, `prove` includes
//! the commitment and excludes witness generation.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p tests --bench sha256_steps`.
//! Knobs:
//! - `BITZ_BENCH_SHAPES`: log2 of the compression counts, space or comma
//!   separated; default `9 10 11 12`.
//! - `BITZ_BENCH_REPS`: measured repetitions after one warm-up; default 3.
//! - `BITZ_LIG_PROFILE`: `fast` (default), `slim` or `secure`.
//!
//! Each repetition generates a fresh batch and verifies its proof. The
//! warm-up also checks that the step-driven proof is byte for byte the one
//! `prove` produces, so the split cannot drift from the protocol.

use std::time::{Duration, Instant};

use common::{OpeningQuery, VirtualMap};
use field::{F128, Q100};
use pcs::{CommitScheme, LigeritoProfile, StatementBinding};
use tests::{MockSpartan, Sha256Batch, Sha256Instance, prover_transcript, verifier_transcript};
use transcript::{Proof, ProverState, VerifierState};

const Q: u128 = Q100;

const KNOWN_ENV: &[&str] = &["BITZ_BENCH_REPS", "BITZ_BENCH_SHAPES", "BITZ_LIG_PROFILE"];
const DEFAULT_SHAPES: &[usize] = &[9, 10, 11, 12];
const SEED: u64 = 0x_5348_4132_5654_4550;

fn main() {
    enforce_known_env();
    let _ = flock_core::init_perf_thread_pool();
    let profile = profile();
    let reps = reps();

    println!(
        "SHA-256 compressions, step by step: profile {profile:?}, {} threads, {reps} reps after 1 warm-up",
        rayon::current_num_threads()
    );
    for log_compressions in shapes() {
        flock_core::scratch::clear();
        // Setup is excluded: the map, the parameters and the scheme are
        // public and shape-only. One committed batch stands for them across
        // the reps.
        let setup = Instant::now();
        bench_instance(
            Sha256Instance::<Q>::new(log_compressions, profile, SEED),
            setup,
            reps,
        );
    }
}

fn bench_instance(instance: Sha256Instance<Q>, setup: Instant, reps: usize) {
    let setup = setup.elapsed();
    let batch = &instance.batch;
    let log_compressions = batch.log_compressions;
    println!(
        "\n=== 2^{log_compressions} = {} compressions: f 2^{} bits, h 2^{} bits ===",
        1usize << log_compressions,
        batch.source_shape().log_bits(),
        batch.assignment_shape().log_bits(),
    );
    println!(
        "  setup (excluded): {}   witness + commit + map",
        fmt(setup)
    );

    // Warm-up on the setup's own batch, and the check that the steps are the
    // protocol: the one-call proof of the same batch must be the same bytes.
    let (warm, proof) = run(&instance, &instance.batch);
    assert_eq!(proof, instance.prove(), "the steps must reproduce prove");
    instance.verify(&proof).expect("the warm-up proof verifies");
    drop(warm);

    let mut witness = Vec::with_capacity(reps);
    let mut runs = Vec::with_capacity(reps);
    let mut last = None;
    for rep in 0..reps {
        let started = Instant::now();
        let batch = Sha256Batch::generate(log_compressions, SEED ^ (rep as u64 + 1));
        witness.push(started.elapsed());
        let (timings, proof) = run(&instance, &batch);
        runs.push(timings);
        last = Some(proof);
    }
    let proof = last.expect("at least one rep");

    print_side("prove", &runs, |run| &run.prove, |run| run.prove_total());
    print_side("verify", &runs, |run| &run.verify, |run| run.verify_total());
    println!("  witness (excluded): {}", fmt(median(witness.into_iter())));
    println!(
        "  proof: {} B = narg {} B + hints {} B",
        proof.narg_string.len() + proof.hints.len(),
        proof.narg_string.len(),
        proof.hints.len()
    );
}

/// One repetition's step timings.
struct Run {
    prove: Vec<(&'static str, Duration)>,
    verify: Vec<(&'static str, Duration)>,
}

impl Run {
    fn prove_total(&self) -> Duration {
        self.prove.iter().map(|(_, time)| *time).sum()
    }

    fn verify_total(&self) -> Duration {
        self.verify.iter().map(|(_, time)| *time).sum()
    }
}

/// Proves and verifies `batch` step by step on `instance`'s public setup.
/// The batch is committed inside the prover's clock.
fn run(instance: &Sha256Instance<Q>, batch: &Sha256Batch) -> (Run, Proof) {
    let mut prove = Vec::new();
    let mut time = |label, work: &mut dyn FnMut()| {
        let started = Instant::now();
        work();
        prove.push((label, started.elapsed()));
    };

    let committed = batch.source.clone();
    let mut data = None;
    time("commit", &mut || {
        data = Some(instance.pcs.commit(&batch.source).unwrap());
    });
    let (com, data) = data.unwrap();

    let mut transcript = prover_transcript();
    let table = instance.params.table(&batch.assignment).unwrap();
    let mut claim = None;
    time("PIOP claim (mock)", &mut || {
        claim = Some(MockSpartan::claim_prover(
            &instance.params,
            &table,
            &mut transcript,
        ));
    });
    let claim = claim.unwrap();

    let statement = instance.statement(&claim);
    time("bind", &mut || {
        transcript.public_message(b"bitz/virtual-statement/v1");
        transcript.public_message(&com.0);
        transcript.public_message(statement.params());
        transcript.public_message(&instance.map.digest());
        transcript.public_message(&claim);
    });

    let mut fold = None;
    time("fold", &mut || {
        fold = Some(
            instance
                .prover
                .send_fold(&claim, &table, &mut transcript)
                .unwrap(),
        );
    });
    let fold = fold.unwrap();

    let mut leaf = None;
    time("GKR leaf", &mut || {
        leaf = Some(prover::gkr_reduce(&mut transcript, &fold, &table).unwrap());
    });

    let mut query = None;
    time("transposition", &mut || {
        query = Some(statement.transpose_query(leaf.take().unwrap()).unwrap());
    });
    let query = query.unwrap();

    // Off the protocol's transcript: the opening runs this inside itself.
    let mut scratch = None;
    time("sumcheck (alone)", &mut || {
        let mut transcript = prover_transcript();
        sumcheck_prover(&query, &batch.source, &mut transcript);
        scratch = Some(transcript.finish());
    });
    let scratch = scratch.unwrap();

    let mut committed = Some(committed);
    time("opening", &mut || {
        instance
            .pcs
            .prove_lin(
                &data,
                committed.take().unwrap(),
                &query,
                StatementBinding::Bind,
                &mut transcript,
            )
            .unwrap();
    });
    let proof = transcript.finish();

    let mut verify = Vec::new();
    let mut time = |label, work: &mut dyn FnMut()| {
        let started = Instant::now();
        work();
        verify.push((label, started.elapsed()));
    };

    let mut transcript = verifier_transcript(&proof);
    let mut claim = None;
    time("PIOP claim (mock)", &mut || {
        claim = Some(MockSpartan::claim_verifier(&instance.params, &mut transcript).unwrap());
    });
    let claim = claim.unwrap();

    let statement = instance.statement(&claim);
    time("bind", &mut || {
        transcript.public_message(b"bitz/virtual-statement/v1");
        transcript.public_message(&com.0);
        transcript.public_message(statement.params());
        transcript.public_message(&instance.map.digest());
        transcript.public_message(&claim);
    });

    let mut fold = None;
    time("fold", &mut || {
        fold = Some(
            instance
                .verifier
                .receive_fold(&claim, &mut transcript)
                .unwrap(),
        );
    });
    let fold = fold.unwrap();

    let mut leaf = None;
    time("GKR leaf", &mut || {
        leaf = Some(verifier::gkr_reduce(&mut transcript, &fold, instance.params.shape()).unwrap());
    });

    let mut query = None;
    time("transposition", &mut || {
        query = Some(statement.transpose_query(leaf.take().unwrap()).unwrap());
    });
    let query = query.unwrap();

    time("sumcheck (alone)", &mut || {
        let mut transcript = verifier_transcript(&scratch);
        sumcheck_verifier(&query, &mut transcript);
    });

    time("opening", &mut || {
        instance
            .pcs
            .verify_lin(&com, &query, StatementBinding::Bind, &mut transcript)
            .unwrap();
    });

    let mut transcript = Some(transcript);
    time("exhaustion", &mut || {
        transcript.take().unwrap().check_eof().unwrap();
    });

    (Run { prove, verify }, proof)
}

/// The post-GKR sumcheck for whichever form the transposition left.
fn sumcheck_prover(query: &OpeningQuery, packed: &[F128], transcript: &mut ProverState) {
    let OpeningQuery::InnerProduct { claim } = query else {
        unreachable!("the transposition leaves an inner-product claim");
    };
    post_gkr::prove(claim, packed, transcript).unwrap();
}

fn sumcheck_verifier(query: &OpeningQuery, transcript: &mut VerifierState<'_>) {
    let OpeningQuery::InnerProduct { claim } = query else {
        unreachable!("the transposition leaves an inner-product claim");
    };
    post_gkr::verify(claim, transcript).unwrap();
}

fn print_side(
    side: &str,
    runs: &[Run],
    steps: impl Fn(&Run) -> &Vec<(&'static str, Duration)>,
    total: impl Fn(&Run) -> Duration,
) {
    println!("  {side}: {}", fmt(median(runs.iter().map(&total))));
    for (index, (label, _)) in steps(&runs[0]).iter().enumerate() {
        let step = median(runs.iter().map(|run| steps(run)[index].1));
        println!("    {label:<18} {}", fmt(step));
    }
}

fn median(samples: impl Iterator<Item = Duration>) -> Duration {
    let mut samples: Vec<Duration> = samples.collect();
    samples.sort();
    samples[samples.len() / 2]
}

fn fmt(time: Duration) -> String {
    let micros = time.as_secs_f64() * 1e6;
    if micros < 1_000.0 {
        format!("{micros:8.1} us")
    } else if micros < 1_000_000.0 {
        format!("{:8.2} ms", micros / 1e3)
    } else {
        format!("{:8.3} s ", micros / 1e6)
    }
}

fn reps() -> usize {
    let reps = std::env::var("BITZ_BENCH_REPS")
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("BITZ_BENCH_REPS: `{value}` is not a count"))
        })
        .unwrap_or(3);
    assert!(reps > 0, "BITZ_BENCH_REPS must be positive");
    reps
}

fn shapes() -> Vec<usize> {
    match std::env::var("BITZ_BENCH_SHAPES") {
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
    }
}

fn profile() -> LigeritoProfile {
    match std::env::var("BITZ_LIG_PROFILE").as_deref() {
        Err(_) | Ok("fast") => LigeritoProfile::Fast,
        Ok("slim") => LigeritoProfile::Slim,
        Ok("secure") => LigeritoProfile::Secure,
        Ok(other) => panic!("BITZ_LIG_PROFILE: unknown profile `{other}` (fast | slim | secure)"),
    }
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
