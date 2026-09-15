//! Transcript-parity check: verify a proof produced elsewhere (the F2Z
//! reference implementation) against the honest instance `dump_bitz` built.
//!
//! Usage: `verify_bitz <log-bits> <seed> <narg-file> <hints-file>` (the reference split)
use common::Shape;
use tests::{Instance, verifier_transcript};
use transcript::Proof;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_bits: usize = args[1].parse().unwrap();
    let seed: u64 = args[2].parse().unwrap();
    let proof = Proof {
        narg_string: std::fs::read(&args[3]).expect("narg file"),
        hints: std::fs::read(&args[4]).expect("hints file"),
    };
    let instance = Instance::honest(Shape::for_log_bits(log_bits).expect("shape"), seed);
    let started = std::time::Instant::now();
    let result = instance.verifier.verify(
        &instance.claim,
        &instance.pcs,
        instance.com,
        verifier_transcript(&proof),
    );
    println!("their verifier on the given proof: {result:?} ({:.1?})", started.elapsed());
    if result.is_err() {
        std::process::exit(1);
    }
}
