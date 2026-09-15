//! Transcript-parity check: verify a proof produced elsewhere (the F2Z
//! reference implementation) against the honest instance `dump_bitz` built.
//!
//! Usage: `verify_bitz <t> <s> <seed> <narg-file> <hints-file>`
use common::Shape;
use tests::{Instance, verifier_transcript};
use transcript::Proof;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args[1].parse().unwrap();
    let s: usize = args[2].parse().unwrap();
    let seed: u64 = args[3].parse().unwrap();
    let proof = Proof {
        narg_string: std::fs::read(&args[4]).expect("narg file"),
        hints: std::fs::read(&args[5]).expect("hints file"),
    };
    let instance = Instance::honest(Shape::new(t, s).expect("shape"), seed);
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
