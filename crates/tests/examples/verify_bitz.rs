//! Transcript-parity check: verify a proof produced elsewhere (the F2Z
//! reference implementation) against the honest instance `dump_bitz` built.
//!
//! Usage: `verify_bitz <log-bits> <seed> <narg-file> <hints-file>` (the reference split)
use common::Shape;
use tests::{Instance, verifier_transcript};
use transcript::Proof;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: verify_bitz <log-bits> <seed> <narg-file> <hints-file>";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [log_bits, seed, narg_file, hints_file] = args.as_slice() else {
        return Err(USAGE.into());
    };
    let log_bits = log_bits.parse().map_err(|_| USAGE)?;
    let seed = seed.parse().map_err(|_| USAGE)?;
    let shape =
        Shape::for_log_bits(log_bits).map_err(|error| format!("invalid shape: {error:?}"))?;
    let proof = Proof {
        narg_string: std::fs::read(narg_file)?,
        hints: std::fs::read(hints_file)?,
    };
    let instance = Instance::honest(shape, seed);
    let started = std::time::Instant::now();
    let result = instance.verifier.verify(
        &instance.claim,
        &instance.pcs,
        instance.com,
        verifier_transcript(&proof),
    );
    println!(
        "their verifier on the given proof: {result:?} ({:.1?})",
        started.elapsed()
    );
    if result.is_err() {
        std::process::exit(1);
    }
    Ok(())
}
