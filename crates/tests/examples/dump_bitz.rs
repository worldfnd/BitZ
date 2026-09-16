//! Transcript-parity dump: one honest instance, proved and verified here,
//! written out for the F2Z reference implementation to re-prove.
//!
//! Usage: `dump_bitz <log-bits> <seed> <out-dir>` (the reference split)
mod support;

use std::io::Write;

use common::Shape;
use crypto_primitives::LiftElement;
use support::{hex, write_binary, write_witness};
use tests::{Instance, Q, prover_transcript, verifier_transcript};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: dump_bitz <log-bits> <seed> <out-dir>";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [log_bits, seed, out] = args.as_slice() else {
        return Err(USAGE.into());
    };
    let log_bits = log_bits.parse().map_err(|_| USAGE)?;
    let seed = seed.parse().map_err(|_| USAGE)?;
    let out = std::path::Path::new(out);

    let shape =
        Shape::for_log_bits(log_bits).map_err(|error| format!("invalid shape: {error:?}"))?;
    std::fs::create_dir_all(out)?;
    let (t, s) = (shape.log_rows(), shape.log_columns());
    let instance = Instance::honest(shape, seed);

    let started = std::time::Instant::now();
    let mut transcript = prover_transcript();
    instance
        .prover
        .prove(
            &instance.claim,
            &instance.pcs,
            &instance.data,
            instance.packed.clone(),
            &mut transcript,
        )
        .expect("honest instance proves");
    let proof = transcript.finish();
    println!("prove: {:.1?}", started.elapsed());
    let started = std::time::Instant::now();
    instance
        .verifier
        .verify(
            &instance.claim,
            &instance.pcs,
            instance.com,
            verifier_transcript(&proof),
        )
        .expect("honest proof verifies");
    println!("verify: {:.1?}", started.elapsed());

    write_binary(out.join("witness.bin"), |output| {
        write_witness(output, &instance.packed)
    })?;
    write_binary(out.join("claim.bin"), |output| {
        for weights in [
            instance.claim.row_weights(),
            instance.claim.column_weights(),
        ] {
            output.write_all(&(weights.len() as u64).to_le_bytes())?;
            for weight in weights {
                output.write_all(&weight.lift().to_le_bytes())?;
            }
        }
        output.write_all(&instance.claim.target().lift().to_le_bytes())
    })?;
    std::fs::write(out.join("narg.bin"), &proof.narg_string)?;
    std::fs::write(out.join("hints.bin"), &proof.hints)?;
    let meta = format!(
        "t={t}\ns={s}\nm={}\nseed={seed}\nq={Q}\ngenerator={}\nroot={}\nsession=bitz-tests\ninstance=fold-round-trip\nnarg_len={}\nhints_len={}\n",
        shape.log_bits(),
        hex(&instance.params.generator().to_bytes()),
        hex(&instance.com.0),
        proof.narg_string.len(),
        proof.hints.len(),
    );
    std::fs::write(out.join("meta.txt"), &meta)?;
    print!("{meta}");
    Ok(())
}
