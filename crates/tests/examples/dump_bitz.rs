//! Transcript-parity dump: one honest instance, proved and verified here,
//! written out for the F2Z reference implementation to re-prove.
//!
//! Usage: `dump_bitz <t> <s> <seed> <out-dir>`
use std::io::Write;

use common::Shape;
use crypto_primitives::LiftElement;
use tests::{Instance, Q, prover_transcript, verifier_transcript};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args[1].parse().unwrap();
    let s: usize = args[2].parse().unwrap();
    let seed: u64 = args[3].parse().unwrap();
    let out = std::path::PathBuf::from(&args[4]);
    std::fs::create_dir_all(&out).unwrap();

    let shape = Shape::new(t, s).expect("shape");
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

    let mut w = std::fs::File::create(out.join("witness.bin")).unwrap();
    for e in &instance.packed {
        w.write_all(&e.lo.to_le_bytes()).unwrap();
        w.write_all(&e.hi.to_le_bytes()).unwrap();
    }
    let mut c = std::fs::File::create(out.join("claim.bin")).unwrap();
    for weights in [instance.claim.row_weights(), instance.claim.column_weights()] {
        c.write_all(&(weights.len() as u64).to_le_bytes()).unwrap();
        for weight in weights {
            c.write_all(&weight.lift().to_le_bytes()).unwrap();
        }
    }
    c.write_all(&instance.claim.target().lift().to_le_bytes()).unwrap();
    std::fs::write(out.join("narg.bin"), &proof.narg_string).unwrap();
    std::fs::write(out.join("hints.bin"), &proof.hints).unwrap();
    let meta = format!(
        "t={t}\ns={s}\nm={}\nseed={seed}\nq={Q}\ngenerator={}\nroot={}\nsession=bitz-tests\ninstance=fold-round-trip\nnarg_len={}\nhints_len={}\n",
        shape.log_bits(),
        hex(&instance.params.generator().to_bytes()),
        hex(&instance.com.0),
        proof.narg_string.len(),
        proof.hints.len(),
    );
    std::fs::write(out.join("meta.txt"), &meta).unwrap();
    print!("{meta}");
}
