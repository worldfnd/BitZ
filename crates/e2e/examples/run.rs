//! Proves SHA-256 end to end and reports sizes and phase timings.
//!
//! Usage: `cargo run --release -p e2e --example run -- [blocks] [reps]`
use std::time::Instant;

use e2e::{Sha256Statement, shape_for, size_of};

fn main() {
    let mut args = std::env::args().skip(1);
    let blocks: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1);
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);

    let message_bits = blocks * 512;
    let size = size_of(message_bits);
    let committed = shape_for(size.witness_bits).unwrap();
    let claim = shape_for(size.assignment_bits).unwrap();

    println!("blocks            {blocks}");
    println!("message bits      {message_bits}");
    println!(
        "committed f       {} bits -> shape 2^{} = 2^{} x 2^{}  ({:.1}% full)",
        size.witness_bits,
        committed.log_bits(),
        committed.log_rows(),
        committed.log_columns(),
        100.0 * size.witness_bits as f64 / (1u64 << committed.log_bits()) as f64,
    );
    println!(
        "integer h         {} bits -> shape 2^{} = 2^{} x 2^{}  ({:.1}% full)",
        size.assignment_bits,
        claim.log_bits(),
        claim.log_rows(),
        claim.log_columns(),
        100.0 * size.assignment_bits as f64 / (1u64 << claim.log_bits()) as f64,
    );
    println!("r1cs rows         {}", size.r1cs_rows);
    println!();

    let message: Vec<bool> = (0..message_bits)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) & 1 == 1)
        .collect();

    // Interleave the reps: this machine drifts enough between rounds that
    // consecutive timings of one phase are not comparable.
    let mut build = Vec::new();
    let mut prove = Vec::new();
    let mut verify = Vec::new();
    let mut proof_bytes = (0usize, 0usize);
    let profiling = std::env::var_os("F2Z_PROFILE").is_some();
    for rep in 0..reps {
        let start = Instant::now();
        let statement = Sha256Statement::build(&message).unwrap();
        build.push(start.elapsed().as_secs_f64());

        let start = Instant::now();
        let (proof, piop) = statement.prove().unwrap();
        prove.push(start.elapsed().as_secs_f64());
        proof_bytes = (proof.narg_string.len(), proof.hints.len());

        let start = Instant::now();
        statement.verify(&proof, &piop).unwrap();
        verify.push(start.elapsed().as_secs_f64());

        // The table is per rep, so a slow first pass does not smear into the
        // rest. Only the last one is printed.
        if profiling && rep + 1 < reps {
            let _ = prof::take();
        }
    }

    report("build ", &mut build);
    report("prove ", &mut prove);
    report("verify", &mut verify);
    println!(
        "\nproof             {} narg bytes + {} hint bytes",
        proof_bytes.0, proof_bytes.1
    );
    dump_or_write_json(
        "regions (last rep)",
        proof_bytes,
        (size.witness_bits, size.assignment_bits, size.r1cs_rows),
    );
}

/// Writes the region tree as JSON when `F2Z_PROFILE_JSON` names a path, and
/// prints the table otherwise.
fn dump_or_write_json(title: &str, proof: (usize, usize), sizes: (usize, usize, usize)) {
    let Some(path) = std::env::var_os("F2Z_PROFILE_JSON") else {
        prof::dump(title);
        return;
    };
    let regions = prof::take();
    let mut out = String::from("{\n");
    out.push_str(&format!(
        "  \"narg_bytes\": {}, \"hint_bytes\": {},\n",
        proof.0, proof.1
    ));
    out.push_str(&format!(
        "  \"witness_bits\": {}, \"assignment_bits\": {}, \"r1cs_rows\": {},\n",
        sizes.0, sizes.1, sizes.2
    ));
    out.push_str("  \"regions\": [\n");
    for (index, region) in regions.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"i\": {index}, \"label\": \"{}\", \"depth\": {}, \"parent\": {}, \"inclusive\": {:.6}, \"exclusive\": {:.6}, \"calls\": {}}}{}\n",
            region.label,
            region.depth,
            region.parent.map_or("null".to_string(), |p| p.to_string()),
            region.inclusive.as_secs_f64(),
            region.exclusive.as_secs_f64(),
            region.calls,
            if index + 1 == regions.len() { "" } else { "," },
        ));
    }
    out.push_str("  ]\n}\n");
    std::fs::write(&path, out).expect("profile json");
    eprintln!("wrote {}", path.to_string_lossy());
}

fn report(label: &str, samples: &mut [f64]) {
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    println!(
        "{label}            median {:>8.3} s   min {:>8.3} s   max {:>8.3} s",
        median,
        samples[0],
        samples[samples.len() - 1],
    );
}
