//! Parity probe (their side): commit a ChaCha8-seeded packed witness exactly
//! as `tests::Instance::honest` builds it, dump the packed bytes, print the root.
//!
//! Usage: `dump_commit [t [s [seed [out-file]]]]`
mod support;

use common::Shape;
use pcs::{HashKind, LigeritoProfile, Pcs};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use support::{hex, write_binary, write_witness};
use tests::packed_witness;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: dump_commit [t [s [seed [out-file]]]]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() > 4 {
        return Err(USAGE.into());
    }
    let t = args
        .first()
        .map(|v| v.parse())
        .transpose()
        .map_err(|_| USAGE)?
        .unwrap_or(7);
    let s = args
        .get(1)
        .map(|v| v.parse())
        .transpose()
        .map_err(|_| USAGE)?
        .unwrap_or(15);
    let seed = args
        .get(2)
        .map(|v| v.parse())
        .transpose()
        .map_err(|_| USAGE)?
        .unwrap_or(31);
    let out = args.get(3).map(String::as_str).unwrap_or("witness.bin");
    let shape = Shape::new(t, s).map_err(|error| format!("invalid shape: {error:?}"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let packed = packed_witness(shape, &mut rng);
    let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3)
        .map_err(|error| format!("PCS configuration failed: {error:?}"))?;
    let (root, _data) = pcs
        .commit(&packed)
        .map_err(|error| format!("commitment failed: {error:?}"))?;
    write_binary(out, |output| write_witness(output, &packed))?;
    let root = hex(&root.0);
    println!(
        "their side: t={t} s={s} m={} packed_len={} profile=Fast hash=blake3 root={root}",
        shape.log_bits(),
        packed.len()
    );
    Ok(())
}
