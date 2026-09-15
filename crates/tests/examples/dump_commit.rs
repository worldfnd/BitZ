//! Parity probe (their side): commit a ChaCha8-seeded packed witness exactly
//! as `tests::Instance::honest` builds it, dump the packed bytes, print the root.
use common::Shape;
use field::F128;
use pcs::{HashKind, LigeritoProfile, Pcs};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(7);
    let s: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(15);
    let seed: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(31);
    let out = args.get(4).cloned().unwrap_or_else(|| "witness.bin".to_string());
    let shape = Shape::new(t, s).expect("admissible shape");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let packed: Vec<F128> = (0..(1usize << shape.log_bits()) / 128)
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).expect("pcs");
    let (root, _data) = pcs.commit(&packed).expect("commit");
    let mut f = std::fs::File::create(&out).expect("open out");
    for e in &packed {
        f.write_all(&e.lo.to_le_bytes()).unwrap();
        f.write_all(&e.hi.to_le_bytes()).unwrap();
    }
    let hex: String = root.0.iter().map(|b| format!("{b:02x}")).collect();
    println!(
        "their side: t={t} s={s} m={} packed_len={} profile=Fast(upstream) log_batch=6 hash=blake3 root={hex}",
        shape.log_bits(),
        packed.len()
    );
}
