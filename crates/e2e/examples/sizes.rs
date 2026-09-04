//! Circuit sizes per block count, to pick shapes against.
use circuit::constraints::ConstraintGenerator;
use circuit::sha256::{block_aligned_witness_bits, sha256_block_aligned_circuit};

fn main() {
    println!(
        "{:>7} {:>12} {:>12} {:>10} {:>8}",
        "blocks", "f", "h", "r1cs", "h/f"
    );
    for blocks in [1usize, 2, 4, 8, 16, 32] {
        let message_bits = blocks * 512;
        let mut generator = ConstraintGenerator::new(message_bits);
        let inputs: Vec<_> = (0..message_bits).map(|i| generator.input(i)).collect();
        let _ = sha256_block_aligned_circuit(&mut generator, message_bits, |b| inputs[b].clone());
        let m = generator.into_matrices();
        let f = m.m.column_count() - 1;
        let h = m.m.row_count();
        println!(
            "{blocks:>7} {f:>12} {h:>12} {:>10} {:>8.2}   (predicted f={})",
            m.a.row_count(),
            h as f64 / f as f64,
            block_aligned_witness_bits(message_bits),
        );
    }
}
