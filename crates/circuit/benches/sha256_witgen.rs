//! Generate `w` and `M * w` for block-aligned SHA-256 circuits.
//!
//! Run with `cargo bench -p circuit --bench sha256_witgen`.

mod support;

use circuit::sha256::{
    SHA256_2KB_WITNESS_BITS, block_aligned_witness_bits, sha256_2kb_circuit,
    sha256_block_aligned_circuit,
};
use circuit::witgen::Witgen;
use divan::{Bencher, black_box};

fn main() {
    divan::main();
}

#[divan::bench]
fn sha256_2kb_witgen(bencher: Bencher) {
    let message = support::sha256::message_2kb();
    bencher.bench_local(|| {
        let mut witgen =
            Witgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
        let digest = sha256_2kb_circuit(&mut witgen, black_box(message.as_ref()));
        black_box((digest, witgen.into_witnesses()))
    });
}

const SHA256_1_MIB_BYTES: usize = 1024 * 1024;
const SHA256_1_MIB_BITS: usize = SHA256_1_MIB_BYTES * 8;
const SHA256_1_MIB_WITNESS_BITS: usize = block_aligned_witness_bits(SHA256_1_MIB_BITS);

#[divan::bench(sample_count = 10, sample_size = 1)]
fn sha256_1_mib_witgen(bencher: Bencher) {
    let message = vec![0_u64; SHA256_1_MIB_BITS / 64];
    bencher.bench_local(|| {
        let message = black_box(&message);
        let mut witgen = Witgen::with_packed_inputs_and_capacity(
            message,
            SHA256_1_MIB_BITS,
            SHA256_1_MIB_WITNESS_BITS,
        );
        let digest = sha256_block_aligned_circuit(&mut witgen, SHA256_1_MIB_BITS, |index| {
            message[index / 64] >> (index % 64) & 1 == 1
        });
        assert_eq!(witgen.witness().bit_len(), SHA256_1_MIB_WITNESS_BITS);
        black_box((digest, witgen.into_witnesses()))
    });
}
