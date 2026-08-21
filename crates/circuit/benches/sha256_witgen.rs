//! Witness-generation time for block-aligned SHA-256 circuits.
//!
//! Run with `cargo bench -p circuit --bench sha256_witgen`.

use circuit::sha256::{
    SHA256_2KB_MESSAGE_BITS, SHA256_2KB_WITNESS_BITS, block_aligned_witness_bits,
    sha256_2kb_circuit, sha256_block_aligned_circuit,
};
use circuit::witgen::Witgen;
use divan::{Bencher, black_box};

fn main() {
    divan::main();
}

fn message_bits() -> [bool; SHA256_2KB_MESSAGE_BITS] {
    std::array::from_fn(|bit| {
        let byte = (bit / 8) as u8;
        byte & (1 << (7 - bit % 8)) != 0
    })
}

#[divan::bench]
fn sha256_2kb_witgen(bencher: Bencher) {
    let message = message_bits();
    bencher.bench_local(|| {
        let mut witgen = Witgen::with_inputs_and_capacity(&message, SHA256_2KB_WITNESS_BITS);
        let digest = sha256_2kb_circuit(&mut witgen, black_box(&message));
        black_box((digest, witgen.into_witness()))
    });
}

const SHA256_30_MIB_BYTES: usize = 30 * 1024 * 1024;
const SHA256_30_MIB_BITS: usize = SHA256_30_MIB_BYTES * 8;
const SHA256_30_MIB_WITNESS_BITS: usize = block_aligned_witness_bits(SHA256_30_MIB_BITS);

#[divan::bench(sample_count = 10, sample_size = 1)]
fn sha256_30_mib_witgen(bencher: Bencher) {
    let message = vec![0_u64; SHA256_30_MIB_BITS / 64];
    bencher.bench_local(|| {
        let message = black_box(&message);
        let mut witgen = Witgen::with_packed_inputs_and_capacity(
            message,
            SHA256_30_MIB_BITS,
            SHA256_30_MIB_WITNESS_BITS,
        );
        let digest = sha256_block_aligned_circuit(&mut witgen, SHA256_30_MIB_BITS, |index| {
            message[index / 64] >> (index % 64) & 1 == 1
        });
        assert_eq!(witgen.witness().bit_len(), SHA256_30_MIB_WITNESS_BITS);
        black_box((digest, witgen.into_witness()))
    });
}
