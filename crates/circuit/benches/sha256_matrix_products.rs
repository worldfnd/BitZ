//! End-to-end exact recording and batch reduction for SHA-256.
//!
//! Run with `cargo bench -p circuit --bench sha256_matrix_products`.

use circuit::matrix_products::RuntimeModulus;
use circuit::sha256::{
    SHA256_2KB_MESSAGE_BITS, SHA256_2KB_WITNESS_BITS, block_aligned_witness_bits,
    sha256_2kb_circuit, sha256_block_aligned_circuit,
};
use circuit::witgen::ProductWitgen;
use divan::{Bencher, black_box};
use num_bigint::BigUint;
use num_traits::One;

fn main() {
    divan::main();
}

fn message_2kb() -> Box<[bool; SHA256_2KB_MESSAGE_BITS]> {
    let bits: Box<[bool]> = (0..SHA256_2KB_MESSAGE_BITS)
        .map(|bit| {
            let byte = (bit / 8) as u8;
            byte & (1 << (7 - bit % 8)) != 0
        })
        .collect();
    bits.try_into()
        .unwrap_or_else(|_| unreachable!("message length is fixed"))
}

const SHA256_1_MIB_BITS: usize = 1024 * 1024 * 8;
const SHA256_1_MIB_WITNESS_BITS: usize = block_aligned_witness_bits(SHA256_1_MIB_BITS);

fn prime_128() -> BigUint {
    (BigUint::one() << 128_usize) - BigUint::from(159_u64)
}

fn prepare_parallel_reduction() {
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});
}

fn run_2kb_recording_witgen_bench(bencher: Bencher) {
    let message = message_2kb();
    bencher.bench_local(|| {
        let message = black_box(message.as_ref());
        let mut witgen = ProductWitgen::with_inputs_and_capacity(message, SHA256_2KB_WITNESS_BITS);
        let digest = sha256_2kb_circuit(&mut witgen, message);
        black_box((digest, witgen.into_parts()))
    });
}

fn run_1_mib_recording_witgen_bench(bencher: Bencher) {
    let message = vec![0_u64; SHA256_1_MIB_BITS / 64];
    bencher.bench_local(|| {
        let message = black_box(&message);
        let mut witgen = ProductWitgen::with_packed_inputs_and_capacity(
            message,
            SHA256_1_MIB_BITS,
            SHA256_1_MIB_WITNESS_BITS,
        );
        let digest = sha256_block_aligned_circuit(&mut witgen, SHA256_1_MIB_BITS, |index| {
            message[index / 64] >> (index % 64) & 1 == 1
        });
        black_box((digest, witgen.into_parts()))
    });
}

fn run_2kb_pipeline_bench<const P: usize>(bencher: Bencher, prime: BigUint) {
    let message = message_2kb();
    let modulus = RuntimeModulus::<P>::new(black_box(prime)).unwrap();
    prepare_parallel_reduction();

    bencher.bench_local(|| {
        let message = black_box(message.as_ref());
        let mut witgen = ProductWitgen::with_inputs_and_capacity(message, SHA256_2KB_WITNESS_BITS);
        let digest = sha256_2kb_circuit(&mut witgen, message);
        let (witness, integer_witness, integer_products) = witgen.into_parts();
        let products = integer_products.reduce_parallel(&modulus);
        black_box((digest, witness, integer_witness, integer_products, products))
    });
}

fn run_1_mib_pipeline_bench<const P: usize>(bencher: Bencher, prime: BigUint) {
    let message = vec![0_u64; SHA256_1_MIB_BITS / 64];
    let modulus = RuntimeModulus::<P>::new(black_box(prime)).unwrap();
    prepare_parallel_reduction();

    bencher.bench_local(|| {
        let message = black_box(&message);
        let mut witgen = ProductWitgen::with_packed_inputs_and_capacity(
            message,
            SHA256_1_MIB_BITS,
            SHA256_1_MIB_WITNESS_BITS,
        );
        let digest = sha256_block_aligned_circuit(&mut witgen, SHA256_1_MIB_BITS, |index| {
            message[index / 64] >> (index % 64) & 1 == 1
        });
        let (witness, integer_witness, integer_products) = witgen.into_parts();
        let products = integer_products.reduce_parallel(&modulus);
        black_box((digest, witness, integer_witness, integer_products, products))
    });
}

#[divan::bench]
fn sha256_2kb_recording_witgen(bencher: Bencher) {
    run_2kb_recording_witgen_bench(bencher);
}

#[divan::bench(sample_count = 10, sample_size = 1)]
fn sha256_1_mib_recording_witgen(bencher: Bencher) {
    run_1_mib_recording_witgen_bench(bencher);
}

#[divan::bench]
fn sha256_2kb_record_reduce_128(bencher: Bencher) {
    run_2kb_pipeline_bench::<2>(bencher, prime_128());
}

#[divan::bench(sample_count = 10, sample_size = 1)]
fn sha256_1_mib_record_reduce_128(bencher: Bencher) {
    run_1_mib_pipeline_bench::<2>(bencher, prime_128());
}
