//! Witgen, `M`, `ABC(Mw)`, and `rM` for SHA-256 circuits.
//!
//! Run with `cargo bench -p circuit --bench sha256_matrix_products`.

mod support;

use circuit::matrix_products::RuntimeModulus;
use circuit::matrix_transpose::{MTransposeGenerator, MaterializedMTranspose};
use circuit::sha256::{
    SHA256_2KB_MESSAGE_BITS, SHA256_2KB_WITNESS_BITS, block_aligned_witness_bits,
    sha256_2kb_circuit, sha256_block_aligned_circuit,
};
use circuit::witgen::ProductWitgen;
use divan::{Bencher, black_box};
use field::F128;
use num_bigint::BigUint;
use num_traits::One;

fn main() {
    divan::main();
}

fn build_transpose() -> MaterializedMTranspose {
    let mut generator = MTransposeGenerator::new(SHA256_2KB_MESSAGE_BITS);
    let inputs = generator.take_boxed_inputs();
    let _ = sha256_2kb_circuit(&mut generator, &inputs);
    generator.finish()
}

fn prime_128() -> BigUint {
    (BigUint::one() << 128_usize) - BigUint::from(159_u64)
}

fn prepare_parallel_reduction() {
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});
}

/// Generate `w`, `M * w`, and exact integer `A/B/C(M * w)`.
#[divan::bench]
fn sha256_2kb_witgen(bencher: Bencher) {
    let message = support::sha256::message_2kb();
    bencher.bench_local(|| {
        let mut witgen =
            ProductWitgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
        let digest = sha256_2kb_circuit(&mut witgen, black_box(message.as_ref()));
        black_box((digest, witgen.into_parts()))
    });
}

const SHA256_1_MIB_BITS: usize = 1024 * 1024 * 8;
const SHA256_1_MIB_WITNESS_BITS: usize = block_aligned_witness_bits(SHA256_1_MIB_BITS);

#[divan::bench(sample_count = 10, sample_size = 1)]
fn sha256_1_mib_witgen(bencher: Bencher) {
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
        assert_eq!(witgen.witness().bit_len(), SHA256_1_MIB_WITNESS_BITS);
        black_box((digest, witgen.into_parts()))
    });
}

/// Generate the sparse representation of `M` used to compute `rM`.
#[divan::bench]
fn sha256_2kb_m(bencher: Bencher) {
    bencher.bench_local(|| {
        let mut generator = MTransposeGenerator::new(SHA256_2KB_MESSAGE_BITS);
        let inputs = generator.take_boxed_inputs();
        let digest = sha256_2kb_circuit(&mut generator, &inputs);
        black_box((digest, generator.finish()))
    });
}

/// Reduce precomputed exact integer `A(M*w)`, `B(M*w)`, and `C(M*w)` values.
#[divan::bench]
fn sha256_2kb_abc_mw(bencher: Bencher) {
    prepare_parallel_reduction();
    let message = support::sha256::message_2kb();
    let mut witgen =
        ProductWitgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
    let _ = sha256_2kb_circuit(&mut witgen, &message);
    let (_, _, integer_products) = witgen.into_parts();
    let modulus = RuntimeModulus::<2>::new(prime_128()).unwrap();
    bencher.bench_local(|| {
        black_box(black_box(&integer_products).reduce_parallel(black_box(&modulus)))
    });
}

#[divan::bench]
fn sha256_2kb_rm(bencher: Bencher) {
    prepare_parallel_reduction();
    let transpose = build_transpose();
    let challenges: Vec<_> = (0..transpose.row_count())
        .map(|index| {
            F128::new(
                (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
            )
        })
        .collect();
    bencher.bench_local(|| black_box(transpose.apply(black_box(&challenges)).unwrap()));
}
