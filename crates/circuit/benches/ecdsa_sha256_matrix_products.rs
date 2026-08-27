//! Witgen, `M`, `ABC(Mw)`, and `rM` for a signed 2 KiB message.
//!
//! Run with `cargo bench -p circuit --bench ecdsa_sha256_matrix_products`.

mod support;

use circuit::ecdsa_sha256::{
    VERIFY_2KB_INPUT_BITS, VERIFY_2KB_WITNESS_BITS, verify_2kb_message_circuit,
};
use circuit::matrix_products::RuntimeModulus;
use circuit::matrix_transpose::{MTransposeGenerator, MaterializedMTranspose};
use circuit::p256::prepare;
use circuit::witgen::ProductWitgen;
use divan::{Bencher, black_box};
use field::F128;
use num_bigint::BigUint;
use num_traits::One;

fn main() {
    divan::main();
}

fn build_transpose() -> MaterializedMTranspose {
    let mut generator = MTransposeGenerator::new(VERIFY_2KB_INPUT_BITS);
    let inputs = generator.take_boxed_inputs();
    verify_2kb_message_circuit(&mut generator, &inputs);
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
fn ecdsa_sha256_2kb_witgen(bencher: Bencher) {
    prepare();
    let inputs = support::ecdsa_sha256::valid_input();
    bencher.bench_local(|| {
        let inputs = black_box(inputs.as_ref());
        let mut witgen = ProductWitgen::with_inputs_and_capacity(inputs, VERIFY_2KB_WITNESS_BITS);
        verify_2kb_message_circuit(&mut witgen, inputs);
        black_box(witgen.into_parts())
    });
}

/// Generate the sparse representation of `M` used to compute `rM`.
#[divan::bench]
fn ecdsa_sha256_2kb_m(bencher: Bencher) {
    prepare();
    bencher.bench_local(|| {
        let mut generator = MTransposeGenerator::new(VERIFY_2KB_INPUT_BITS);
        let inputs = generator.take_boxed_inputs();
        verify_2kb_message_circuit(&mut generator, &inputs);
        black_box(generator.finish())
    });
}

/// Reduce precomputed exact integer `A(M*w)`, `B(M*w)`, and `C(M*w)` values.
#[divan::bench]
fn ecdsa_sha256_2kb_abc_mw(bencher: Bencher) {
    prepare();
    prepare_parallel_reduction();
    let inputs = support::ecdsa_sha256::valid_input();
    let mut witgen =
        ProductWitgen::with_inputs_and_capacity(inputs.as_ref(), VERIFY_2KB_WITNESS_BITS);
    verify_2kb_message_circuit(&mut witgen, &inputs);
    let (_, _, integer_products) = witgen.into_parts();
    let modulus = RuntimeModulus::<2>::new(prime_128()).unwrap();
    bencher.bench_local(|| {
        black_box(black_box(&integer_products).reduce_parallel(black_box(&modulus)))
    });
}

#[divan::bench]
fn ecdsa_sha256_2kb_rm(bencher: Bencher) {
    prepare();
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
