//! Generate `w` and `M * w` while verifying a signed 2 KiB message.
//!
//! Run with `cargo bench -p circuit --bench ecdsa_sha256_witgen`.

mod support;

use circuit::ecdsa_sha256::{VERIFY_2KB_WITNESS_BITS, verify_2kb_message_circuit};
use circuit::p256::prepare;
use circuit::witgen::Witgen;
use divan::{Bencher, black_box};

fn main() {
    divan::main();
}

#[divan::bench]
fn ecdsa_sha256_2kb_witgen(bencher: Bencher) {
    prepare();
    let inputs = support::ecdsa_sha256::valid_input();
    bencher.bench_local(|| {
        let inputs = black_box(inputs.as_ref());
        let mut witgen = Witgen::with_inputs_and_capacity(inputs, VERIFY_2KB_WITNESS_BITS);
        verify_2kb_message_circuit(&mut witgen, inputs);
        black_box(witgen.into_witnesses())
    });
}
