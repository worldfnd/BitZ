//! Generate `w` and `M * w` for the standalone prehashed P-256 verifier.
//!
//! Run with `cargo bench -p circuit --bench p256_witgen`.

mod support;

use circuit::p256::{VERIFY_DIGEST_WITNESS_BITS, prepare, verify_digest_circuit};
use circuit::witgen::Witgen;
use divan::{Bencher, black_box};

fn main() {
    divan::main();
}

#[divan::bench]
fn p256_witgen(bencher: Bencher) {
    prepare();
    let inputs = support::p256::valid_input();
    bencher.bench_local(|| {
        let inputs = black_box(inputs.as_ref());
        let mut witgen = Witgen::with_inputs_and_capacity(inputs, VERIFY_DIGEST_WITNESS_BITS);
        verify_digest_circuit(&mut witgen, inputs);
        black_box(witgen.into_witnesses())
    });
}
