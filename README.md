# BitZ-benchmark

An implementation of **BitZ** — an integer-MLE-evaluation polynomial commitment
scheme over an `F_2` commitment, folded in the exponent of a binary field and
opened through a ring-switch + recursive Ligerito pipeline.

## Building

```sh
cargo test --workspace
cargo clippy --workspace --all-targets
```

## Benchmarking

Independent SHA-256 compressions end to end, per batch size, with the PIOP
mocked and the grand product real:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p tests --bench sha256
```

The same pipeline one step at a time, prover and verifier, with medians per
step:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p tests --bench sha256_steps
```

Each bench's header documents its knobs. Micro-benchmarks live next to their
crates: `cargo bench -p field`, `-p poly`, `-p circuit`, and `-p post_gkr`
for the post-GKR sumcheck on its own.
