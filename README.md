# BitZ-benchmark

An implementation of **BitZ** — an integer-MLE-evaluation polynomial commitment
scheme over an `F_2` commitment, folded in the exponent of a binary field and
opened through a ring-switch + recursive Ligerito pipeline.

## Building

```sh
cargo test --workspace
cargo clippy --workspace --all-targets
```

## Ordinary Spartan transcript parity

The independent F2Z compatibility path and reproducible comparison commands are
documented in [Spartan transcript parity](docs/spartan-transcript-parity.md).
