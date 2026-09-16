# BitZ

**BitZ** is a hash-based polynomial commitment scheme (PCS) for committing to polynomials with coefficients in a ring S (e.g a Finite Field F, integers Z) and proves evaluation claims over another arbitrary ring R. This repo provides a modular implementation of a concrete instantiation of this protocol where S is the integers Z and R is a Finite Field Fq.

# BitZ-SNARK

This repo also implements a Spartan like PIOP using BitZ as its PCS which is then used to implement among others end to end SHA-256 proving and verifying.

## Build

The workspace uses Rust **1.97.1**, pinned in `rust-toolchain.toml`.
Run these commands from the repository root:

```sh
cargo build --release --workspace
cargo test --release --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

## Prove and verify

Prove eight SHA-256 compression steps and verify the proof:

```sh
cargo run --release -p bitz-cli -- circuit-e2e \
  --circuit sha256-chain --num-blocks 8 --threads 1
```

The command generates random inputs and reports timings for each proof stage.
All input and output bits are public.
Select the workload with `--circuit`:

| Circuit                | Workload                                                                  |
| ---------------------- | ------------------------------------------------------------------------- |
| `sha256-compression`   | One raw block without padding.                                            |
| `sha256-chain`         | One or more raw blocks without padding.                                   |
| `sha256-block-aligned` | A block-aligned message with SHA-256 padding; empty messages are allowed. |
| `sha256-2kb`           | A 2 KiB message with SHA-256 padding.                                     |

Use `--num-blocks` to set the chain or block-aligned message length in 64-byte blocks.
Use `--threads` to set the number of worker threads.

## Benchmarks

Run all SHA-256 circuit benchmarks with one Rayon worker:

```sh
RAYON_NUM_THREADS=1 cargo bench -p bitz-cli --bench circuits
```

The suite measures all four workloads through the complete proof process. It also measures setup, witness generation, commitment, proving, and verification separately. Chain and block-aligned benchmarks use one input block.

Set `RAYON_NUM_THREADS` to change the number of worker threads.

## Layout

The workspace contains library crates under `crates/` and the `bitz-cli` package under `tooling/cli/`.

| Path                | Package      | Role                                                                               |
| ------------------- | ------------ | ---------------------------------------------------------------------------------- |
| `crates/circuit`    | `circuit`    | SHA-256 and ECDSA circuits, witness generation, and matrix operations.             |
| `crates/spartan`    | `spartan`    | Outer and inner sumchecks that reduce circuit constraints to evaluation claims.    |
| `crates/prover`     | `prover`     | BitZ proving, including direct and virtual witness openings.                       |
| `crates/verifier`   | `verifier`   | BitZ verification, including direct and virtual witness openings.                  |
| `crates/pcs`        | `pcs`        | Binary commitments, ring switching, and recursive Ligerito openings through Flock. |
| `crates/gkr`        | `gkr`        | Grand-product reductions over the binary field.                                    |
| `crates/common`     | `common`     | Protocol parameters, shapes, claims, and virtual witness maps.                     |
| `crates/field`      | `field`      | Binary-field and prime-field arithmetic.                                           |
| `crates/poly`       | `poly`       | Multilinear polynomials, equality polynomials, and evaluation operations.          |
| `crates/transcript` | `transcript` | Fiat-Shamir transcripts, challenges, and proof bytes.                              |
| `crates/host`       | `host`       | Proof serialization and parsing.                                                   |
| `crates/tests`      | `tests`      | Integration tests and proof comparison examples.                                   |
| `tooling/cli`       | `bitz-cli`   | SHA-256 proving CLI, circuit adapters, proof library, and circuit benchmarks.      |

## Related work

[todo]: related work
