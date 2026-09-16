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

[todo]: end to end prove command and benchmarks

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
