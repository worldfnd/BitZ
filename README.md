# BitZ

**BitZ** is a hash-based polynomial commitment scheme (PCS) for commiting to polynomials with coeffients in a ring S (e.g a finite field F, integers Z) and proves evaluation claims over another arbitrary ring R. This repo provides an implementation of a concrete instantiation of this protocol where S is the integers Z and R is a Finite Field Fq.

The repository also implements a **Spartan polynomial interactive oracle proof
(PIOP)** for end-to-end circuit proving. Spartan reduces circuit constraints to
a witness evaluation claim. BitZ proves that claim against the committed
witness. **SHA-256** is the current workload for end-to-end proving and
benchmarking.

BitZ folds integer values, then uses GKR over `GF(2^128)`.
Ring-switching and recursive Ligerito complete the opening.
The binary commitment backend uses
[Flock](https://github.com/succinctlabs/flock).
Virtualization lets circuits derive a larger witness `h = M(1 || f)` over `F_2`
from committed bits `f`.

This is a research implementation. Current proofs do not provide zero knowledge.

## Build

The workspace uses Rust **1.97.1**, pinned in `rust-toolchain.toml`.
Run these commands from the repository root:

```sh
cargo build --release --workspace
cargo test --release --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

## SHA-256 end-to-end proving

The SHA-256 CLI and circuit benchmarks currently require `feat/all-circuit-benches`
([PR #68](https://github.com/worldfnd/BitZ/pull/68), built on
[PR #64](https://github.com/worldfnd/BitZ/pull/64)).
These tools are not yet on `main`.

Select that branch before running the commands below:

```sh
git fetch origin feat/all-circuit-benches
git switch feat/all-circuit-benches
```

Generate random inputs, prove eight SHA-256 compression steps, and verify the proof:

```sh
cargo run --release -p bitz-cli -- circuit-e2e \
  --circuit sha256-chain --num-blocks 8 --threads 1
```

The command runs circuit setup, witness generation, commitment, proving, and
verification. The verifier uses the public statement and proof without the
witness. The SHA-256 adapters expose input bits and outputs as public values.

`--circuit` selects one of four adapters:

| Adapter                | Workload                                                                          |
| ---------------------- | --------------------------------------------------------------------------------- |
| `sha256-compression`   | One raw compression block, with an optional initial state.                        |
| `sha256-chain`         | A positive number of raw blocks from the standard initial state, without padding. |
| `sha256-block-aligned` | A block-aligned message, including an empty message, with SHA-256 padding.        |
| `sha256-2kb`           | A 2 KiB message with SHA-256 padding.                                             |

Variable-length adapters default to one input block. The 2 KiB adapter uses
32 input blocks. Use `--num-blocks` to set the length of a chain or block-aligned
message.

The CLI reports the opening path, circuit dimensions, and timings for each
stage. `total_prove_ms` includes commitment and proving. It excludes setup,
witness generation, and verification. Input generation and thread-pool
initialization occur outside the reported timings.

## Benchmarks

Run the SHA-256 circuit benchmarks with one Rayon worker:

```sh
RAYON_NUM_THREADS=1 cargo bench -p bitz-cli --bench circuits
```

The suite measures all four adapters across six stages: end-to-end, setup,
witness generation, commitment, proving, and verification.
The end-to-end measurement includes all five individual stages.
Chain and block-aligned benchmark cases use one input block.
Use the CLI to measure other lengths.

Run every benchmark once to check the complete suite:

```sh
RAYON_NUM_THREADS=1 cargo bench -p bitz-cli --bench circuits -- --test
```

Input generation occurs outside measurements. End-to-end and setup benchmarks
use fresh instances for each sample. Other stages reuse a prepared instance
for each benchmark case.

Component benchmarks are also available on `main`:

```sh
# SHA-256 witness generation and matrix operations.
cargo bench -p circuit --bench sha256_matrix_products

# Spartan proving and verification for SHA-256 constraints.
cargo bench -p spartan --bench sha256

# BitZ's GKR reduction.
cargo bench -p prover --bench prover
```

Component timings measure their named stages. They do not measure a complete
SHA-256 proof.

Run benchmarks separately. Record the commit, CPU, operating system, Rust
version, thread count, workload, and proof configuration with each result.
Record whether timings include setup and witness generation.
Use the release benchmark profile for comparisons; the `profiling` profile
uses different compiler settings.

## Layout

| Crate                     | Role                                                                               |
| ------------------------- | ---------------------------------------------------------------------------------- |
| `circuit`                 | Circuit definitions, witness generation, and matrix operations.                    |
| `spartan`                 | Outer and inner sumchecks that reduce circuit constraints to evaluation claims.    |
| `prover`, `verifier`      | BitZ proving and verification, including virtual witness openings.                 |
| `pcs`                     | Binary commitments, ring-switching, and recursive Ligerito openings through Flock. |
| `gkr`                     | Grand-product reductions over the binary field.                                    |
| `common`, `field`, `poly` | Protocol parameters, claims, field arithmetic, and polynomial operations.          |
| `transcript`, `host`      | Fiat-Shamir transcripts and proof serialization.                                   |
| `tests`                   | Integration tests and proof comparison examples.                                   |
| `bitz-cli`                | Circuit proving tools and benchmarks in PR #68, under `tooling/cli`.               |

## Related work

[BitZ-pcs](https://github.com/albert-garreta/BitZ-pcs), previously named
`f2z-pcs`, contains a separate research implementation and paper experiments.
Its [bitz-parity branch](https://github.com/albert-garreta/BitZ-pcs/tree/bitz-parity)
contains PCS transcript comparison work against this implementation.
Full Spartan-plus-BitZ parity remains separate work.

The repositories have different interfaces and benchmark drivers.
Compare results only with matching workloads, proof configurations, and timing
definitions.

See the [optimization tracker](https://github.com/worldfnd/BitZ/issues/41)
for current performance work.
