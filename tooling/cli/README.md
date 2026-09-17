# Circuit benchmarks

Run an end-to-end proof with fresh random inputs:

```sh
cargo run -p bitz-cli --release -- circuit-e2e --circuit sha256-chain --num-blocks 8 --threads 1
```

`--circuit` selects one of these compiled-in adapters:

- `sha256-compression`: one raw block; optional `--initial-state` accepts eight words as 64 hexadecimal digits.
- `sha256-chain`: a positive number of raw blocks from the standard IV; no padding is added.
- `sha256-block-aligned`: a block-aligned message, including an empty message; adds SHA-256 padding.
- `sha256-2kb`: a 2 KiB message, with SHA-256 padding.

Variable-length cases default to one input block. The 2 KiB case uses 32 input blocks. All input bits and SHA outputs are public. SHA reference outputs use `sha2`. Generation happens before timing.

Every proof checks R1CS constraints modulo Q100. SHA-256 constraint residuals are bounded below Q100 in absolute value for Boolean assignments, so reduction preserves their integer equalities. P-256 and combined ECDSA/SHA are excluded until the driver supports integer projection.

The output reports the selected opening path, constraint and witness sizes, and setup/witness/commit/prove/verify timings. `total_prove_ms` includes commitment and proving only. Input generation and thread-pool initialization are excluded.

Run the four circuits through end-to-end and individual setup, witness, commitment, proving, and verification benchmarks:

```sh
RAYON_NUM_THREADS=1 cargo bench -p bitz-cli --bench circuits
RAYON_NUM_THREADS=1 cargo bench -p bitz-cli --bench circuits -- --test
```

The second command smoke-tests each benchmark once. Input generation is excluded from measurements. End-to-end and setup benchmarks generate fresh instances for each sample; the other stages reuse one generated instance per benchmark case. SHA chain and block-aligned benchmark cases use one input block; use the CLI for other lengths. These measurements do not establish parity with BitZ-pcs.
