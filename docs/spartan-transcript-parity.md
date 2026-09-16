# Spartan transcript parity

The opt-in compatibility path independently executes the `f2z-pcs` ordinary U32
Spartan profile through the last inner sumcheck challenge. It does not use the
existing benchmark const-field Spartan kernels, and has no dependency on
`f2z-pcs`. The existing benchmark entry points remain available.

The supported public profile is deliberately exact: 32,768 U32 multiplications,
W1 packing, 15 outer cubic rounds, 17 inner quadratic rounds, Lambda100,
plain UDR `udrg:1:4`, BLAKE3, no skip, no OOD, and zero grinding. The runtime
prime is sampled from `[2^110, 2^111 - 1]`. A different profile requires an
explicit implementation update; it cannot silently reuse this profile's binding.

## Run the comparison

Use fresh output directories. In the reference worktree:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run --release --locked \
  --example u32_mul_transcript -- \
  --variant plain-udr --through-spartan --fixture canonical \
  --out /tmp/spartan-reference
```

In this benchmark worktree:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run --release --locked \
  -p tests --example u32_mul_transcript -- \
  --reference /tmp/spartan-reference --out /tmp/spartan-benchmark \
  --fixture canonical
```

Both commands support `--fixture edges` or an unsigned 64-bit seed such as
`--fixture 42`. Use the same fixture on both sides. `edges` covers zero, one,
maximum U32 operands and high-bit products. The seeded fixture uses a documented
LCG in each implementation, with multiplication performed locally. These are
private witness recipes, not additional public operands or absorbed JSON.

The benchmark also supports `--no-logs`. It still proves, verifies the prefix,
and checks the assignment claim directly; it cannot produce a parity report
without a capture. The reference equivalent is
`--fiat-shamir-transcript-logs false`.

## Boundaries and ownership

- `transcript::reference::Absorbable` emits ordered byte chunks. `ByteTranscript`
  absorbs bytes and squeezes raw bytes with the BLAKE3 feedback framing.
  Neither interface knows about fields or prime sampling. `Transcript` adds
  logical operation recording; JSON descriptions never enter the hash state.
- `field::runtime::PrimeContext` provides explicit-context canonical arithmetic.
  It performs no sampling. Big-integer differential tests cover arithmetic and
  reduction boundaries; inversion and MLE tests cover the other operations.
- `spartan::reference` owns prime and field samplers, protocol message encodings,
  and separate ordinary Spartan proving and verification. Rejected draws and
  accepted-value feedback remain inside their logical squeeze operation.
- `Witness::from_operands` accepts exactly 32,768 `(u32, u32)` pairs. Private
  products and committed packed bits are derived together. `check_claim` uses a
  direct Boolean-hypercube sum, independent of the prover's folding helpers.
- `f2z_compat::capture::run` consumes only the validated public configuration and
  local witness. It constructs its own root, policy/matrix/binding digests,
  prime, messages, and challenges. It writes its own public statement.
- `f2z_compat::parity::compare` reads the reference only after execution. It
  requires complete manifests and exactly 88 events **for each role**, ending
  with inner round 16's challenge. An opening-statement event is rejected.

The comparator checks operation, purpose, semantic context, complete values,
draw counts, chunk boundaries, and every wire field. It replays every squeeze
from a fresh BLAKE3 state and checks both final state digests. Diagnostic spans,
file paths, JSON object ordering and global sequence offsets are irrelevant;
role-local sequences must remain contiguous. The reference's inner-stage context
is present for the prover and absent for the verifier, and is checked that way.

The manifests' root and assignment/policy/matrix digests are checked against the
public statement and captured operations. Terminal points, scales and values are
recomputed from the transcript's inner challenges, final polynomial, outer point
and batching challenge. The benchmark separately checks
`scale * assignment(point) == value` against its actual witness.

A successful prefix verification returns a **pending assignment-opening claim**.
It is not a complete commitment-opening proof. Opening/GKR/PCS equivalence is the
next milestone; the earlier forest prototype is retained separately.

## Errors and regression tests

Any transcript sampling or writer error permanently invalidates that transcript.
Further operations and `finish()` fail, with recording both enabled and disabled.
The completion manifest is published only after both recorders finish and both
terminal claims pass. Partial output is not an eligible comparison artifact.

```sh
RUSTFLAGS='-C target-cpu=native' cargo test --locked \
  -p transcript -p field -p spartan -p f2z-compat
```

Tests include forced rejection boundaries and exhaustion; sink write/flush
failures; mismatched witness lengths; all outer/inner round coefficient
mutations; noncanonical coefficients; changed terminal evaluations and binding;
public-profile mutations; truncated or malformed captures; changed draw counts,
contexts, bytes, and feedback; independent execution against a test-only reference
fixture; and logging-on/off equality. Fresh cross-repository runs cover canonical,
edge and seeded operands. `parity.json` is written only after every comparison
passes.
