//! The same field, two implementations: this crate against binius64's.
//!
//! binius64's `BinaryField128bGhash` is `GF(2^128)` modulo the same polynomial
//! in the same bit order, so the two types hold the same element for the same
//! bits and every pattern below can be run twice on one set of operands. The
//! run starts by checking that agreement, because a timing against a different
//! field is not a comparison.
//!
//! Run with `cargo bench -p field --bench binius64`. Reps via `BitZ_BENCH_REPS`.
//!
//! The two implementations are timed rep by rep in the same loop rather than
//! one after the other, so the drift that makes a single run on this machine
//! unusable moves both figures together instead of landing on one of them.
//!
//! `ratio` is theirs over ours: above 1 means this crate is faster.

use std::hint::black_box;
use std::time::Instant;

use binius_field::arithmetic_traits::{InvertOrZero, Square};
use binius_field::{BinaryField128bGhash as Ghash, Field, WideMul};
use field::gf128::KERNEL;
use field::{F128, Wide256};
use num_traits::{ConstZero, Inv, Pow};

/// Elements per pass, matching the `gf128` bench: 1024 of them is 16 KiB, so a
/// pair of operand arrays stays in L1 and neither implementation is measured
/// against the memory system.
const N: usize = 1024;
/// Passes per sample. The scalar fallback is some two orders of magnitude
/// slower than a PMULL build, so it gets a smaller count; both implementations
/// fall back together, so the comparison still holds there.
const PASSES: usize = if cfg!(all(target_arch = "aarch64", target_feature = "aes")) {
    4096
} else {
    32
};
/// Repeats for the patterns whose unit cost is hundreds of nanoseconds.
const SLOW_PASSES: usize = 16;

fn reps() -> usize {
    std::env::var("BITZ_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

/// Deterministic operands, the same generator the `gf128` bench uses.
fn operands(count: usize, seed: u64) -> Vec<F128> {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    (0..count).map(|_| F128::new(next(), next())).collect()
}

fn to_binius(a: F128) -> Ghash {
    Ghash::from(a.to_u128())
}

fn from_binius(a: Ghash) -> F128 {
    F128::from(u128::from(a))
}

/// One measurement of `f`, in nanoseconds per operation.
fn time(ops: usize, f: impl Fn()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1e9 / ops as f64
}

struct Cmp {
    rows: Vec<(&'static str, f64, f64)>,
    reps: usize,
}

impl Cmp {
    /// Times both implementations of one pattern, alternating within each rep,
    /// and keeps the median of each.
    fn run(&mut self, name: &'static str, ops: usize, ours: impl Fn(), theirs: impl Fn()) {
        ours(); // warm up: fault the operands in and let the clock ramp
        theirs();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..self.reps {
            a.push(time(ops, &ours));
            b.push(time(ops, &theirs));
        }
        self.rows.push((name, median(a), median(b)));
    }
}

/// The two types have to name the same element and agree on every operation
/// timed below, or the figures are of two different fields.
fn check_agreement(xs: &[F128], ys: &[F128]) {
    for (&x, &y) in xs.iter().zip(ys) {
        let (bx, by) = (to_binius(x), to_binius(y));
        assert_eq!(x, from_binius(bx), "encoding {x:?}");
        assert_eq!(x * y, from_binius(bx * by), "{x:?} * {y:?}");
        assert_eq!(x.square(), from_binius(bx.square()), "square {x:?}");
        assert_eq!(
            Wide256::mul(x, y).reduce(),
            from_binius(Ghash::reduce(Ghash::wide_mul(bx, by))),
            "wide {x:?} * {y:?}"
        );
        assert_eq!(
            x.inv().unwrap_or(F128::ZERO),
            from_binius(bx.invert_or_zero()),
            "inverse {x:?}"
        );
        // The timed exponents are the operands read as integers, so raising the
        // generator to `x` here is the same call the `pow` row makes.
        let e = x.to_u128();
        assert_eq!(
            F128::GENERATOR.pow(e),
            from_binius(to_binius(F128::GENERATOR).pow([e as u64, (e >> 64) as u64])),
            "pow {e:#034x}"
        );
    }
}

// One pass of each pattern, kept out of line on both sides.
//
// Left inline, all twelve bodies share one register allocator inside `main`,
// and it does not treat them alike: in an earlier form of this bench LLVM
// rematerialized the `0x87` constant with a `dup` inside binius64's multiply
// loop while hoisting ours, which is worth most of a PMULL slot per element
// and moved the reported ratio from 1.06 to 1.27. `#[inline(never)]` gives
// each body its own allocation, and the call is amortized over a whole pass.
//
// The multiply loops then compile to the same 18 instructions on both sides,
// so `mul/batch` is a wash. `mul/chain` is not, and it is also codegen rather
// than arithmetic: out of line, LLVM holds binius64's loop-carried element in
// a GPR pair and rebuilds the vector every iteration, where this kernel's
// stays in `q0`. Inlined into one large `main` the difference disappears and
// the row reads 1.00, so take it as fragile, not as a fact about the
// multiply — both are the same 6 PMULL. What decides the representation is
// not their PMULL helper: rewriting it to `vmull_high_p64` and
// `vreinterpretq_u64_p128`, so no product is scalar in the IR, left both the
// timing and the round trip unchanged.

#[inline(never)]
fn ours_dot(xs: &[F128], ys: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (&x, &y) in xs.iter().zip(ys) {
        acc += x * y;
    }
    acc
}

#[inline(never)]
fn binius_dot(xs: &[Ghash], ys: &[Ghash]) -> Ghash {
    let mut acc = Ghash::ZERO;
    for (&x, &y) in xs.iter().zip(ys) {
        acc += x * y;
    }
    acc
}

#[inline(never)]
fn ours_wide_dot(xs: &[F128], ys: &[F128]) -> F128 {
    let mut acc = Wide256::zero();
    for (&x, &y) in xs.iter().zip(ys) {
        acc += Wide256::mul(x, y);
    }
    acc.reduce()
}

#[inline(never)]
fn binius_wide_dot(xs: &[Ghash], ys: &[Ghash]) -> Ghash {
    let mut acc = <Ghash as WideMul>::Output::default();
    for (&x, &y) in xs.iter().zip(ys) {
        acc += Ghash::wide_mul(x, y);
    }
    Ghash::reduce(acc)
}

#[inline(never)]
fn ours_mul_chain(mut a: F128, b: F128, n: usize) -> F128 {
    for _ in 0..n {
        a *= b;
    }
    a
}

#[inline(never)]
fn binius_mul_chain(mut a: Ghash, b: Ghash, n: usize) -> Ghash {
    for _ in 0..n {
        a *= b;
    }
    a
}

#[inline(never)]
fn ours_square_chain(mut a: F128, n: usize) -> F128 {
    for _ in 0..n {
        a = a.square();
    }
    a
}

#[inline(never)]
fn binius_square_chain(mut a: Ghash, n: usize) -> Ghash {
    for _ in 0..n {
        a = a.square();
    }
    a
}

#[inline(never)]
fn ours_inverses(xs: &[F128]) {
    for &x in xs {
        black_box(x.inv());
    }
}

#[inline(never)]
fn binius_inverses(xs: &[Ghash]) {
    for &x in xs {
        black_box(x.invert_or_zero());
    }
}

#[inline(never)]
fn ours_pows(exps: &[u128]) {
    for &e in exps {
        black_box(F128::GENERATOR.pow(e));
    }
}

#[inline(never)]
fn binius_pows(base: Ghash, exps: &[u128]) {
    for &e in exps {
        black_box(base.pow([e as u64, (e >> 64) as u64]));
    }
}

fn main() {
    let reps = reps();
    let xs = operands(N, 0x243f_6a88_85a3_08d3);
    let ys = operands(N, 0x1319_8a2e_0370_7344);
    check_agreement(&xs, &ys);

    let bxs: Vec<Ghash> = xs.iter().copied().map(to_binius).collect();
    let bys: Vec<Ghash> = ys.iter().copied().map(to_binius).collect();

    let mut c = Cmp {
        rows: Vec::new(),
        reps,
    };
    println!(
        "kernel {KERNEL}, eor3 {}, {reps} reps",
        if cfg!(target_feature = "sha3") {
            "yes"
        } else {
            "no"
        }
    );

    // Independent products: throughput, the shape of the sumcheck round bodies.
    c.run(
        "mul/batch",
        N * PASSES,
        || {
            for _ in 0..PASSES {
                black_box(ours_dot(&xs, &ys));
            }
        },
        || {
            for _ in 0..PASSES {
                black_box(binius_dot(&bxs, &bys));
            }
        },
    );

    // A dependent chain: latency, where throughput cannot be hidden.
    c.run(
        "mul/chain",
        N * PASSES,
        || {
            black_box(ours_mul_chain(xs[0], ys[0], N * PASSES));
        },
        || {
            black_box(binius_mul_chain(bxs[0], bys[0], N * PASSES));
        },
    );

    c.run(
        "square/chain",
        N * PASSES,
        || {
            black_box(ours_square_chain(xs[0], N * PASSES));
        },
        || {
            black_box(binius_square_chain(bxs[0], N * PASSES));
        },
    );

    // The same products as mul/batch, accumulated unreduced and reduced once.
    // Both sides defer the reduction; the accumulators differ in width, ours
    // 256 bits and theirs three 128-bit limbs.
    c.run(
        "wide/dot",
        N * PASSES,
        || {
            for _ in 0..PASSES {
                black_box(ours_wide_dot(&xs, &ys));
            }
        },
        || {
            for _ in 0..PASSES {
                black_box(binius_wide_dot(&bxs, &bys));
            }
        },
    );

    // Both invert by Itoh-Tsujii, but only this crate runs the power maps
    // `x -> x^(2^n)` as repeated squarings. binius64 precomputes each one as a
    // byte-indexed table of its `F_2`-linear matrix, so its chain pays lookups
    // where this one pays 127 squarings.
    c.run(
        "inverse",
        N * SLOW_PASSES,
        || {
            for _ in 0..SLOW_PASSES {
                ours_inverses(&xs);
            }
        },
        || {
            for _ in 0..SLOW_PASSES {
                binius_inverses(&bxs);
            }
        },
    );

    // Full-width exponents, square-and-multiply on both sides. This crate's
    // fixed-base comb has no counterpart in binius64, so it is left out.
    let exps: Vec<u128> = xs.iter().map(|x| x.to_u128()).collect();
    let bgen = to_binius(F128::GENERATOR);
    c.run(
        "pow/square-and-multiply",
        N * SLOW_PASSES,
        || {
            for _ in 0..SLOW_PASSES {
                ours_pows(&exps);
            }
        },
        || {
            for _ in 0..SLOW_PASSES {
                binius_pows(bgen, &exps);
            }
        },
    );

    let width = c.rows.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
    println!(
        "  {:<width$}  {:>12}  {:>12}  {:>6}",
        "", "ours", "binius64", "ratio"
    );
    for (name, ours, theirs) in &c.rows {
        println!(
            "  {name:<width$}  {ours:9.3} ns  {theirs:9.3} ns  {:6.2}",
            theirs / ours
        );
    }
}
