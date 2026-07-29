//! Timings for the patterns the prover spends its time in.
//!
//! Run with `cargo bench -p field`. Reps via `F2Z_BENCH_REPS`.
//!
//! Each figure is the median of several samples, because run-to-run drift on a
//! laptop exceeds most differences worth seeing, and the working set stays in
//! cache so what is measured is arithmetic rather than memory.
//!
//! The header reports which kernel and which instructions the build selected.
//! Both are silent: `pmull` needs the `aes` target feature and the three-way
//! `eor3` needs `sha3`, and `aarch64-unknown-linux-gnu` enables neither by
//! default. Dropping `eor3` alone costs about a third of the multiply.

use std::hint::black_box;
use std::time::Instant;

use field::gf128::KERNEL;
use field::{F128, FixedBasePow, Wide256};

/// Elements per pass. 1024 of them is 16 KiB, so a pair of operand arrays
/// stays in L1 and the loop is not measuring the memory system.
const N: usize = 1024;
/// Passes over the operands per sample. Enough that a sample runs for several
/// milliseconds: shorter than that and the core has not reached a steady clock,
/// which inflated every figure here by about 2x before it was raised.
///
/// The scalar kernel multiplies bit by bit and is some two orders of magnitude
/// slower, so it gets a smaller count. It is a correctness path, and no figure
/// from it is comparable with one from a SIMD build anyway.
const PASSES: usize = if cfg!(all(target_arch = "aarch64", target_feature = "aes")) {
    4096
} else {
    32
};
/// Repeats for the patterns whose unit cost is hundreds of nanoseconds, so
/// their samples last as long as the cheap ones.
const SLOW_PASSES: usize = 16;

fn reps() -> usize {
    std::env::var("F2Z_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

/// Deterministic operands. xorshift rather than a dependency: the bench only
/// needs values that are not all alike.
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

/// One measurement of `f`, in nanoseconds per operation.
fn time(ops: usize, f: impl Fn()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1e9 / ops as f64
}

struct Bench {
    rows: Vec<(&'static str, f64)>,
    reps: usize,
}

impl Bench {
    /// Times `f` `reps` times and keeps the median, after one warm-up run.
    /// Samples of a pattern run consecutively, so figures from different
    /// patterns are comparable only up to the drift between them.
    fn run(&mut self, name: &'static str, ops: usize, f: impl Fn()) {
        f(); // warm up: fault the operands in and let the clock ramp
        let samples = (0..self.reps).map(|_| time(ops, &f)).collect();
        self.rows.push((name, median(samples)));
    }
}

fn main() {
    let reps = reps();
    let xs = operands(N, 0x243f_6a88_85a3_08d3);
    let ys = operands(N, 0x1319_8a2e_0370_7344);
    let mut b = Bench {
        rows: Vec::new(),
        reps,
    };

    let eor3 = cfg!(target_feature = "sha3");
    println!(
        "kernel {KERNEL}, eor3 {}, {reps} reps",
        if eor3 { "yes" } else { "no" }
    );

    // Independent products: throughput, the shape of the sumcheck round bodies.
    b.run("mul/batch", N * PASSES, || {
        let mut acc = F128::ZERO;
        for _ in 0..PASSES {
            for i in 0..N {
                acc += xs[i] * ys[i];
            }
        }
        black_box(acc);
    });

    // A dependent chain: latency, where throughput cannot be hidden.
    b.run("mul/chain", N * PASSES, || {
        let mut a = xs[0];
        for _ in 0..(N * PASSES) {
            a *= ys[0];
        }
        black_box(a);
    });

    b.run("square/chain", N * PASSES, || {
        let mut a = xs[0];
        for _ in 0..(N * PASSES) {
            a = a.square();
        }
        black_box(a);
    });

    // The same products as mul/batch, accumulated unreduced and reduced once.
    b.run("wide/dot", N * PASSES, || {
        let mut acc = Wide256::zero();
        for _ in 0..PASSES {
            for i in 0..N {
                acc += Wide256::mul(xs[i], ys[i]);
            }
        }
        black_box(acc.reduce());
    });

    b.run("inverse", N * SLOW_PASSES, || {
        for _ in 0..SLOW_PASSES {
            for &x in &xs {
                black_box(x.inverse());
            }
        }
    });

    // Full-width exponents: what the fold values are.
    let comb = FixedBasePow::new(F128::GENERATOR, 8);
    let exps: Vec<u128> = xs
        .iter()
        .map(|x| (x.hi as u128) << 64 | x.lo as u128)
        .collect();
    b.run("pow/square-and-multiply", N * SLOW_PASSES, || {
        for _ in 0..SLOW_PASSES {
            for &e in &exps {
                black_box(F128::GENERATOR.pow(e));
            }
        }
    });
    b.run("pow/comb-w8", N * SLOW_PASSES, || {
        for _ in 0..SLOW_PASSES {
            for &e in &exps {
                black_box(comb.pow(e));
            }
        }
    });

    let width = b.rows.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (name, ns) in &b.rows {
        println!("  {name:<width$}  {ns:9.3} ns/op");
    }
}
