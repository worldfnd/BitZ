//! NEON kernels for `GF(2^128)`.
//!
//! `pmull` does a 64x64 carryless multiply in one instruction — all of
//! [`super::portable::clmul64`] in a few cycles. Everything here is the
//! portable pipeline with that substitution, held in vector registers so no
//! product crosses into the general-purpose file mid-multiply.
//!
//! `pmull` is an ARM crypto extension, not baseline NEON, so the gate is
//! `aes`, not `neon`. Every intrinsic call below is sound on that basis: the
//! module compiles only with the feature enabled crate-wide.

use core::arch::aarch64::{
    uint64x2_t, vdupq_n_u64, veorq_u64, vextq_u64, vgetq_lane_u64, vld1q_u64, vmull_high_p64,
    vmull_p64, vreinterpretq_p64_u64, vreinterpretq_u64_p128, vst1q_u64,
};

use super::REDUCTION;

/// Taking an address here does not force the array to memory — the compiled
/// loop reads straight from the caller's slice. Building the register lane-wise
/// with `vcombine_u64` to avoid the address measured inside run-to-run noise,
/// so this stays as the simpler form.
#[inline(always)]
unsafe fn load(w: [u64; 2]) -> uint64x2_t {
    unsafe { vld1q_u64(w.as_ptr()) }
}

#[inline(always)]
unsafe fn store(v: uint64x2_t) -> [u64; 2] {
    let mut out = [0u64; 2];
    unsafe { vst1q_u64(out.as_mut_ptr(), v) };
    out
}

/// Carryless product of the low lanes: `a[0] * b[0]`.
#[inline(always)]
unsafe fn pmull_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe { vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b))) }
}

/// Carryless product of the high lanes: `a[1] * b[1]`.
#[inline(always)]
unsafe fn pmull_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ))
    }
}

/// One word times the reduction polynomial: `w * g`, at most 71 bits.
#[inline(always)]
unsafe fn pmull_g(w: u64) -> uint64x2_t {
    unsafe { vreinterpretq_u64_p128(vmull_p64(w, REDUCTION)) }
}

/// `v * X^64`, discarding the half that would land at or above `X^128`.
#[inline(always)]
unsafe fn shift_up(v: uint64x2_t) -> uint64x2_t {
    unsafe { vextq_u64::<1>(vdupq_n_u64(0), v) }
}

/// The half of `v * X^64` that lands at or above `X^128`, rebased to zero.
#[inline(always)]
unsafe fn shift_down(v: uint64x2_t) -> uint64x2_t {
    unsafe { vextq_u64::<1>(v, vdupq_n_u64(0)) }
}

/// 128x128 -> 256-bit carryless product as `(low, high)`, schoolbook, 4 PMULL.
///
/// Karatsuba would trade one PMULL for an XOR-dependency chain — a loss on
/// M-class cores, so its absence here is deliberate.
#[inline(always)]
unsafe fn clmul_256(a: uint64x2_t, b: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    unsafe {
        let swapped = vextq_u64::<1>(b, b);
        let low = pmull_lo(a, b); // a0*b0
        let high = pmull_hi(a, b); // a1*b1
        let mid = veorq_u64(pmull_lo(a, swapped), pmull_hi(a, swapped)); // a0*b1 + a1*b0
        (
            veorq_u64(low, shift_up(mid)),
            veorq_u64(high, shift_down(mid)),
        )
    }
}

/// Reduce a 256-bit product with 3 PMULL: fold the whole high half against `g`
/// at once, then clear the at-most-7-bit spill that fold leaves behind.
#[inline(always)]
unsafe fn reduce_256(low: uint64x2_t, high: uint64x2_t) -> uint64x2_t {
    unsafe {
        // high * g, as a 192-bit quantity spread over two products.
        let p0 = pmull_lo(high, vdupq_n_u64(REDUCTION));
        let p1 = pmull_hi(high, vdupq_n_u64(REDUCTION));
        let folded = veorq_u64(low, veorq_u64(p0, shift_up(p1)));
        // `p1`'s high word is the spill; at most 7 bits, so one more
        // substitution lands it inside word 0.
        veorq_u64(folded, pmull_g(vgetq_lane_u64::<1>(p1)))
    }
}

/// Schoolbook product then a single 3-PMULL reduction. 7 PMULL total.
#[inline(always)]
unsafe fn mul_schoolbook(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let (low, high) = clmul_256(a, b);
        reduce_256(low, high)
    }
}

/// Same product, reduced 64 bits at a time between the partial sums. 6 PMULL
/// total — one fewer than [`mul_schoolbook`], at the cost of a longer
/// dependency chain.
///
/// Writing `a*b = t0 + t1*X^64 + t2*X^128`, the first stage rewrites `t2*X^64`
/// as `t2.lo*X^64 + t2.hi*g` and folds it into `t1`; the second does the same
/// to `t1` and folds it into `t0`.
///
/// A candidate for [`mul`]'s default until the field bench decides; the tests
/// already pin it to the portable path.
#[allow(dead_code)]
#[inline(always)]
unsafe fn mul_interleaved(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let swapped = vextq_u64::<1>(b, b);
        let t0 = pmull_lo(a, b);
        let t2 = pmull_hi(a, b);
        let mut t1 = veorq_u64(pmull_lo(a, swapped), pmull_hi(a, swapped));

        t1 = veorq_u64(t1, shift_up(t2));
        t1 = veorq_u64(t1, pmull_g(vgetq_lane_u64::<1>(t2)));

        let folded = veorq_u64(t0, shift_up(t1));
        veorq_u64(folded, pmull_g(vgetq_lane_u64::<1>(t1)))
    }
}

/// In characteristic 2 the cross terms cancel, so the product needs 2 PMULL
/// rather than 4.
#[inline(always)]
unsafe fn square_inner(a: uint64x2_t) -> uint64x2_t {
    unsafe { reduce_256(pmull_lo(a, a), pmull_hi(a, a)) }
}

pub fn mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    // Default pending the field bench; `mul_interleaved` is the alternative.
    unsafe { store(mul_schoolbook(load(a), load(b))) }
}

pub fn square(a: [u64; 2]) -> [u64; 2] {
    unsafe {
        let v = load(a);
        store(square_inner(v))
    }
}

/// Every multiply variant on the same input, so the equivalence test pins all
/// of them to the portable pipeline, not only the one [`mul`] selects.
#[cfg(test)]
pub fn mul_variants(a: [u64; 2], b: [u64; 2]) -> [[u64; 2]; 2] {
    unsafe {
        let (va, vb) = (load(a), load(b));
        [
            store(mul_schoolbook(va, vb)),
            store(mul_interleaved(va, vb)),
        ]
    }
}
