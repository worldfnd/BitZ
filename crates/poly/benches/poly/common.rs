/// Hypercube dimensions corresponding to tables of size `2^18` through `2^21`.
pub const LOG_SIZES: &[usize] = &[18, 19, 20, 21];

/// Power-of-two lengths for the additive-subspace natural-domain fast path.
pub const POWER_OF_TWO_NATURAL_SIZES: &[usize] = &[8, 32, 128, 256, 512];

/// Non-power-of-two lengths for the general quadratic domain constructor.
pub const GENERAL_NATURAL_SIZES: &[usize] = &[7, 31, 127, 255, 511];

/// Natural-polynomial lengths used by the current F2Z sumcheck verifier.
pub const PROTOCOL_NATURAL_SIZES: &[usize] = &[3, 4, 6];

/// Step used to generate deterministic but non-sequential benchmark values.
///
/// This conventional Weyl step is `floor(0.618033... * 2^64)`: about a 61.8%
/// jump around a 64-bit clock. Its odd, mixed-bit value avoids the nearly
/// sequential inputs that a step of `1` would produce.
const BENCH_VALUE_STEP: u128 = 0x9e37_79b9_7f4a_7c15;

pub fn field_values<F: From<u128>>(len: usize, seed: u128) -> Vec<F> {
    (0..len)
        .map(|i| {
            F::from(
                (i as u128)
                    .wrapping_mul(BENCH_VALUE_STEP)
                    .wrapping_add(seed),
            )
        })
        .collect()
}
