/// Hypercube dimensions corresponding to tables of size `2^18` through `2^21`.
pub const LOG_SIZES: &[usize] = &[18, 19, 20, 21];

/// Natural-domain lengths; fresh-domain construction is quadratic in this value.
pub const NATURAL_SIZES: &[usize] = &[8, 32, 128];

pub fn field_values<F: From<u128>>(len: usize, seed: u128) -> Vec<F> {
    (0..len)
        .map(|i| {
            F::from(
                (i as u128)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(seed),
            )
        })
        .collect()
}
