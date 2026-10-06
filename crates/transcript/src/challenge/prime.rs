//! Public prime sampling from the challenge stream.
//!
//! Candidates uniform among the odd integers of the requested width, sieved
//! by the small primes, then Miller–Rabin with bases drawn from the same
//! stream, so a prover grinding the transcript faces fresh bases at every
//! candidate.
//!
//! Arithmetic is variable-time.

use field::MAX_MODULUS_BITS;
use field::helpers::{self, FieldMetadata};

/// Miller–Rabin bases per candidate.
///
/// A composite passes a base with probability at most `1/4`, so it survives
/// with probability at most `2^-160`, and a search of up to `2^16` candidates
/// accepts a composite with probability below `2^-144`: the 128 bits and its
/// 16 bits of slack.
///
/// A longer search does not happen: at least one odd integer in 44 is prime
/// below `2^126`, so `2^16` composites in a row have probability below `2^-2000`.
const MILLER_RABIN_ROUNDS: u32 = 80;

/// A prime of exactly `bits` bits, `2 <= bits <= MAX_MODULUS_BITS`, from
/// successive `u128` draws, the same on both sides: the fingerprint prime.
///
/// Each candidate is one draw, uniform among the odd integers in
/// `[2^(bits-1), 2^bits)`; each base is uniform in `[2, candidate - 2]` by
/// rejection. The search stops at the first candidate that passes.
///
/// Should be drawn once per proof after the commitment and the statement
/// are absorbed and before the PIOP.
pub fn sample(mut next_u128: impl FnMut() -> u128, bits: u32) -> u128 {
    assert!(
        (2..=MAX_MODULUS_BITS).contains(&bits),
        "prime width out of range"
    );
    let bottom = 1u128 << (bits - 1);
    loop {
        let candidate = bottom | (next_u128() & (bottom - 1)) | 1;
        let is_prime = helpers::sieve(candidate).unwrap_or_else(|| {
            let modulus = FieldMetadata::new(candidate);
            (0..MILLER_RABIN_ROUNDS)
                .all(|_| modulus.strong_round(2 + sample_below(&mut next_u128, candidate - 3)))
        });
        if is_prime {
            return candidate;
        }
    }
}

/// Uniform in `[0, bound)` for `bound >= 2`: draws masked to the bit length
/// of `bound - 1`, accepted below `bound`, so at most half are rejected.
fn sample_below(next_u128: &mut impl FnMut() -> u128, bound: u128) -> u128 {
    debug_assert!(bound >= 2, "nothing to sample");
    let mask = u128::MAX >> (bound - 1).leading_zeros();
    loop {
        let candidate = next_u128() & mask;
        if candidate < bound {
            return candidate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, two outputs per draw.
    fn stream(mut state: u64) -> impl FnMut() -> u128 {
        let mut next_u64 = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        move || u128::from(next_u64()) | (u128::from(next_u64()) << 64)
    }

    /// `is_prime` is exact below `2^81.4` and a strong probable-prime test
    /// above, which is all a check on the sampler's own test needs.
    #[test]
    fn sampled_primes_are_prime_and_of_the_requested_width() {
        for bits in [2, 3, 8, 40, 64, 100, 113, MAX_MODULUS_BITS] {
            for seed in 0..8 {
                let prime = sample(stream(seed | u64::from(bits) << 8), bits);
                assert_eq!(
                    u128::BITS - prime.leading_zeros(),
                    bits,
                    "{bits} bits, seed {seed}"
                );
                assert!(
                    helpers::is_prime(prime),
                    "{bits} bits, seed {seed}: {prime} is composite"
                );
            }
        }
    }

    #[test]
    fn the_prime_is_a_function_of_the_stream() {
        assert_eq!(sample(stream(7), 100), sample(stream(7), 100));
        assert_ne!(sample(stream(7), 100), sample(stream(8), 100));
    }

    #[test]
    fn sampling_below_masks_then_rejects() {
        // Bound 5 masks to three bits: 7 and 5 are rejected, `11 & 7 = 3` is not.
        let mut draws = [7u128, 5, 8 | 3].into_iter();
        assert_eq!(sample_below(&mut || draws.next().unwrap(), 5), 3);
        assert!(draws.next().is_none());
    }

    #[test]
    #[should_panic(expected = "prime width out of range")]
    fn rejects_a_width_the_field_cannot_hold() {
        sample(stream(0), MAX_MODULUS_BITS + 1);
    }
}
