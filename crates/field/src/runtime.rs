//! Canonical arithmetic for the transcript-selected U32 projection modulus.
//! The context is explicit; it never changes the existing const-modulus field.
use num_bigint::BigUint;
use num_traits::ToPrimitive;

#[derive(Clone, Debug)]
pub struct PrimeContext {
    q: u128,
    bits: u32,
    mu: u128,
}

impl PrimeContext {
    /// Arithmetic only. Primality must be established by the protocol sampler.
    pub fn new(q: u128) -> Result<Self, &'static str> {
        if q < 3 || q & 1 == 0 || q >= 1 << 126 {
            return Err("modulus must be odd and below 2^126");
        }
        let bits = 128 - q.leading_zeros();
        let mu = ((BigUint::from(1u8) << (2 * bits as usize)) / BigUint::from(q))
            .to_u128()
            .ok_or("reciprocal overflow")?;
        Ok(Self { q, bits, mu })
    }
    pub fn modulus(&self) -> u128 {
        self.q
    }
    pub const fn zero(&self) -> u128 {
        0
    }
    pub const fn one(&self) -> u128 {
        1
    }
    pub fn reduce(&self, value: u128) -> u128 {
        value % self.q
    }
    pub fn neg(&self, value: u128) -> u128 {
        self.sub(0, value)
    }
    pub fn bits(&self) -> u32 {
        self.bits
    }
    pub fn canonical(&self, v: u128) -> Result<u128, &'static str> {
        if v < self.q {
            Ok(v)
        } else {
            Err("noncanonical field value")
        }
    }
    #[inline]
    pub fn add(&self, a: u128, b: u128) -> u128 {
        debug_assert!(a < self.q && b < self.q);
        let s = a + b;
        if s >= self.q { s - self.q } else { s }
    }
    #[inline]
    pub fn sub(&self, a: u128, b: u128) -> u128 {
        debug_assert!(a < self.q && b < self.q);
        if a >= b { a - b } else { self.q - (b - a) }
    }
    #[inline]
    pub fn mul(&self, a: u128, b: u128) -> u128 {
        debug_assert!(a < self.q && b < self.q);
        let (lo, hi) = wide(a, b);
        let top = shift(lo, hi, self.bits - 1);
        let (ml, mh) = wide(top, self.mu);
        let quotient = shift(ml, mh, self.bits + 1);
        let mut r = lo.wrapping_sub(wide(quotient, self.q).0);
        if r >= self.q {
            r -= self.q;
        }
        if r >= self.q {
            r -= self.q;
        }
        r
    }
    pub fn pow(&self, mut a: u128, mut n: u128) -> u128 {
        let mut r = 1;
        while n != 0 {
            if n & 1 != 0 {
                r = self.mul(r, a);
            }
            a = self.mul(a, a);
            n >>= 1;
        }
        r
    }
    pub fn inverse(&self, a: u128) -> Result<u128, &'static str> {
        self.canonical(a)?;
        if a == 0 {
            return Err("inverse of zero");
        }
        let r = self.pow(a, self.q - 2);
        if self.mul(a, r) != 1 {
            return Err("invalid inverse");
        }
        Ok(r)
    }
    pub fn eq_table(&self, point: &[u128]) -> Vec<u128> {
        let mut v = vec![1];
        for &r in point {
            let n = v.len();
            v.resize(2 * n, 0);
            for i in 0..n {
                let right = self.mul(v[i], r);
                v[n + i] = right;
                v[i] = self.sub(v[i], right);
            }
        }
        v
    }
    pub fn eq(&self, a: &[u128], b: &[u128]) -> u128 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(1, |p, (&a, &b)| {
            self.mul(
                p,
                self.add(self.mul(a, b), self.mul(self.sub(1, a), self.sub(1, b))),
            )
        })
    }
    pub fn evaluate(&self, values: &[u128], point: &[u128]) -> u128 {
        assert_eq!(values.len(), 1 << point.len());
        let mut v = values.to_vec();
        for &r in point {
            for i in 0..v.len() / 2 {
                v[i] = self.add(v[2 * i], self.mul(r, self.sub(v[2 * i + 1], v[2 * i])));
            }
            v.truncate(v.len() / 2);
        }
        v[0]
    }
}

#[inline]
fn wide(a: u128, b: u128) -> (u128, u128) {
    let (al, ah, bl, bh) = (a as u64 as u128, a >> 64, b as u64 as u128, b >> 64);
    let (mid, carry) = (al * bh).overflowing_add(ah * bl);
    let (lo, carry2) = (al * bl).overflowing_add(mid << 64);
    (
        lo,
        ah * bh + (mid >> 64) + ((carry as u128) << 64) + carry2 as u128,
    )
}
#[inline]
fn shift(lo: u128, hi: u128, n: u32) -> u128 {
    if n == 0 {
        lo
    } else {
        (lo >> n) | (hi << (128 - n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arithmetic_matches_big_integers() {
        for q in [
            3,
            5,
            65537,
            (1u128 << 64) - 59,
            (1u128 << 100) - 15,
            (1u128 << 110) + 1,
            (1u128 << 111) - 1,
            (1u128 << 125) - 9,
            (1u128 << 126) - 1,
        ] {
            let f = PrimeContext::new(q).unwrap();
            let mut seed = 17u128;
            for _ in 0..1000 {
                seed = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
                let a = seed % q;
                seed = seed.rotate_left(53) ^ 0xbeef;
                let b = seed % q;
                let expected = ((BigUint::from(a) * BigUint::from(b)) % BigUint::from(q))
                    .to_u128()
                    .unwrap();
                assert_eq!(f.mul(a, b), expected);
                assert_eq!(f.sub(f.add(a, b), b), a);
            }
            for a in [0, 1, q / 2, q - 2, q - 1] {
                for b in [0, 1, q / 2, q - 2, q - 1] {
                    assert_eq!(
                        f.mul(a, b),
                        ((BigUint::from(a) * BigUint::from(b)) % BigUint::from(q))
                            .to_u128()
                            .unwrap()
                    );
                    assert_eq!(f.add(a, b), (a + b) % q);
                    assert_eq!(f.sub(a, b), (a + q - b) % q);
                    assert_eq!(f.add(a, f.neg(a)), 0);
                }
            }
            assert!(f.canonical(q).is_err());
        }
    }
}

#[cfg(test)]
mod field_tests {
    use super::*;
    #[test]
    fn inverses_and_mle_match_direct_evaluation() {
        for q in [65537, 0x7e491c075b45202661dd17daefefu128] {
            let f = PrimeContext::new(q).unwrap();
            for a in [1, 2, 17, q - 1] {
                assert_eq!(f.mul(a, f.inverse(a).unwrap()), 1);
            }
            assert!(f.inverse(0).is_err());
            assert!(f.inverse(q).is_err());
            let point = [2, 3, 5];
            let values = [7, 11, 13, 17, 19, 23, 29, 31];
            let mut sum = 0;
            for (i, &v) in values.iter().enumerate() {
                let mut w = 1;
                for (bit, &r) in point.iter().enumerate() {
                    w = f.mul(w, if i >> bit & 1 == 1 { r } else { f.sub(1, r) });
                }
                sum = f.add(sum, f.mul(v, w));
            }
            assert_eq!(f.evaluate(&values, &point), sum);
            assert_eq!(f.eq_table(&point).iter().fold(0, |a, &b| f.add(a, b)), 1);
        }
    }
}
