//! Mirrored from f2z-pcs `src/bitz/ecdsa.rs` by `scripts/bitz_mirror_statements.py`
//! (tests dropped, paths retargeted, `f2z`/`f2z_unsigned` → `bitz`/`bitz_unsigned`): do not edit here.
//!
//! The paper's SHA-256 + ECDSA statement in their gadget language: one
//! block-aligned message of `64·(2^L − 1)` bytes hashed in circuit (`2^L`
//! compressions with the padding block) and one P-256 verification of the
//! digest, through the vendored `verify_block_aligned_message_circuit`.
//! Public: `L`, the public key `Q` and the signature `(r, s)` (the paper's
//! `Sha256EcdsaStatement`); private: the message and the two inverses the
//! verifier takes as advice.
//!
//! The instance derives from a seed on both sides — the message bytes, the
//! private key and the nonce (SplitMix64) — signed here with a plain
//! big-integer P-256 (affine formulas, inversions by Fermat), so their
//! examples and ours agree without exchanging files.

use circuit::Circuit;
use circuit::p256::{VERIFY_DIGEST_INPUT_BITS, verify_digest_circuit};
use circuit::sha256::sha256_block_aligned_circuit;
use num_bigint::BigUint;
use num_traits::Zero;
use sha2::{Digest, Sha256};

use bitz_cli::end_to_end::{CircuitStatement, Error};
/// The probes' seed derivation.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `2^L` compressions, `L` in this range (the paper's).
pub const LOG_COMPRESSIONS: std::ops::RangeInclusive<u8> = 3..=16;

/// The six P-256 words after the digest: `Q.x, Q.y, r, s, r⁻¹, s⁻¹`.
pub const P256_INPUT_BITS: usize = VERIFY_DIGEST_INPUT_BITS - 256;

/// The P-256 field prime, group order, `b`, and the generator.
const P256_P: &str = "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff";
const P256_N: &str = "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";
const P256_GX: &str = "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296";
const P256_GY: &str = "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5";

fn hex(text: &str) -> BigUint {
    BigUint::parse_bytes(text.as_bytes(), 16).expect("a hex constant")
}

/// An affine point, `None` at infinity.
type Point = Option<(BigUint, BigUint)>;

/// Host-side P-256, enough to sign: affine add and double with Fermat
/// inversions, double-and-add.
struct Curve {
    p: BigUint,
    n: BigUint,
    g: (BigUint, BigUint),
}

impl Curve {
    fn new() -> Self {
        Self {
            p: hex(P256_P),
            n: hex(P256_N),
            g: (hex(P256_GX), hex(P256_GY)),
        }
    }

    fn inverse(&self, value: &BigUint, modulus: &BigUint) -> BigUint {
        value.modpow(&(modulus - 2u32), modulus)
    }

    fn sub_mod(&self, a: &BigUint, b: &BigUint) -> BigUint {
        (a + &self.p - b % &self.p) % &self.p
    }

    fn add(&self, left: &Point, right: &Point) -> Point {
        let (Some((x1, y1)), Some((x2, y2))) = (left, right) else {
            return left.clone().or_else(|| right.clone());
        };
        let lambda = if x1 == x2 {
            if (y1 + y2) % &self.p == BigUint::zero() {
                return None;
            }
            // 3x² − 3 over 2y (a = −3).
            let numerator = self.sub_mod(&(3u32 * x1 * x1 % &self.p), &BigUint::from(3u32));
            numerator * self.inverse(&(2u32 * y1 % &self.p), &self.p) % &self.p
        } else {
            self.sub_mod(y2, y1) * self.inverse(&self.sub_mod(x2, x1), &self.p) % &self.p
        };
        let x3 = self.sub_mod(&self.sub_mod(&(&lambda * &lambda % &self.p), x1), x2);
        let y3 = self.sub_mod(&(&lambda * self.sub_mod(x1, &x3) % &self.p), y1);
        Some((x3, y3))
    }

    fn mul(&self, scalar: &BigUint, point: &Point) -> Point {
        let mut result: Point = None;
        for bit in (0..scalar.bits()).rev() {
            result = self.add(&result, &result);
            if scalar.bit(bit) {
                result = self.add(&result, point);
            }
        }
        result
    }

    /// A scalar in `1..n` from four seed words.
    fn scalar(&self, state: &mut u64) -> BigUint {
        let mut bytes = [0u8; 32];
        for chunk in bytes.chunks_exact_mut(8) {
            chunk.copy_from_slice(&splitmix64(state).to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes) % (&self.n - 1u32) + 1u32
    }
}

fn to_bytes_be(value: &BigUint) -> [u8; 32] {
    let bytes = value.to_bytes_be();
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(&bytes);
    out
}

/// The statement: `2^log_compressions` compressions of a seeded message,
/// the key and signature the seed produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EcdsaStatement {
    pub log_compressions: u8,
    pub seed: u64,
    pub qx: [u8; 32],
    pub qy: [u8; 32],
    pub r: [u8; 32],
    pub s: [u8; 32],
}

impl EcdsaStatement {
    /// The message: `64·(2^L − 1)` bytes of the seed's stream.
    pub fn message(log_compressions: u8, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let bytes = 64 * ((1usize << log_compressions) - 1);
        let mut message = Vec::with_capacity(bytes);
        while message.len() < bytes {
            message.extend_from_slice(&splitmix64(&mut state).to_le_bytes());
        }
        message
    }

    /// Signs the seeded message with a seeded key and nonce (the key and
    /// nonce streams follow the message's).
    pub fn seeded(log_compressions: u8, seed: u64) -> Result<Self, Error> {
        if !LOG_COMPRESSIONS.contains(&log_compressions) {
            return Err(Error::Input("compression exponent must be in 3..=16"));
        }
        let curve = Curve::new();
        let message = Self::message(log_compressions, seed);
        let z = BigUint::from_bytes_be(&Sha256::digest(&message)) % &curve.n;
        let mut state = seed ^ 0x5eed_5eed_5eed_5eed;
        let d = curve.scalar(&mut state);
        let g = Some(curve.g.clone());
        let q = curve.mul(&d, &g).expect("the key is not the identity");
        let (r, s) = loop {
            let k = curve.scalar(&mut state);
            let (rx, _) = curve.mul(&k, &g).expect("the nonce point is not the identity");
            let r = rx % &curve.n;
            if r.is_zero() {
                continue;
            }
            let s = curve.inverse(&k, &curve.n) * ((&z + &r * &d) % &curve.n) % &curve.n;
            if s.is_zero() {
                continue;
            }
            break (r, s);
        };
        Ok(Self {
            log_compressions,
            seed,
            qx: to_bytes_be(&q.0),
            qy: to_bytes_be(&q.1),
            r: to_bytes_be(&r),
            s: to_bytes_be(&s),
        })
    }

    pub fn message_bits(&self) -> usize {
        512 * ((1usize << self.log_compressions) - 1)
    }

    /// The six P-256 input words: `Q.x, Q.y, r, s, r⁻¹, s⁻¹`, as the
    /// vendored verifier takes them (little-endian bits of each integer).
    fn p256_words(&self) -> [BigUint; 6] {
        let curve = Curve::new();
        let r = BigUint::from_bytes_be(&self.r);
        let s = BigUint::from_bytes_be(&self.s);
        [
            BigUint::from_bytes_be(&self.qx),
            BigUint::from_bytes_be(&self.qy),
            r.clone(),
            s.clone(),
            curve.inverse(&r, &curve.n),
            curve.inverse(&s, &curve.n),
        ]
    }

    /// The circuit's inputs: the message in stream order (bit 7 of byte 0
    /// first), then the six words, 256 little-endian bits each.
    pub fn input(&self) -> Vec<bool> {
        let message = Self::message(self.log_compressions, self.seed);
        let mut bits: Vec<bool> = (0..self.message_bits())
            .map(|bit| message[bit / 8] >> (7 - bit % 8) & 1 != 0)
            .collect();
        for word in self.p256_words() {
            bits.extend((0..256).map(|bit| word.bit(bit)));
        }
        bits
    }

    /// The inverse of [`CircuitStatement::public_bytes`] (the seed is not
    /// public; it comes from the dump).
    pub fn from_public_bytes(public: &[u8], seed: u64) -> Option<Self> {
        if public.len() != 1 + 4 * 32 {
            return None;
        }
        let word = |i: usize| -> [u8; 32] { public[1 + 32 * i..1 + 32 * (i + 1)].try_into().expect("32 bytes") };
        Some(Self {
            log_compressions: public[0],
            seed,
            qx: word(0),
            qy: word(1),
            r: word(2),
            s: word(3),
        })
    }
}

impl CircuitStatement for EcdsaStatement {
    fn domain(&self) -> &'static [u8] {
        b"sha256-ecdsa/v1"
    }

    /// `L`, then `Q.x`, `Q.y`, `r`, `s` as 32 big-endian bytes each (the
    /// paper's statement bytes).
    fn public_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.log_compressions];
        for word in [&self.qx, &self.qy, &self.r, &self.s] {
            bytes.extend_from_slice(word);
        }
        bytes
    }

    fn input_bits(&self) -> usize {
        self.message_bits() + P256_INPUT_BITS
    }

    fn synthesize<CS: Circuit>(&self, cs: &mut CS, inputs: &[CS::Bool]) -> Result<(), Error> {
        if !LOG_COMPRESSIONS.contains(&self.log_compressions) {
            return Err(Error::Input("compression exponent must be in 3..=16"));
        }
        if inputs.len() != self.input_bits() {
            return Err(Error::Input("wrong ECDSA input length"));
        }
        let message_bits = self.message_bits();
        // The public words are inputs pinned to their public values; the
        // inverses are free advice the verifier checks.
        let public = [&self.qx, &self.qy, &self.r, &self.s];
        for (word, value) in public.into_iter().enumerate() {
            let value = BigUint::from_bytes_be(value);
            for bit in 0..256 {
                constrain_bit(cs, inputs[message_bits + 256 * word + bit].clone(), value.bit(bit as u64));
            }
        }
        // The vendored `verify_block_aligned_message_circuit`, spelled out:
        // the digest in stream order, reversed into the verifier's
        // little-endian word, then the six input words.
        let digest = sha256_block_aligned_circuit(cs, message_bits, |bit| inputs[bit].clone());
        let verifier_inputs: Box<[CS::Bool; VERIFY_DIGEST_INPUT_BITS]> = (0..VERIFY_DIGEST_INPUT_BITS)
            .map(|index| {
                if index < 256 {
                    digest[255 - index].clone()
                } else {
                    inputs[message_bits + index - 256].clone()
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
            .try_into()
            .map_err(|_| Error::Input("wrong P-256 input length"))?;
        verify_digest_circuit(cs, &verifier_inputs);
        Ok(())
    }
}

fn constrain_bit<CS: Circuit>(cs: &mut CS, bit: CS::Bool, expected: bool) {
    let value = cs.bitz::<1>(bit);
    let expected = CS::Z::<1>::from(CS::Coefficient::<1>::from(u64::from(expected)));
    cs.assert_r1c::<1>(
        CS::Z::<1>::from(CS::Coefficient::<1>::from(1u64)),
        value,
        expected,
    );
}
