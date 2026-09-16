//! Mirrored from f2z-pcs `src/bitz/modr1cs.rs` by `scripts/bitz_mirror_statements.py`
//! (tests dropped, paths retargeted, `f2z`/`f2z_unsigned` → `bitz`/`bitz_unsigned`): do not edit here.
//!
//! An integer Mod-R1CS instance (Limber's MultiSwap in the paper) in their
//! gadget language, from an instance file both sides read: `A·z ∘ B·z =
//! C·z + m ∘ q` over `ℤ`, every value committed as bits — `WIDE` bits per
//! value, one bit for the values a `v·v = v` row constrains — and, per
//! modular row, the quotient `q` as a `WIDE`-bit hint and the product
//! `t = m·q` as a `2·WIDE`-bit hint with its own rank-1 row `m · q = t`
//! (so every coefficient on a committed bit is a power of two; the moduli
//! and the instance's other constants sit on the constant column). The
//! instance's rows, moduli, coefficients and widths are public through
//! their digest; the values are the witness.
//!
//! The file format is this module's own (`to_bytes`/`from_bytes`); the
//! exporter `bitz_modr1cs_export` writes the paper's 6,209-row MultiSwap
//! instance from the crate's Limber port.

use std::sync::Arc;

use circuit::{Circuit, HintResult, PackedBits};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use sha2::{Digest, Sha256};

use bitz_cli::end_to_end::{CircuitStatement, Error};

/// The width of every value that is not a bit, and of every quotient.
pub const WIDE: usize = 2048;
const WIDE_WORDS: usize = WIDE / 64;
const PRODUCT: usize = 2 * WIDE;
const PRODUCT_WORDS: usize = PRODUCT / 64;
/// Signed intermediates: products of two `WIDE`-bit values, with room.
const LIMBS: usize = PRODUCT / 64 + 1;

/// One coefficient: on a value column, or the constant (`column == None`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub column: Option<u32>,
    pub coefficient: BigUint,
}

/// One row `A·z ∘ B·z = C·z (+ m·q)`; `modulus` zero means exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub a: Vec<Entry>,
    pub b: Vec<Entry>,
    pub c: Vec<Entry>,
    pub modulus: BigUint,
}

/// The instance: the rows, the columns' widths (`1` or [`WIDE`]), the
/// witness values and the quotients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance {
    pub rows: Vec<Row>,
    pub widths: Vec<u16>,
    pub values: Vec<BigUint>,
    pub quotients: Vec<BigUint>,
}

fn put_u64(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_le_bytes());
}

fn put_big(out: &mut Vec<u8>, value: &BigUint) {
    let bytes = value.to_bytes_le();
    put_u64(out, bytes.len());
    out.extend_from_slice(&bytes);
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn u64(&mut self) -> Option<usize> {
        let word = self.bytes.get(self.at..self.at + 8)?;
        self.at += 8;
        Some(u64::from_le_bytes(word.try_into().ok()?) as usize)
    }

    fn big(&mut self) -> Option<BigUint> {
        let len = self.u64()?;
        let bytes = self.bytes.get(self.at..self.at + len)?;
        self.at += len;
        Some(BigUint::from_bytes_le(bytes))
    }

    fn entries(&mut self, columns: usize) -> Option<Vec<Entry>> {
        let count = self.u64()?;
        (0..count)
            .map(|_| {
                let column = self.u64()?;
                let coefficient = self.big()?;
                Some(Entry {
                    column: (column < columns).then_some(column as u32),
                    coefficient,
                })
            })
            .collect()
    }
}

impl Instance {
    /// The public part first (the digest covers it), then the values.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.public_bytes();
        put_u64(&mut out, self.values.len());
        for value in &self.values {
            put_big(&mut out, value);
        }
        put_u64(&mut out, self.quotients.len());
        for quotient in &self.quotients {
            put_big(&mut out, quotient);
        }
        out
    }

    /// The rows, moduli, coefficients and widths.
    fn public_bytes(&self) -> Vec<u8> {
        let mut out = b"bitz/mod-r1cs/instance/v1".to_vec();
        let columns = self.widths.len();
        put_u64(&mut out, self.rows.len());
        put_u64(&mut out, columns);
        for width in &self.widths {
            out.extend_from_slice(&width.to_le_bytes());
        }
        for row in &self.rows {
            put_big(&mut out, &row.modulus);
            for entries in [&row.a, &row.b, &row.c] {
                put_u64(&mut out, entries.len());
                for entry in entries {
                    put_u64(&mut out, entry.column.map_or(columns, |c| c as usize));
                    put_big(&mut out, &entry.coefficient);
                }
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let magic = b"bitz/mod-r1cs/instance/v1";
        if !bytes.starts_with(magic) {
            return None;
        }
        let mut reader = Reader {
            bytes,
            at: magic.len(),
        };
        let rows = reader.u64()?;
        let columns = reader.u64()?;
        let widths = (0..columns)
            .map(|_| {
                let word = bytes.get(reader.at..reader.at + 2)?;
                reader.at += 2;
                Some(u16::from_le_bytes(word.try_into().ok()?))
            })
            .collect::<Option<Vec<_>>>()?;
        let rows = (0..rows)
            .map(|_| {
                let modulus = reader.big()?;
                let a = reader.entries(columns)?;
                let b = reader.entries(columns)?;
                let c = reader.entries(columns)?;
                Some(Row { a, b, c, modulus })
            })
            .collect::<Option<Vec<_>>>()?;
        let values = (0..reader.u64()?).map(|_| reader.big()).collect::<Option<Vec<_>>>()?;
        let quotients = (0..reader.u64()?).map(|_| reader.big()).collect::<Option<Vec<_>>>()?;
        if values.len() != columns || quotients.len() != rows.len() || reader.at != bytes.len() {
            return None;
        }
        Some(Self {
            rows,
            widths,
            values,
            quotients,
        })
    }

    /// SHA-256 of the public part.
    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.public_bytes()).into()
    }

    /// Whether every row holds over the integers with these values.
    pub fn satisfied(&self) -> bool {
        let evaluate = |entries: &[Entry]| -> BigUint {
            entries
                .iter()
                .map(|entry| match entry.column {
                    Some(column) => &entry.coefficient * &self.values[column as usize],
                    None => entry.coefficient.clone(),
                })
                .sum()
        };
        self.rows.iter().zip(&self.quotients).all(|(row, quotient)| {
            evaluate(&row.a) * evaluate(&row.b) == evaluate(&row.c) + &row.modulus * quotient
        })
    }

}

/// The statement: a Mod-R1CS instance, identified by its digest.
#[derive(Clone, Debug)]
pub struct ModR1csStatement {
    instance: Arc<Instance>,
    digest: [u8; 32],
}

impl PartialEq for ModR1csStatement {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl Eq for ModR1csStatement {}

impl ModR1csStatement {
    pub fn new(instance: Instance) -> Self {
        let digest = instance.digest();
        Self {
            instance: Arc::new(instance),
            digest,
        }
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// No inputs: every value is a hint.
    pub fn input(&self) -> Vec<bool> {
        Vec::new()
    }

    /// The public bytes must be this instance's.
    pub fn matches_public_bytes(&self, public: &[u8]) -> bool {
        public == self.public_bytes()
    }
}

impl CircuitStatement for ModR1csStatement {
    fn domain(&self) -> &'static [u8] {
        b"mod-r1cs/v1"
    }

    /// The instance digest, `u64` rows, `u64` columns.
    fn public_bytes(&self) -> Vec<u8> {
        let mut bytes = self.digest.to_vec();
        put_u64(&mut bytes, self.instance.rows.len());
        put_u64(&mut bytes, self.instance.widths.len());
        bytes
    }

    fn input_bits(&self) -> usize {
        0
    }

    fn synthesize<CS: Circuit>(&self, cs: &mut CS, inputs: &[CS::Bool]) -> Result<(), Error> {
        if !inputs.is_empty() {
            return Err(Error::Input("a Mod-R1CS statement takes no inputs"));
        }
        let instance = &self.instance;
        // The values, hinted at their widths and lifted once.
        let lifted: Vec<CS::Z<LIMBS>> = instance
            .widths
            .iter()
            .zip(&instance.values)
            .map(|(&width, value)| {
                if width == 1 {
                    let bit = value.is_one();
                    let bits = cs.hint::<LIMBS, 1, 1, _>(move |_| HintResult::Ok(PackedBits::<1, 1>::from_u64(u64::from(bit))));
                    cs.bitz_unsigned::<LIMBS, 1, 1, 1>(&bits).0
                } else {
                    let words = words::<WIDE_WORDS>(value);
                    let bits = cs.hint::<LIMBS, WIDE, WIDE_WORDS, _>(move |_| HintResult::Ok(PackedBits::from_words(words)));
                    cs.bitz_unsigned::<LIMBS, WIDE, WIDE_WORDS, WIDE>(&bits).0
                }
            })
            .collect();
        for (row, quotient) in instance.rows.iter().zip(&instance.quotients) {
            let combine = |cs: &mut CS, entries: &[Entry]| -> CS::Z<LIMBS> {
                let mut sum = CS::Z::<LIMBS>::zero();
                for entry in entries {
                    let coefficient = coefficient::<CS>(&entry.coefficient);
                    sum += match entry.column {
                        Some(column) => lifted[column as usize].clone() * coefficient,
                        None => CS::Z::<LIMBS>::from(coefficient),
                    };
                }
                sum
            };
            let a = combine(cs, &row.a);
            let b = combine(cs, &row.b);
            let mut c = combine(cs, &row.c);
            if !row.modulus.is_zero() {
                // `t = m·q` on its own row, then `C + t`.
                let quotient_words = words::<WIDE_WORDS>(quotient);
                let quotient_bits = cs.hint::<LIMBS, WIDE, WIDE_WORDS, _>(move |_| {
                    HintResult::Ok(PackedBits::from_words(quotient_words))
                });
                let product_words = words::<PRODUCT_WORDS>(&(&row.modulus * quotient));
                let product_bits = cs.hint::<LIMBS, PRODUCT, PRODUCT_WORDS, _>(move |_| {
                    HintResult::Ok(PackedBits::from_words(product_words))
                });
                let (q, _) = cs.bitz_unsigned::<LIMBS, WIDE, WIDE_WORDS, WIDE>(&quotient_bits);
                let (t, _) = cs.bitz_unsigned::<LIMBS, PRODUCT, PRODUCT_WORDS, PRODUCT>(&product_bits);
                let m = CS::Z::<LIMBS>::from(coefficient::<CS>(&row.modulus));
                cs.assert_r1c::<LIMBS>(m, q, t.clone());
                c += t;
            }
            cs.assert_r1c::<LIMBS>(a, b, c);
        }
        Ok(())
    }
}

/// A coefficient from an integer, by Horner over its 64-bit words (ring
/// operations only, so every backend on either side builds the same).
fn coefficient<CS: Circuit>(value: &BigUint) -> CS::Coefficient<LIMBS> {
    let mut radix = CS::Coefficient::<LIMBS>::one();
    for _ in 0..64 {
        radix += radix.clone();
    }
    value
        .to_u64_digits()
        .iter()
        .rev()
        .fold(CS::Coefficient::<LIMBS>::zero(), |acc, word| {
            acc * radix.clone() + CS::Coefficient::<LIMBS>::from(*word)
        })
}

/// The little-endian words of a value, `N` of them.
fn words<const N: usize>(value: &BigUint) -> [u64; N] {
    let digits = value.to_u64_digits();
    assert!(digits.len() <= N, "a value wider than its width");
    let mut out = [0u64; N];
    out[..digits.len()].copy_from_slice(&digits);
    out
}
