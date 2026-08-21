//! SHA-256 circuits built from the backend-independent F2Z operations.
//!
//! Words are represented twice: as little-endian F2 bits for Boolean logic and
//! as lifted Z bits for integer linear combinations. This follows Freigen's
//! SHA-256 example, including its doubled `Ch` and `Maj` optimization.

use std::array;

use ark_ff::BigInteger64;

use crate::{BoolWitness, Circuit, Coefficient, HintResult, WitnessContext, ZWitness};

/// SHA-256 round constants from FIPS 180-4 section 4.2.2.
pub const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial chaining value from FIPS 180-4 section 5.3.3.
pub const INITIAL_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Number of bytes accepted by [`sha256_2kb_circuit`].
pub const SHA256_2KB_MESSAGE_BYTES: usize = 2048;

/// Number of input bits accepted by [`sha256_2kb_circuit`].
pub const SHA256_2KB_MESSAGE_BITS: usize = SHA256_2KB_MESSAGE_BYTES * 8;

/// Number of bits in a flattened compression input: one block and one state.
pub const COMPRESSION_INPUT_BITS: usize = 512 + 256;

/// An F2 word whose bits are stored least-significant first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Word<BW, const N: usize> {
    pub bits_le: [BW; N],
}

impl<BW, const N: usize> Word<BW, N>
where
    BW: BoolWitness,
{
    /// Constructs a word from little-endian bits.
    pub fn new(bits_le: [BW; N]) -> Self {
        Self { bits_le }
    }

    /// Constructs a constant word from the low `N` bits of `value`.
    pub fn constant(value: u64) -> Self {
        assert!(N <= u64::BITS as usize, "word is wider than u64");
        Self {
            bits_le: array::from_fn(|bit| BW::from((value >> bit) & 1 == 1)),
        }
    }

    /// Bitwise XOR, represented by addition over F2.
    pub fn xor(&self, rhs: &Self) -> Self {
        Self {
            bits_le: array::from_fn(|i| self.bits_le[i].clone() + rhs.bits_le[i].clone()),
        }
    }

    /// Bitwise XOR of three words.
    pub fn xor3(&self, second: &Self, third: &Self) -> Self {
        self.xor(second).xor(third)
    }

    /// Rotates the word right by `amount` bits.
    pub fn rotate_right(&self, amount: usize) -> Self {
        assert!(N != 0, "cannot rotate an empty word");
        let amount = amount % N;
        Self {
            bits_le: array::from_fn(|i| self.bits_le[(i + amount) % N].clone()),
        }
    }

    /// Shifts the word right, filling high bits with zero.
    pub fn shift_right(&self, amount: usize) -> Self {
        Self {
            bits_le: array::from_fn(|i| {
                i.checked_add(amount)
                    .filter(|&source| source < N)
                    .map_or_else(BW::zero, |source| self.bits_le[source].clone())
            }),
        }
    }
}

/// A word paired with the Z-side lift of each of its bits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UInt<ZW, BW, const N: usize> {
    pub word: Word<BW, N>,
    pub z_bits: [ZW; N],
}

impl<ZW, BW, const N: usize> UInt<ZW, BW, N> {
    /// Returns the Z linear combination represented by this unsigned integer.
    pub fn int_value<C>(&self) -> ZW
    where
        C: Coefficient,
        ZW: ZWitness<C>,
    {
        let mut result = ZW::zero();
        let mut power = C::one();
        for bit in &self.z_bits {
            result += bit.clone() * power.clone();
            power += power.clone();
        }
        result
    }
}

impl<ZW, BW, const N: usize> UInt<ZW, BW, N>
where
    BW: BoolWitness,
{
    fn from_word<C, CS>(circuit: &mut CS, word: Word<BW, N>) -> Self
    where
        C: Coefficient,
        ZW: ZWitness<C>,
        CS: Circuit<ZW, BW, C>,
    {
        let z_bits = array::from_fn(|i| circuit.f2z(word.bits_le[i].clone()));
        Self { word, z_bits }
    }

    fn constant<C>(value: u64) -> Self
    where
        C: Coefficient,
        ZW: ZWitness<C>,
    {
        let word = Word::constant(value);
        let z_bits = array::from_fn(|bit| {
            ZW::from(if (value >> bit) & 1 == 1 {
                C::one()
            } else {
                C::zero()
            })
        });
        Self { word, z_bits }
    }
}

type UInt32<ZW, BW> = UInt<ZW, BW, 32>;

#[derive(Clone)]
enum DoubledValue<BW> {
    Choice(Word<BW, 32>, Word<BW, 32>, Word<BW, 32>),
    Majority(Word<BW, 32>, Word<BW, 32>, Word<BW, 32>),
}

impl<BW> DoubledValue<BW>
where
    BW: BoolWitness,
{
    fn evaluate<ZW, C>(&self, context: &dyn WitnessContext<ZW, BW, C>) -> u64 {
        match self {
            Self::Choice(x, y, z) => {
                let x = evaluate_word(context, x) as u32;
                let y = evaluate_word(context, y) as u32;
                let z = evaluate_word(context, z) as u32;
                ((x & y) ^ (!x & z)).into()
            }
            Self::Majority(x, y, z) => {
                let x = evaluate_word(context, x) as u32;
                let y = evaluate_word(context, y) as u32;
                let z = evaluate_word(context, z) as u32;
                ((x & y) ^ (x & z) ^ (y & z)).into()
            }
        }
    }
}

fn coefficient<C: Coefficient>(value: u64) -> C {
    C::from_big_integer(BigInteger64::from(value))
}

fn evaluate_word<ZW, BW, C, const N: usize>(
    context: &dyn WitnessContext<ZW, BW, C>,
    word: &Word<BW, N>,
) -> u64 {
    assert!(N <= u64::BITS as usize, "word is wider than u64");
    word.bits_le
        .iter()
        .enumerate()
        .fold(0, |value, (bit, witness)| {
            value | (u64::from(context.eval_bool(witness)) << bit)
        })
}

fn assert_zero<CS, ZW, BW, C>(circuit: &mut CS, value: ZW)
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness,
    CS: Circuit<ZW, BW, C>,
{
    circuit.assert_r1c(ZW::zero(), ZW::zero(), value);
}

/// Decomposes a sum into `WIDTH` bits, constrains the decomposition, and
/// returns its low 32 bits.
fn sum_32<const WIDTH: usize, CS, ZW, BW, C>(
    circuit: &mut CS,
    terms: Vec<UInt32<ZW, BW>>,
) -> UInt32<ZW, BW>
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness + Send + Sync + 'static,
    CS: Circuit<ZW, BW, C>,
{
    assert!((32..=u64::BITS as usize).contains(&WIDTH));
    assert!(
        terms.len() <= 1usize << (WIDTH - 32),
        "sum does not fit its witnessed width"
    );
    let words: Vec<_> = terms.iter().map(|term| term.word.clone()).collect();
    let bits = circuit.hint(move |context| {
        let sum = words
            .iter()
            .map(|word| evaluate_word(context, word))
            .sum::<u64>();
        HintResult::Ok(array::from_fn(|bit| (sum >> bit) & 1 == 1))
    });
    let wide = UInt::<ZW, BW, WIDTH>::from_word(circuit, Word::new(bits));

    let input_sum = terms
        .into_iter()
        .fold(ZW::zero(), |sum, term| sum + term.int_value::<C>());
    assert_zero(circuit, input_sum - wide.int_value::<C>());

    UInt {
        word: Word::new(array::from_fn(|i| wide.word.bits_le[i].clone())),
        z_bits: array::from_fn(|i| wide.z_bits[i].clone()),
    }
}

/// Freigen's optimized sum: Z expressions for `2*Ch` and `2*Maj` are added to
/// twice the ordinary inputs, then the witnessed quotient by two is constrained.
fn sum_doubled_32<CS, ZW, BW, C>(
    circuit: &mut CS,
    ordinary: Vec<UInt32<ZW, BW>>,
    doubled: Vec<ZW>,
    doubled_values: Vec<DoubledValue<BW>>,
) -> UInt32<ZW, BW>
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness + Send + Sync + 'static,
    CS: Circuit<ZW, BW, C>,
{
    assert_eq!(doubled.len(), doubled_values.len());
    assert!(
        ordinary.len() + doubled_values.len() <= 7,
        "35 bits cannot hold this sum"
    );

    let words: Vec<_> = ordinary.iter().map(|term| term.word.clone()).collect();
    let bits = circuit.hint(move |context| {
        let ordinary = words
            .iter()
            .map(|word| evaluate_word(context, word))
            .sum::<u64>();
        let extras = doubled_values
            .iter()
            .map(|value| value.evaluate(context))
            .sum::<u64>();
        let half = ordinary + extras;
        HintResult::Ok(array::from_fn(|bit| (half >> bit) & 1 == 1))
    });
    let half = UInt::<ZW, BW, 35>::from_word(circuit, Word::new(bits));

    let ordinary = ordinary
        .into_iter()
        .fold(ZW::zero(), |sum, term| sum + term.int_value::<C>());
    let doubled = doubled.into_iter().fold(ZW::zero(), |sum, term| sum + term);
    let two = coefficient::<C>(2);
    let total = ordinary * two.clone() + doubled;
    assert_zero(circuit, total - half.int_value::<C>() * two);

    UInt {
        word: Word::new(array::from_fn(|i| half.word.bits_le[i].clone())),
        z_bits: array::from_fn(|i| half.z_bits[i].clone()),
    }
}

/// Returns a Z expression equal to twice SHA-256's `Ch(x, y, z)`.
fn choice_twice<CS, ZW, BW, C>(
    circuit: &mut CS,
    x: &UInt32<ZW, BW>,
    y: &UInt32<ZW, BW>,
    z: &UInt32<ZW, BW>,
) -> ZW
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness,
    CS: Circuit<ZW, BW, C>,
{
    let xy = UInt::from_word(circuit, x.word.xor(&y.word));
    let xz = UInt::from_word(circuit, x.word.xor(&z.word));
    y.int_value::<C>() + z.int_value::<C>() - xy.int_value::<C>() + xz.int_value::<C>()
}

/// Returns a Z expression equal to twice SHA-256's `Maj(x, y, z)`.
fn majority_twice<CS, ZW, BW, C>(
    circuit: &mut CS,
    x: &UInt32<ZW, BW>,
    y: &UInt32<ZW, BW>,
    z: &UInt32<ZW, BW>,
) -> ZW
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness,
    CS: Circuit<ZW, BW, C>,
{
    let xyz = UInt::from_word(circuit, x.word.xor3(&y.word, &z.word));
    x.int_value::<C>() + y.int_value::<C>() + z.int_value::<C>() - xyz.int_value::<C>()
}

/// Applies one SHA-256 compression to a 512-bit block and chaining value.
///
/// Both block and state words store their bits least-significant first. The
/// returned words retain both the Boolean and Z representations so callers may
/// inspect the generated relation or feed their Boolean representation into the
/// next compression.
pub fn compress<CS, ZW, BW, C>(
    circuit: &mut CS,
    block: [Word<BW, 32>; 16],
    state: [Word<BW, 32>; 8],
) -> [UInt32<ZW, BW>; 8]
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness + Send + Sync + 'static,
    CS: Circuit<ZW, BW, C>,
{
    let state = state.map(|word| UInt::from_word(circuit, word));
    let mut schedule: Vec<UInt32<ZW, BW>> = Vec::with_capacity(64);
    for word in block {
        schedule.push(UInt::from_word(circuit, word));
    }

    for i in 16..64 {
        let word_15 = &schedule[i - 15].word;
        let sigma_0 = word_15
            .rotate_right(7)
            .xor3(&word_15.rotate_right(18), &word_15.shift_right(3));
        let sigma_0 = UInt::from_word(circuit, sigma_0);

        let word_2 = &schedule[i - 2].word;
        let sigma_1 = word_2
            .rotate_right(17)
            .xor3(&word_2.rotate_right(19), &word_2.shift_right(10));
        let sigma_1 = UInt::from_word(circuit, sigma_1);

        schedule.push(sum_32::<34, _, _, _, _>(
            circuit,
            vec![
                schedule[i - 16].clone(),
                sigma_0,
                schedule[i - 7].clone(),
                sigma_1,
            ],
        ));
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state.clone();

    for i in 0..64 {
        let big_sigma_1_word = e
            .word
            .rotate_right(6)
            .xor3(&e.word.rotate_right(11), &e.word.rotate_right(25));
        let big_sigma_1 = UInt::from_word(circuit, big_sigma_1_word);
        let choice = choice_twice(circuit, &e, &f, &g);

        let big_sigma_0_word = a
            .word
            .rotate_right(2)
            .xor3(&a.word.rotate_right(13), &a.word.rotate_right(22));
        let big_sigma_0 = UInt::from_word(circuit, big_sigma_0_word);
        let majority = majority_twice(circuit, &a, &b, &c);

        let old_a = a;
        let old_b = b;
        let old_c = c;
        let old_d = d;
        let old_e = e;
        let old_f = f;
        let old_g = g;
        let old_h = h;
        let round_constant = UInt::constant::<C>(u64::from(ROUND_CONSTANTS[i]));

        h = old_g.clone();
        g = old_f.clone();
        f = old_e.clone();
        e = sum_doubled_32(
            circuit,
            vec![
                old_d,
                old_h.clone(),
                big_sigma_1.clone(),
                round_constant.clone(),
                schedule[i].clone(),
            ],
            vec![choice.clone()],
            vec![DoubledValue::Choice(
                old_e.word.clone(),
                old_f.word.clone(),
                old_g.word.clone(),
            )],
        );
        d = old_c.clone();
        c = old_b.clone();
        b = old_a.clone();
        a = sum_doubled_32(
            circuit,
            vec![
                old_h,
                big_sigma_1,
                round_constant,
                schedule[i].clone(),
                big_sigma_0,
            ],
            vec![choice, majority],
            vec![
                DoubledValue::Choice(old_e.word.clone(), old_f.word.clone(), old_g.word.clone()),
                DoubledValue::Majority(old_a.word.clone(), old_b.word.clone(), old_c.word.clone()),
            ],
        );
    }

    let working = [a, b, c, d, e, f, g, h];
    array::from_fn(|i| {
        sum_32::<33, _, _, _, _>(circuit, vec![state[i].clone(), working[i].clone()])
    })
}

/// Flattened counterpart of [`compress`], matching Freigen's `permCirc'`.
///
/// The first 512 input bits are 16 block words and the final 256 bits are eight
/// state words. Bits within every word, including the output words, are ordered
/// least-significant first.
pub fn compression_circuit<CS, ZW, BW, C>(
    circuit: &mut CS,
    input: &[BW; COMPRESSION_INPUT_BITS],
) -> [BW; 256]
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness + Send + Sync + 'static,
    CS: Circuit<ZW, BW, C>,
{
    let block =
        array::from_fn(|word| Word::new(array::from_fn(|bit| input[word * 32 + bit].clone())));
    let state = array::from_fn(|word| {
        Word::new(array::from_fn(|bit| input[512 + word * 32 + bit].clone()))
    });
    let output = compress(circuit, block, state);
    array::from_fn(|i| output[i / 32].word.bits_le[i % 32].clone())
}

/// Returns the standard SHA-256 initial state as constant Boolean words.
pub fn initial_state<BW>() -> [Word<BW, 32>; 8]
where
    BW: BoolWitness,
{
    INITIAL_STATE.map(|word| Word::constant(u64::from(word)))
}

/// Computes SHA-256 for an exactly 2 KiB message.
///
/// Input and output bits use conventional stream order: bytes are ordered from
/// first to last and bits within each byte are most-significant first.
pub fn sha256_2kb_circuit<CS, ZW, BW, C>(
    circuit: &mut CS,
    message: &[BW; SHA256_2KB_MESSAGE_BITS],
) -> [BW; 256]
where
    C: Coefficient,
    ZW: ZWitness<C>,
    BW: BoolWitness + Send + Sync + 'static,
    CS: Circuit<ZW, BW, C>,
{
    let mut state = initial_state();

    for block_index in 0..32 {
        let block = array::from_fn(|word| {
            Word::new(array::from_fn(|bit| {
                let stream_bit = block_index * 512 + word * 32 + (31 - bit);
                message[stream_bit].clone()
            }))
        });
        state = compress(circuit, block, state).map(|word| word.word);
    }

    let padding_values: [u32; 16] = [
        0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00004000,
    ];
    let padding = padding_values.map(|word| Word::constant(u64::from(word)));
    let digest = compress(circuit, padding, state);

    array::from_fn(|i| digest[i / 32].word.bits_le[31 - (i % 32)].clone())
}

#[cfg(test)]
mod tests {
    use std::iter::Sum;
    use std::ops::{Add, AddAssign};

    use num_traits::Zero;

    use super::*;
    use crate::HintError;
    use crate::stats::{Dummy, LeanStats, Stats};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Bit(bool);

    impl From<bool> for Bit {
        fn from(value: bool) -> Self {
            Self(value)
        }
    }

    impl Zero for Bit {
        fn zero() -> Self {
            Self(false)
        }

        fn is_zero(&self) -> bool {
            !self.0
        }
    }

    impl Add for Bit {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 != rhs.0)
        }
    }

    impl AddAssign for Bit {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sum for Bit {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), Add::add)
        }
    }

    struct Values;

    impl WitnessContext<i128, Bit, i128> for Values {
        fn eval_z(&self, witness: &i128) -> i128 {
            *witness
        }

        fn eval_bool(&self, witness: &Bit) -> bool {
            witness.0
        }
    }

    #[derive(Default)]
    struct EvaluatingCircuit {
        assertions: usize,
    }

    impl Circuit<i128, Bit, i128> for EvaluatingCircuit {
        fn hint<const N: usize, H>(&mut self, hint: H) -> [Bit; N]
        where
            H: Fn(&dyn WitnessContext<i128, Bit, i128>) -> Result<[bool; N], HintError>
                + Send
                + Sync
                + 'static,
        {
            hint(&Values)
                .expect("SHA-256 hint should be defined")
                .map(Bit)
        }

        fn f2z(&mut self, value: Bit) -> i128 {
            i128::from(value.0)
        }

        fn assert_r1c(&mut self, a: i128, b: i128, c: i128) {
            self.assertions += 1;
            assert_eq!(a * b, c, "unsatisfied SHA-256 constraint");
        }
    }

    fn value(word: &Word<Bit, 32>) -> u32 {
        word.bits_le
            .iter()
            .enumerate()
            .fold(0, |value, (bit, witness)| {
                value | (u32::from(witness.0) << bit)
            })
    }

    #[test]
    fn compression_matches_the_fips_abc_vector() {
        let block_values: [u32; 16] = [
            0x61626380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000018,
        ];
        let block = block_values.map(|word| Word::constant(u64::from(word)));
        let mut circuit = EvaluatingCircuit::default();

        let output = compress(&mut circuit, block, initial_state());
        let output = output.map(|word| value(&word.word));

        assert_eq!(
            output,
            [
                0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
                0xf20015ad,
            ]
        );
        assert_eq!(circuit.assertions, 184);
    }

    #[test]
    fn fixed_2kb_circuit_matches_sha256_stream_order() {
        let message: [Bit; SHA256_2KB_MESSAGE_BITS] = array::from_fn(|bit| {
            let byte = (bit / 8) as u8;
            Bit(byte & (1 << (7 - bit % 8)) != 0)
        });
        let mut circuit = EvaluatingCircuit::default();

        let digest_bits = sha256_2kb_circuit(&mut circuit, &message);
        let digest: [u8; 32] = array::from_fn(|byte| {
            (0..8).fold(0, |value, bit| {
                value | (u8::from(digest_bits[byte * 8 + bit].0) << (7 - bit))
            })
        });

        assert_eq!(
            digest,
            [
                0x10, 0xfc, 0x3c, 0x51, 0xa1, 0x52, 0xe9, 0x0e, 0x5b, 0x90, 0x31, 0x9b, 0x60, 0x1d,
                0x92, 0xcc, 0xf3, 0x72, 0x90, 0xef, 0x53, 0xc3, 0x5f, 0xf9, 0x25, 0x07, 0x68, 0x7d,
                0x8a, 0x91, 0x1a, 0x08,
            ]
        );
        assert_eq!(circuit.assertions, 33 * 184);
    }

    #[test]
    fn compression_layout_exactly_matches_freigen() {
        let mut stats = Stats::new(COMPRESSION_INPUT_BITS);

        let _ = compression_circuit(&mut stats, &[Dummy; COMPRESSION_INPUT_BITS]);

        assert_eq!(
            stats,
            Stats {
                witnesses: 7_144,
                f2z_calls: 20_456,
                constraints: 184,
            }
        );
        assert_eq!(
            stats.lean_stats(),
            LeanStats {
                m_rows: 20_457,
                m_cols: 7_145,
                r1cs_rows: 184,
            }
        );
    }
}
