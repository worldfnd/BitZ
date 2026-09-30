//! An eager witness-generation evaluator.

use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use num_traits::{One, Zero};

use crate::matrix_products::IntegerProducts;
use crate::{BoolWitness, Circuit, HintResult, PackedBits, WitnessContext};

// TODO(alex): Make into a proper ring

/// A signed two's-complement integer with a compile-time capacity.
///
/// Arithmetic wraps modulo `2^(64 * LIMBS)`. A witness generator must be
/// instantiated with enough limbs for every signed intermediate in its
/// circuit. This makes arithmetic allocation-free and non-panicking, allowing
/// constraint-only calculations to be eliminated by the optimizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Z<const LIMBS: usize = 1> {
    words: [u64; LIMBS],
}

impl<const LIMBS: usize> Z<LIMBS> {
    /// Constructs a nonnegative integer from little-endian machine words.
    pub fn from_le_words(words: &[u64]) -> Self {
        let mut output = [0; LIMBS];
        for (output, input) in output.iter_mut().zip(words) {
            *output = *input;
        }
        Self { words: output }
    }

    /// Constructs a nonnegative integer from little-endian bits.
    pub fn from_le_bits(bits: &[bool]) -> Self {
        let mut words = [0; LIMBS];
        for (index, set) in bits.iter().take(LIMBS * 64).enumerate() {
            words[index / 64] |= u64::from(*set) << (index % 64);
        }
        Self { words }
    }

    /// Constructs a nonnegative integer from packed little-endian bits.
    ///
    /// Values wider than the declared capacity are truncated. The caller's
    /// circuit-wide bound is responsible for ruling that out.
    pub fn from_packed_bits<const N: usize, const M: usize>(bits: &PackedBits<N, M>) -> Self {
        let mut words = [0; LIMBS];
        for (output, input) in words.iter_mut().zip(bits.words()) {
            *output = *input;
        }
        Self { words }
    }

    fn from_packed_prefix<const N: usize, const M: usize>(
        bits: &PackedBits<N, M>,
        width: usize,
    ) -> Self {
        assert!(width <= N);
        let mut value = Self::from_packed_bits(bits);
        let retained_words = width.div_ceil(64).min(LIMBS);
        for word in &mut value.words[retained_words..] {
            *word = 0;
        }
        if !width.is_multiple_of(64) && retained_words != 0 {
            value.words[retained_words - 1] &= (1_u64 << (width % 64)) - 1;
        }
        value
    }

    /// Little-endian two's-complement storage words.
    pub fn words(&self) -> &[u64; LIMBS] {
        &self.words
    }

    /// Whether the fixed-width value is negative.
    pub fn is_negative(&self) -> bool {
        self.words.last().is_some_and(|word| word >> 63 == 1)
    }

    /// Sign-extends this value into a wider fixed representation.
    pub fn sign_extend<const TO_LIMBS: usize>(self) -> Z<TO_LIMBS> {
        assert!(TO_LIMBS >= LIMBS, "cannot sign-extend into fewer limbs");
        let mut words = [if self.is_negative() { u64::MAX } else { 0 }; TO_LIMBS];
        for (output, input) in words.iter_mut().zip(self.words) {
            *output = input;
        }
        Z { words }
    }
}

impl<const LIMBS: usize> Default for Z<LIMBS> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const LIMBS: usize> From<i128> for Z<LIMBS> {
    fn from(value: i128) -> Self {
        let mut words = [if value < 0 { u64::MAX } else { 0 }; LIMBS];
        if LIMBS != 0 {
            words[0] = value as u64;
        }
        if LIMBS > 1 {
            words[1] = (value >> 64) as u64;
        }
        Self { words }
    }
}

impl<const LIMBS: usize> From<u64> for Z<LIMBS> {
    fn from(value: u64) -> Self {
        let mut words = [0; LIMBS];
        if LIMBS != 0 {
            words[0] = value;
        }
        Self { words }
    }
}

impl<const LIMBS: usize> Zero for Z<LIMBS> {
    fn zero() -> Self {
        Self { words: [0; LIMBS] }
    }

    fn is_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }
}

impl<const LIMBS: usize> One for Z<LIMBS> {
    fn one() -> Self {
        Self::from(1_u64)
    }
}

impl<const LIMBS: usize> Add for Z<LIMBS> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut words = [0; LIMBS];
        let mut carry = false;
        for (index, output) in words.iter_mut().enumerate() {
            let (sum, first_carry) = self.words[index].overflowing_add(rhs.words[index]);
            let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
            *output = sum;
            carry = first_carry || second_carry;
        }
        Self { words }
    }
}

impl<const LIMBS: usize> AddAssign for Z<LIMBS> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const LIMBS: usize> Sub for Z<LIMBS> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut words = [0; LIMBS];
        let mut borrow = false;
        for (index, output) in words.iter_mut().enumerate() {
            let (difference, first_borrow) = self.words[index].overflowing_sub(rhs.words[index]);
            let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
            *output = difference;
            borrow = first_borrow || second_borrow;
        }
        Self { words }
    }
}

impl<const LIMBS: usize> SubAssign for Z<LIMBS> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const LIMBS: usize> Mul for Z<LIMBS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut words = [0; LIMBS];
        for left in 0..LIMBS {
            let mut carry = 0_u128;
            for right in 0..(LIMBS - left) {
                let output = left + right;
                let product = u128::from(self.words[left]) * u128::from(rhs.words[right]);
                let total = u128::from(words[output]) + product + carry;
                words[output] = total as u64;
                carry = total >> 64;
            }
        }
        Self { words }
    }
}

impl<const LIMBS: usize> Neg for Z<LIMBS> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl<const LIMBS: usize> Sum for Z<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

struct ValueContext;

impl<const LIMBS: usize> WitnessContext<Z<LIMBS>, bool, Z<LIMBS>> for ValueContext {
    fn eval_z(&self, witness: &Z<LIMBS>) -> Z<LIMBS> {
        *witness
    }

    fn eval_z_words<'a>(&self, witness: &'a Z<LIMBS>) -> Option<&'a [u64]> {
        Some(witness.words())
    }

    fn eval_bool(&self, witness: &bool) -> bool {
        *witness
    }
}

/// A Boolean witness packed least-significant-bit first into `u64` words.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PackedWitness {
    words: Vec<u64>,
    bit_len: usize,
}

impl PackedWitness {
    /// Packs Boolean values least-significant-bit first into witness storage.
    pub fn from_bits(bits: &[bool]) -> Self {
        let mut witness = Self::with_capacity(bits.len());
        witness.extend(bits);
        witness
    }

    fn with_capacity(bit_capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(bit_capacity.div_ceil(64)),
            bit_len: 0,
        }
    }

    fn extend(&mut self, bits: &[bool]) {
        for chunk in bits.chunks(64) {
            let value = chunk
                .iter()
                .enumerate()
                .fold(0_u64, |value, (bit, set)| value | (u64::from(*set) << bit));
            self.append_word(value, chunk.len());
        }
    }

    fn extend_packed<const N: usize, const M: usize>(&mut self, bits: &PackedBits<N, M>) {
        for (index, word) in bits.words().iter().enumerate() {
            self.append_word(*word, (N - index * 64).min(64));
        }
    }

    fn extend_packed_words(&mut self, words: &[u64], bit_len: usize) {
        assert!(bit_len <= words.len().saturating_mul(64));
        for (index, word) in words.iter().take(bit_len.div_ceil(64)).enumerate() {
            self.append_word(*word, (bit_len - index * 64).min(64));
        }
    }

    fn append_word(&mut self, value: u64, width: usize) {
        debug_assert!(width <= 64);
        if width == 0 {
            return;
        }
        let value = if width == 64 {
            value
        } else {
            value & ((1_u64 << width) - 1)
        };
        let offset = self.bit_len % 64;
        if offset == 0 {
            self.words.push(value);
        } else {
            let last = self
                .words
                .last_mut()
                .expect("partial witness word must exist");
            *last |= value << offset;
            if offset + width > 64 {
                self.words.push(value >> (64 - offset));
            }
        }
        self.bit_len += width;
    }

    /// Packed storage words.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Number of meaningful witness bits.
    pub const fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Reads one witness bit.
    pub fn bit(&self, index: usize) -> bool {
        assert!(index < self.bit_len);
        self.words[index / 64] >> (index % 64) & 1 == 1
    }
}

/// Executes hints eagerly and accumulates the resulting packed Boolean witness.
///
/// The runner supports every [`Z<LIMBS>`] width simultaneously; each gadget
/// chooses its own width through its [`Circuit`] instantiation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Witgen {
    witness: PackedWitness,
    integer_witness: PackedWitness,
}

/// Executes hints and accumulates only the Boolean witness `w`.
///
/// Z-side values are still evaluated because later hints depend on them, but
/// the `M * w` image and rank-1 products are deliberately not retained. This
/// is the baseline witness-generation backend used to measure the cost of
/// witness generation independently from output recording.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WitnessOnly {
    witness: PackedWitness,
}

impl WitnessOnly {
    /// Starts with already assigned Boolean inputs and reserves the complete
    /// Boolean witness capacity.
    pub fn with_inputs_and_capacity(inputs: &[bool], witness_capacity: usize) -> Self {
        assert!(witness_capacity >= inputs.len());
        let mut witness = PackedWitness::with_capacity(witness_capacity);
        witness.extend(inputs);
        Self { witness }
    }

    /// Packed Boolean witness accumulated so far.
    pub const fn witness(&self) -> &PackedWitness {
        &self.witness
    }

    /// Consumes the evaluator into the packed Boolean witness.
    pub fn into_witness(self) -> PackedWitness {
        self.witness
    }
}

impl Circuit for WitnessOnly {
    type Bool = bool;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = Z<LIMBS>;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Z<LIMBS> {
        Z::from_le_words(words)
    }

    fn xor(&mut self, lhs: bool, rhs: bool) -> bool {
        lhs ^ rhs
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        hint: H,
    ) -> PackedBits<N, M>
    where
        H: Fn(&dyn WitnessContext<Z<LIMBS>, bool, Z<LIMBS>>) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        let bits =
            hint(&ValueContext).unwrap_or_else(|error| panic!("witness hint failed: {error}"));
        self.witness.extend_packed(&bits);
        bits
    }

    fn bitz<const LIMBS: usize>(&mut self, value: bool) -> Z<LIMBS> {
        if value { Z::one() } else { Z::zero() }
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        bits_le: &<bool as BoolWitness>::Repr<N, M>,
    ) -> (Z<LIMBS>, Z<LIMBS>) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        (
            Z::from_packed_bits(bits_le),
            Z::from_packed_prefix(bits_le, LOW),
        )
    }

    fn assert_r1c<const LIMBS: usize>(&mut self, _: Z<LIMBS>, _: Z<LIMBS>, _: Z<LIMBS>) {}

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: Z<FROM_LIMBS>,
    ) -> Z<TO_LIMBS> {
        value.sign_extend()
    }
}

impl Witgen {
    fn initial_integer_witness() -> PackedWitness {
        let mut witness = PackedWitness::with_capacity(1);
        witness.extend(&[true]);
        witness
    }

    fn seeded(inputs: &[bool], witness_capacity: usize) -> Self {
        assert!(witness_capacity >= inputs.len());
        let mut witness = PackedWitness::with_capacity(witness_capacity);
        witness.extend(inputs);
        Self {
            witness,
            integer_witness: Self::initial_integer_witness(),
        }
    }

    /// Witness bits accumulated so far.
    pub fn witness(&self) -> &PackedWitness {
        &self.witness
    }

    /// Packed values returned by logical `BitZ` calls.
    ///
    /// Entry zero is the implicit integer constant one. Every later entry is
    /// the 0/1 result of one `BitZ`, in circuit order, so this is exactly `M * w`.
    pub fn integer_witness(&self) -> &PackedWitness {
        &self.integer_witness
    }

    /// Consumes the evaluator and returns its packed witness.
    pub fn into_witness(self) -> PackedWitness {
        self.witness
    }

    /// Consumes the evaluator and returns both the Boolean witness and `M * w`.
    pub fn into_witnesses(self) -> (PackedWitness, PackedWitness) {
        (self.witness, self.integer_witness)
    }

    /// Constructs the fast witness-generation evaluator.
    pub fn new() -> Self {
        Self {
            witness: PackedWitness {
                words: Vec::new(),
                bit_len: 0,
            },
            integer_witness: Self::initial_integer_witness(),
        }
    }

    /// Starts with already assigned Boolean input witnesses.
    pub fn with_inputs(inputs: &[bool]) -> Self {
        Self::seeded(inputs, inputs.len())
    }

    /// Starts with inputs and reserves space for `witness_capacity` total bits.
    pub fn with_inputs_and_capacity(inputs: &[bool], witness_capacity: usize) -> Self {
        Self::seeded(inputs, witness_capacity)
    }

    /// Starts from packed input words and reserves the total witness size.
    pub fn with_packed_inputs_and_capacity(
        input_words: &[u64],
        input_bits: usize,
        witness_capacity: usize,
    ) -> Self {
        assert!(witness_capacity >= input_bits);
        let mut witness = PackedWitness::with_capacity(witness_capacity);
        witness.extend_packed_words(input_words, input_bits);
        Self {
            witness,
            integer_witness: Self::initial_integer_witness(),
        }
    }
}

impl Default for Witgen {
    fn default() -> Self {
        Self::new()
    }
}

impl Circuit for Witgen {
    type Bool = bool;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = Z<LIMBS>;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Z<LIMBS> {
        Z::from_le_words(words)
    }

    fn xor(&mut self, lhs: bool, rhs: bool) -> bool {
        lhs ^ rhs
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        hint: H,
    ) -> PackedBits<N, M>
    where
        H: Fn(&dyn WitnessContext<Z<LIMBS>, bool, Z<LIMBS>>) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        let bits =
            hint(&ValueContext).unwrap_or_else(|error| panic!("witness hint failed: {error}"));
        self.witness.extend_packed(&bits);
        bits
    }

    fn bitz<const LIMBS: usize>(&mut self, value: bool) -> Z<LIMBS> {
        self.integer_witness.extend(&[value]);
        if value { Z::one() } else { Z::zero() }
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        bits_le: &<bool as BoolWitness>::Repr<N, M>,
    ) -> (Z<LIMBS>, Z<LIMBS>) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        self.integer_witness.extend_packed(bits_le);
        (
            Z::from_packed_bits(bits_le),
            Z::from_packed_prefix(bits_le, LOW),
        )
    }

    fn assert_r1c<const LIMBS: usize>(&mut self, _: Z<LIMBS>, _: Z<LIMBS>, _: Z<LIMBS>) {}

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: Z<FROM_LIMBS>,
    ) -> Z<TO_LIMBS> {
        value.sign_extend()
    }
}

/// Generates `w` and `M * w` while retaining exact integer R1CS inputs.
///
/// Unlike [`Witgen`], this makes constraint-side integer arithmetic observable
/// during the first pass. The recorded values can subsequently be batch
/// reduced with [`IntegerProducts::reduce_parallel`], avoiding a second circuit
/// replay at the cost of a slower witness-generation pass.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProductWitgen {
    witgen: Witgen,
    products: IntegerProducts,
}

impl ProductWitgen {
    /// Starts with already assigned Boolean input witnesses.
    pub fn with_inputs(inputs: &[bool]) -> Self {
        Self {
            witgen: Witgen::with_inputs(inputs),
            products: IntegerProducts::default(),
        }
    }

    /// Starts with inputs and reserves space for the complete Boolean witness.
    pub fn with_inputs_and_capacity(inputs: &[bool], witness_capacity: usize) -> Self {
        Self {
            witgen: Witgen::with_inputs_and_capacity(inputs, witness_capacity),
            products: IntegerProducts::default(),
        }
    }

    /// Starts from packed inputs and reserves the complete Boolean witness.
    pub fn with_packed_inputs_and_capacity(
        input_words: &[u64],
        input_bits: usize,
        witness_capacity: usize,
    ) -> Self {
        Self {
            witgen: Witgen::with_packed_inputs_and_capacity(
                input_words,
                input_bits,
                witness_capacity,
            ),
            products: IntegerProducts::default(),
        }
    }

    /// Packed Boolean witness accumulated so far.
    pub fn witness(&self) -> &PackedWitness {
        self.witgen.witness()
    }

    /// Packed `M * w` accumulated so far.
    pub fn integer_witness(&self) -> &PackedWitness {
        self.witgen.integer_witness()
    }

    /// Exact integer constraint inputs accumulated so far.
    pub const fn products(&self) -> &IntegerProducts {
        &self.products
    }

    /// Consumes the runner into `w`, `M * w`, and exact matrix products.
    pub fn into_parts(self) -> (PackedWitness, PackedWitness, IntegerProducts) {
        let (witness, integer_witness) = self.witgen.into_witnesses();
        (witness, integer_witness, self.products)
    }
}

impl Circuit for ProductWitgen {
    type Bool = bool;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = Z<LIMBS>;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Z<LIMBS> {
        Z::from_le_words(words)
    }

    fn xor(&mut self, lhs: bool, rhs: bool) -> bool {
        lhs ^ rhs
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        hint: H,
    ) -> PackedBits<N, M>
    where
        H: Fn(&dyn WitnessContext<Z<LIMBS>, bool, Z<LIMBS>>) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        self.witgen.hint(hint)
    }

    fn bitz<const LIMBS: usize>(&mut self, value: bool) -> Z<LIMBS> {
        self.witgen.bitz(value)
    }

    fn bitz_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        bits_le: &<bool as BoolWitness>::Repr<N, M>,
    ) -> (Z<LIMBS>, Z<LIMBS>) {
        self.witgen.bitz_unsigned::<LIMBS, N, M, LOW>(bits_le)
    }

    fn assert_r1c<const LIMBS: usize>(&mut self, a: Z<LIMBS>, b: Z<LIMBS>, c: Z<LIMBS>) {
        self.products.push(a, b, c);
    }

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: Z<FROM_LIMBS>,
    ) -> Z<TO_LIMBS> {
        value.sign_extend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_bits_are_arbitrary_width() {
        const WIDTH: usize = 197;
        let expected = |bit: usize| bit % 5 == 1 || bit % 7 == 3;
        let bits = PackedBits::<WIDTH, 4>::from_fn(expected);

        assert_eq!(bits.words().len(), WIDTH.div_ceil(64));
        assert!((0..WIDTH).all(|index| bits.bit(index) == expected(index)));
        assert_eq!(
            PackedBits::<WIDTH, 4>::from_u64(u64::MAX).words()[1..],
            [0, 0, 0]
        );

        let rotated =
            <PackedBits<WIDTH, 4> as crate::BoolRepresentation<bool, WIDTH, 4>>::rotate_right(
                &bits, 73,
            );
        assert!((0..WIDTH).all(|index| rotated.bit(index) == expected((index + 73) % WIDTH)));

        let shifted =
            <PackedBits<WIDTH, 4> as crate::BoolRepresentation<bool, WIDTH, 4>>::shift_right(
                &bits, 81,
            );
        assert!(
            (0..WIDTH)
                .all(|index| shifted.bit(index) == (index + 81 < WIDTH && expected(index + 81)))
        );

        let sliced = <PackedBits<WIDTH, 4> as crate::BoolRepresentation<bool, WIDTH, 4>>::slice::<
            65,
            2,
        >(&bits, 63);
        assert!((0..65).all(|index| sliced.bit(index) == expected(index + 63)));

        let integer = Z::<4>::from_packed_bits(&bits);
        assert_eq!(integer.words(), bits.words());
    }

    #[test]
    #[should_panic(expected = "incorrect packed limb count")]
    fn packed_bits_reject_an_incorrect_limb_count() {
        let _ = PackedBits::<65, 1>::zero();
    }

    #[test]
    fn packed_witness_preserves_bits_across_word_boundaries() {
        let expected: Vec<bool> = (0..197).map(|bit| bit % 5 == 1 || bit % 7 == 3).collect();
        let mut witness = PackedWitness::with_capacity(expected.len());

        witness.extend(&expected[..17]);
        witness.extend(&expected[17..81]);
        witness.extend(&expected[81..]);

        assert_eq!(witness.bit_len(), expected.len());
        assert_eq!(witness.words().len(), expected.len().div_ceil(64));
        assert!(
            expected
                .iter()
                .enumerate()
                .all(|(index, expected)| witness.bit(index) == *expected)
        );
    }

    #[test]
    fn hints_evaluate_plain_bits_and_fixed_integers() {
        let integer = Z::<1>::from(13_i128);
        let captured = integer;
        let mut witgen = Witgen::new();

        let bits = <Witgen as Circuit>::hint::<1, 4, 1, _>(&mut witgen, move |context| {
            let integer = context.eval_z(&captured).words()[0];
            Ok(PackedBits::<4, 1>::from_fn(|bit| integer >> bit & 1 == 1))
        });

        assert_eq!(
            bits,
            PackedBits::<4, 1>::from_array([true, false, true, true])
        );
        assert_eq!(witgen.bitz::<1>(true), Z::<1>::one());
    }

    #[test]
    fn records_the_integer_witness_in_logical_bitz_order() {
        let mut witgen = Witgen::new();
        let _ = witgen.bitz::<1>(true);
        let _ = witgen.bitz::<8>(false);
        let bits = PackedBits::<4, 1>::from_array([false, true, true, false]);
        let _: (Z<1>, Z<1>) = witgen.bitz_unsigned::<1, 4, 1, 2>(&bits);

        let expected = [true, true, false, false, true, true, false];
        assert_eq!(witgen.integer_witness().bit_len(), expected.len());
        assert!(
            expected
                .iter()
                .enumerate()
                .all(|(index, expected)| witgen.integer_witness().bit(index) == *expected)
        );
    }

    #[test]
    fn one_runner_supports_local_widths_and_explicit_sign_extension() {
        let mut witgen = Witgen::new();
        let small: Z<1> = witgen.bitz::<1>(true);
        let large: Z<128> = witgen.sign_extend_z::<1, 128>(-small);
        let rsa_bit: Z<128> = witgen.bitz::<128>(false);

        assert_eq!(large.words(), &[u64::MAX; 128]);
        assert_eq!(rsa_bit.words(), &[0; 128]);
    }

    #[test]
    fn fixed_integers_wrap_at_the_declared_width() {
        assert_eq!(std::mem::size_of::<Z<4>>(), 4 * std::mem::size_of::<u64>());
        assert!(!std::mem::needs_drop::<Z<4>>());
        assert_eq!(
            (Z::<2>::from(i128::MAX) + Z::one()).words(),
            &[0, 1_u64 << 63]
        );
        assert_eq!(
            Z::<4>::from_le_bits(&[true; 200]).words(),
            &[u64::MAX, u64::MAX, u64::MAX, 0xff]
        );
    }

    #[test]
    fn fixed_integer_arithmetic_carries_across_limbs() {
        let left = Z::<3> {
            words: [u64::MAX, 4, 5],
        };
        let right = Z::<3> { words: [2, 8, 9] };

        assert_eq!((left + right).words(), &[1, 13, 14]);
        assert_eq!(((left + right) - right), left);
        assert_eq!(((left - right) + right), left);
        assert_eq!(
            (Z::<3> { words: [3, 4, 5] } * Z::<3> { words: [7, 8, 9] }).words(),
            &[21, 52, 94]
        );
    }
}
