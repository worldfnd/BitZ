//! Exact matrix-product recording and parallel runtime-field reduction.
//!
//! [`crate::witgen::ProductWitgen`] evaluates each rank-1 input over the
//! gadget-local fixed integer type and records dense integer `A(Mw)`, `B(Mw)`,
//! and `C(Mw)` vectors during witness generation. This module subsequently
//! reduces those vectors modulo a runtime modulus. Large batches use Rayon;
//! small batches stay sequential to avoid scheduling overhead.

use std::array;
use std::cmp::Ordering;

use num_bigint::BigUint;
use num_traits::{One, Zero};
use rayon::prelude::*;

use crate::witgen::Z as Integer;

/// A runtime modulus with a compile-time limb count.
#[derive(Debug, Eq, PartialEq)]
pub struct RuntimeModulus<const PRIME_LIMBS: usize> {
    modulus: [u64; PRIME_LIMBS],
}

impl<const PRIME_LIMBS: usize> RuntimeModulus<PRIME_LIMBS> {
    /// Validates and stores a runtime modulus.
    pub fn new(modulus: BigUint) -> Result<Self, &'static str> {
        if PRIME_LIMBS == 0 {
            return Err("a runtime field needs at least one limb");
        }
        if modulus <= BigUint::one() {
            return Err("the modulus must be greater than one");
        }
        if modulus.bits() > (PRIME_LIMBS as u64) * 64 {
            return Err("the modulus does not fit the selected limb count");
        }
        Ok(Self {
            modulus: biguint_words(&modulus),
        })
    }

    /// The runtime modulus in canonical little-endian limbs.
    pub const fn modulus_words(&self) -> &[u64; PRIME_LIMBS] {
        &self.modulus
    }

    /// Converts the modulus back to a `BigUint`.
    pub fn modulus(&self) -> BigUint {
        BigUint::from_bytes_le(
            &self
                .modulus
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn reduce(&self, value: &StoredInteger) -> [u64; PRIME_LIMBS] {
        if value.words.is_empty() {
            return [0; PRIME_LIMBS];
        }

        let negative = value.is_negative();
        let magnitude;
        let words = if negative {
            magnitude = twos_complement_magnitude(&value.words);
            magnitude.as_slice()
        } else {
            value.words.as_ref()
        };

        let mut reduced = [0; PRIME_LIMBS];
        let mut one = [0; PRIME_LIMBS];
        one[0] = 1;
        for &word in words.iter().rev() {
            for bit in (0..64).rev() {
                reduced = add_mod_words(reduced, &reduced, &self.modulus);
                if word >> bit & 1 == 1 {
                    reduced = add_mod_words(reduced, &one, &self.modulus);
                }
            }
        }

        if negative && reduced.iter().any(|word| *word != 0) {
            subtract_words(self.modulus, &reduced).0
        } else {
            reduced
        }
    }
}

fn twos_complement_magnitude(words: &[u64]) -> Vec<u64> {
    let mut magnitude = Vec::with_capacity(words.len());
    let mut carry = true;
    for word in words {
        let (word, overflow) = (!word).overflowing_add(u64::from(carry));
        magnitude.push(word);
        carry = overflow;
    }
    while magnitude.last() == Some(&0) {
        magnitude.pop();
    }
    magnitude
}

fn biguint_words<const LIMBS: usize>(value: &BigUint) -> [u64; LIMBS] {
    let digits = value.to_u64_digits();
    array::from_fn(|index| digits.get(index).copied().unwrap_or(0))
}

fn compare_words<const LIMBS: usize>(left: &[u64; LIMBS], right: &[u64; LIMBS]) -> Ordering {
    for index in (0..LIMBS).rev() {
        match left[index].cmp(&right[index]) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    Ordering::Equal
}

fn subtract_words<const LIMBS: usize>(
    left: [u64; LIMBS],
    right: &[u64; LIMBS],
) -> ([u64; LIMBS], bool) {
    let mut output = [0; LIMBS];
    let mut borrow = false;
    for index in 0..LIMBS {
        let (difference, first) = left[index].overflowing_sub(right[index]);
        let (difference, second) = difference.overflowing_sub(u64::from(borrow));
        output[index] = difference;
        borrow = first || second;
    }
    (output, borrow)
}

fn add_words<const LIMBS: usize>(left: [u64; LIMBS], right: &[u64; LIMBS]) -> ([u64; LIMBS], bool) {
    let mut output = [0; LIMBS];
    let mut carry = false;
    for index in 0..LIMBS {
        let (sum, first) = left[index].overflowing_add(right[index]);
        let (sum, second) = sum.overflowing_add(u64::from(carry));
        output[index] = sum;
        carry = first || second;
    }
    (output, carry)
}

fn add_mod_words<const LIMBS: usize>(
    left: [u64; LIMBS],
    right: &[u64; LIMBS],
    modulus: &[u64; LIMBS],
) -> [u64; LIMBS] {
    let (sum, carry) = add_words(left, right);
    if carry || compare_words(&sum, modulus) != Ordering::Less {
        subtract_words(sum, modulus).0
    } else {
        sum
    }
}

/// A dense vector of canonical runtime-field elements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModularVector<const PRIME_LIMBS: usize> {
    values: Vec<[u64; PRIME_LIMBS]>,
}

impl<const PRIME_LIMBS: usize> ModularVector<PRIME_LIMBS> {
    /// Number of materialized elements.
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the vector contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Dense canonical field elements in row order.
    pub fn values(&self) -> &[[u64; PRIME_LIMBS]] {
        &self.values
    }

    /// Returns one canonical field element.
    pub fn get(&self, index: usize) -> [u64; PRIME_LIMBS] {
        self.values[index]
    }
}

/// Dense runtime-field `A(Mw)`, `B(Mw)`, and `C(Mw)` vectors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixProducts<const PRIME_LIMBS: usize> {
    pub a_mw: ModularVector<PRIME_LIMBS>,
    pub b_mw: ModularVector<PRIME_LIMBS>,
    pub c_mw: ModularVector<PRIME_LIMBS>,
}

/// An exact signed two's-complement integer with redundant sign limbs removed.
///
/// Every matrix row has a materialized [`StoredInteger`]. Zero uses an empty
/// boxed slice, so it does not perform a separate limb allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredInteger {
    words: Box<[u64]>,
}

impl StoredInteger {
    /// Stores a gadget-local fixed integer without changing its value.
    pub fn from_fixed<const LIMBS: usize>(value: Integer<LIMBS>) -> Self {
        if value.is_zero() {
            return Self {
                words: Box::default(),
            };
        }

        let words = value.words();
        let mut len = LIMBS;
        while len > 1 {
            let top = words[len - 1];
            let next_is_negative = words[len - 2] >> 63 == 1;
            if (top == 0 && !next_is_negative) || (top == u64::MAX && next_is_negative) {
                len -= 1;
            } else {
                break;
            }
        }
        Self {
            words: words[..len].into(),
        }
    }

    /// Normalized little-endian two's-complement words; zero is empty.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Whether this integer is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.words.is_empty()
    }

    /// Whether this integer is negative.
    pub fn is_negative(&self) -> bool {
        self.words.last().is_some_and(|word| word >> 63 == 1)
    }
}

/// Dense exact integer `A(Mw)`, `B(Mw)`, and `C(Mw)` vectors.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IntegerProducts {
    pub a_mw: Vec<StoredInteger>,
    pub b_mw: Vec<StoredInteger>,
    pub c_mw: Vec<StoredInteger>,
}

impl IntegerProducts {
    /// Records one rank-1 row using its gadget-local integer width.
    pub fn push<const LIMBS: usize>(
        &mut self,
        a: Integer<LIMBS>,
        b: Integer<LIMBS>,
        c: Integer<LIMBS>,
    ) {
        self.a_mw.push(StoredInteger::from_fixed(a));
        self.b_mw.push(StoredInteger::from_fixed(b));
        self.c_mw.push(StoredInteger::from_fixed(c));
    }

    /// Reduces every materialized element modulo `modulus`.
    ///
    /// Vectors with at least 32,768 entries use Rayon. Smaller vectors remain
    /// sequential because dispatching them to the pool costs more than the
    /// available parallel work.
    pub fn reduce_parallel<const PRIME_LIMBS: usize>(
        &self,
        modulus: &RuntimeModulus<PRIME_LIMBS>,
    ) -> MatrixProducts<PRIME_LIMBS> {
        MatrixProducts {
            a_mw: reduce_vector(&self.a_mw, modulus),
            b_mw: reduce_vector(&self.b_mw, modulus),
            c_mw: reduce_vector(&self.c_mw, modulus),
        }
    }
}

fn reduce_vector<const PRIME_LIMBS: usize>(
    values: &[StoredInteger],
    modulus: &RuntimeModulus<PRIME_LIMBS>,
) -> ModularVector<PRIME_LIMBS> {
    const PARALLEL_THRESHOLD: usize = 1 << 15;
    let reduced = if values.len() >= PARALLEL_THRESHOLD {
        values
            .par_iter()
            .map(|value| modulus.reduce(value))
            .collect()
    } else {
        values.iter().map(|value| modulus.reduce(value)).collect()
    };
    ModularVector { values: reduced }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;
    use num_traits::Signed;

    use super::*;
    use crate::constraints::{ConstraintGenerator, ConstraintMatrices, SparseRow};
    use crate::sha256::{SHA256_2KB_MESSAGE_BITS, SHA256_2KB_WITNESS_BITS, sha256_2kb_circuit};
    use crate::witgen::{PackedWitness, ProductWitgen};

    fn stored_bigint(value: &StoredInteger) -> BigInt {
        let bytes = value
            .words()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        BigInt::from_signed_bytes_le(&bytes)
    }

    fn direct_row(row: &SparseRow<BigInt>, integer_witness: &PackedWitness) -> BigInt {
        row.entries()
            .iter()
            .filter(|(column, _)| integer_witness.bit(*column))
            .map(|(_, coefficient)| coefficient.clone())
            .sum()
    }

    fn reduce_bigint<const P: usize>(value: BigInt, modulus: &BigUint) -> [u64; P] {
        let modulus = BigInt::from(modulus.clone());
        let mut reduced = value % &modulus;
        if reduced.is_negative() {
            reduced += &modulus;
        }
        biguint_words(&reduced.to_biguint().unwrap())
    }

    fn assert_products_match_direct<const P: usize>(
        exact: &IntegerProducts,
        products: &MatrixProducts<P>,
        matrices: &ConstraintMatrices,
        integer_witness: &PackedWitness,
        modulus: &BigUint,
    ) {
        assert_eq!(exact.a_mw.len(), matrices.a.row_count());
        assert_eq!(exact.b_mw.len(), matrices.b.row_count());
        assert_eq!(exact.c_mw.len(), matrices.c.row_count());
        assert_eq!(products.a_mw.len(), matrices.a.row_count());
        assert_eq!(products.b_mw.len(), matrices.b.row_count());
        assert_eq!(products.c_mw.len(), matrices.c.row_count());
        for row in 0..matrices.a.row_count() {
            let expected_a = direct_row(&matrices.a.rows()[row], integer_witness);
            let expected_b = direct_row(&matrices.b.rows()[row], integer_witness);
            let expected_c = direct_row(&matrices.c.rows()[row], integer_witness);

            assert_eq!(
                stored_bigint(&exact.a_mw[row]),
                expected_a,
                "exact A(Mw) differs at row {row}"
            );
            assert_eq!(
                stored_bigint(&exact.b_mw[row]),
                expected_b,
                "exact B(Mw) differs at row {row}"
            );
            assert_eq!(
                stored_bigint(&exact.c_mw[row]),
                expected_c,
                "exact C(Mw) differs at row {row}"
            );
            assert_eq!(
                products.a_mw.get(row),
                reduce_bigint(expected_a, modulus),
                "A(Mw) differs at row {row}"
            );
            assert_eq!(
                products.b_mw.get(row),
                reduce_bigint(expected_b, modulus),
                "B(Mw) differs at row {row}"
            );
            assert_eq!(
                products.c_mw.get(row),
                reduce_bigint(expected_c, modulus),
                "C(Mw) differs at row {row}"
            );
        }
    }

    #[test]
    fn signed_stored_integers_reduce_correctly() {
        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = RuntimeModulus::<2>::new(modulus.clone()).unwrap();

        for value in [0_i128, 1, 7, -7, i128::from(i64::MIN)] {
            let stored = StoredInteger::from_fixed(Integer::<2>::from(value));
            let mut expected = BigInt::from(value) % BigInt::from(modulus.clone());
            if expected.is_negative() {
                expected += BigInt::from(modulus.clone());
            }
            assert_eq!(
                runtime_modulus.reduce(&stored),
                biguint_words::<2>(&expected.to_biguint().unwrap())
            );
        }
    }

    #[test]
    fn modulus_round_trips() {
        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = RuntimeModulus::<2>::new(modulus.clone()).unwrap();
        assert_eq!(runtime_modulus.modulus(), modulus);
    }

    #[test]
    fn sha256_recorded_products_match_direct_matrices() {
        let message: Box<[bool]> = (0..SHA256_2KB_MESSAGE_BITS)
            .map(|bit| {
                let byte = (bit / 8) as u8;
                byte & (1 << (7 - bit % 8)) != 0
            })
            .collect();
        let message: Box<[bool; SHA256_2KB_MESSAGE_BITS]> = message.try_into().unwrap();

        let mut witgen =
            ProductWitgen::with_inputs_and_capacity(message.as_ref(), SHA256_2KB_WITNESS_BITS);
        let _ = sha256_2kb_circuit(&mut witgen, &message);

        let mut generator = ConstraintGenerator::new(SHA256_2KB_MESSAGE_BITS);
        let symbolic_inputs = generator.boxed_inputs();
        let _ = sha256_2kb_circuit(&mut generator, &symbolic_inputs);
        let matrices = generator.into_matrices();

        let modulus = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let runtime_modulus = RuntimeModulus::<2>::new(modulus.clone()).unwrap();
        let products = witgen.products().reduce_parallel(&runtime_modulus);
        assert_products_match_direct(
            witgen.products(),
            &products,
            &matrices,
            witgen.integer_witness(),
            &modulus,
        );
    }
}
