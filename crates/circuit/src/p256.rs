//! ECDSA-P256 verification over a prehashed 256-bit digest.
//!
//! This is a direct circuit-shape port of Freigen's Lean implementation.  It
//! uses the same lazy integer representatives, quotient widths, complete
//! affine formulas, and joint fixed/variable-base scalar multiplication.

use std::array;
use std::cmp::Ordering;
use std::sync::{LazyLock, OnceLock};

use crypto_bigint::{Odd, U256 as CryptoU256};
use num_traits::{One, Zero};

use crate::{
    BoolRepresentation, BoolWitness, Circuit, HintError, HintResult, PackedBits, WitnessContext,
};

/// Number of Boolean inputs: digest, public-key coordinates, signature
/// scalars, and the two inverse witnesses.
pub const VERIFY_DIGEST_INPUT_BITS: usize = 7 * 256;

/// Total Boolean witness size of the standalone verifier, including inputs.
pub const VERIFY_DIGEST_WITNESS_BITS: usize = 1_215_662;

/// Number of packed `M * w` bits, including the implicit constant one.
pub const VERIFY_DIGEST_INTEGER_WITNESS_BITS: usize = 1_215_663;

/// Number of rank-1 constraints.
pub const VERIFY_DIGEST_R1CS_ROWS: usize = 7_061;

/// Signed width used by witness-oriented backends for P-256 intermediates.
/// The widest values are products of 262-bit affine-formula operands.
pub const P256_Z_LIMBS: usize = 9;

const WIDTH: usize = 256;
const WORD_LIMBS: usize = 4;

type P256Z<CS> = <CS as Circuit>::Z<P256_Z_LIMBS>;
type P256Coefficient<CS> = <CS as Circuit>::Coefficient<P256_Z_LIMBS>;
type S = num_bigint::BigUint;

fn parse_hex(hex: &[u8]) -> S {
    S::parse_bytes(hex, 16).unwrap()
}
static BASE_MODULUS: LazyLock<S> = LazyLock::new(|| {
    parse_hex(b"ffffffff00000001000000000000000000000000ffffffffffffffffffffffff")
});

static SCALAR_MODULUS: LazyLock<S> = LazyLock::new(|| {
    parse_hex(b"ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551")
});

static CURVE_B: LazyLock<S> = LazyLock::new(|| {
    parse_hex(b"5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b")
});

static GENERATOR_X: LazyLock<S> = LazyLock::new(|| {
    parse_hex(b"6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296")
});

static GENERATOR_Y: LazyLock<S> = LazyLock::new(|| {
    parse_hex(b"4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5")
});

#[derive(Clone, Copy)]
enum Modulus {
    Base,
    Scalar,
}

impl Modulus {
    fn value(self) -> &'static S {
        match self {
            Self::Base => &BASE_MODULUS,
            Self::Scalar => &SCALAR_MODULUS,
        }
    }

    fn words(self) -> U256 {
        match self {
            Self::Base => U256([
                0xffff_ffff_ffff_ffff,
                0x0000_0000_ffff_ffff,
                0,
                0xffff_ffff_0000_0001,
            ]),
            Self::Scalar => U256([
                0xf3b9_cac2_fc63_2551,
                0xbce6_faad_a717_9e84,
                0xffff_ffff_ffff_ffff,
                0xffff_ffff_0000_0000,
            ]),
        }
    }
}

struct Lc<CS: Circuit> {
    z: P256Z<CS>,
}

#[derive(Clone)]
struct CapturedValue<ZW> {
    z: ZW,
}

impl<ZW> CapturedValue<ZW> {
    fn evaluate_words<'a, BW, C>(&'a self, context: &dyn WitnessContext<ZW, BW, C>) -> &'a [u64] {
        context
            .eval_z_words(&self.z)
            .expect("fixed-width integer hint context required by the P-256 circuit")
    }
}

impl<CS: Circuit> Clone for Lc<CS> {
    fn clone(&self) -> Self {
        Self { z: self.z.clone() }
    }
}

impl<CS: Circuit> Lc<CS> {
    fn capture(&self) -> CapturedValue<P256Z<CS>> {
        CapturedValue { z: self.z.clone() }
    }

    fn add(self, rhs: Self) -> Self {
        Self { z: self.z + rhs.z }
    }

    fn sub(self, rhs: Self) -> Self {
        Self { z: self.z - rhs.z }
    }

    fn scale(self, coefficient: &S) -> Self {
        self.scale_coefficient(coefficient_for::<CS>(coefficient))
    }

    fn scale_coefficient(self, coefficient: P256Coefficient<CS>) -> Self {
        Self {
            z: self.z * coefficient,
        }
    }
}

struct UInt<CS: Circuit, const N: usize, const M: usize> {
    bits: <CS::Bool as BoolWitness>::Repr<N, M>,
    value: Lc<CS>,
}

impl<CS: Circuit, const N: usize, const M: usize> Clone for UInt<CS, N, M> {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits.clone(),
            value: self.value.clone(),
        }
    }
}

type UInt256<CS> = UInt<CS, WIDTH, WORD_LIMBS>;

struct Elem<CS: Circuit> {
    value: UInt256<CS>,
}

impl<CS: Circuit> Clone for Elem<CS> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

struct Rep<CS: Circuit> {
    value: Lc<CS>,
    bound: usize,
}

impl<CS: Circuit> Clone for Rep<CS> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            bound: self.bound,
        }
    }
}

struct Point<CS: Circuit> {
    x: Rep<CS>,
    y: Rep<CS>,
    infinity: Lc<CS>,
}

impl<CS: Circuit> Clone for Point<CS> {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
            infinity: self.infinity.clone(),
        }
    }
}

fn coefficient_for<CS: Circuit>(value: &S) -> P256Coefficient<CS> {
    let mut words = [0; P256_Z_LIMBS];
    for (output, word) in words.iter_mut().zip(value.iter_u64_digits()) {
        *output = word;
    }
    CS::coefficient_from_le_words(&words)
}

fn lc_constant<CS: Circuit>(value: &S) -> Lc<CS> {
    Lc {
        z: P256Z::<CS>::from(coefficient_for::<CS>(value)),
    }
}

fn lc_words<CS: Circuit>(words: &[u64]) -> Lc<CS> {
    Lc {
        z: P256Z::<CS>::from(CS::coefficient_from_le_words(words)),
    }
}

fn lc_u64<CS: Circuit>(value: u64) -> Lc<CS> {
    Lc {
        z: P256Z::<CS>::from(P256Coefficient::<CS>::from(value)),
    }
}

fn assert_zero<CS: Circuit>(circuit: &mut CS, value: Lc<CS>) {
    circuit.assert_r1c::<P256_Z_LIMBS>(P256Z::<CS>::zero(), P256Z::<CS>::zero(), value.z);
}

fn repr_bits<BW, const N: usize, const M: usize>(repr: &BW::Repr<N, M>) -> Vec<BW>
where
    BW: BoolWitness,
{
    (0..N).map(|index| repr.bit(index)).collect()
}

fn uint_from_repr<CS: Circuit, const N: usize, const M: usize>(
    circuit: &mut CS,
    bits: <CS::Bool as BoolWitness>::Repr<N, M>,
) -> UInt<CS, N, M> {
    let (z, _) = circuit.bitz_unsigned::<P256_Z_LIMBS, N, M, N>(&bits);
    UInt {
        bits,
        value: Lc { z },
    }
}

fn uint_from_repr_with_lifts<CS: Circuit, const N: usize, const M: usize>(
    circuit: &mut CS,
    bits: <CS::Bool as BoolWitness>::Repr<N, M>,
) -> (UInt<CS, N, M>, Vec<Lc<CS>>) {
    let handles = repr_bits::<CS::Bool, N, M>(&bits);
    let mut lifted = Vec::with_capacity(N);
    let mut z = P256Z::<CS>::zero();
    let mut power = P256Coefficient::<CS>::one();
    for bit in handles {
        let bit_z = circuit.bitz::<P256_Z_LIMBS>(bit);
        z += bit_z.clone() * power.clone();
        lifted.push(Lc { z: bit_z });
        power += power.clone();
    }
    (
        UInt {
            bits,
            value: Lc { z },
        },
        lifted,
    )
}

fn hinted_bits<CS: Circuit, const N: usize, const M: usize>(
    circuit: &mut CS,
    expression: CapturedValue<P256Z<CS>>,
    description: &'static str,
) -> <CS::Bool as BoolWitness>::Repr<N, M> {
    circuit.hint::<P256_Z_LIMBS, N, M, _>(move |context| {
        packed_evaluated(expression.evaluate_words(context), description)
    })
}

fn uint_from_int<CS: Circuit, const N: usize, const M: usize>(
    circuit: &mut CS,
    value: Lc<CS>,
    description: &'static str,
) -> UInt<CS, N, M> {
    let bits = hinted_bits::<CS, N, M>(circuit, value.capture(), description);
    let output = uint_from_repr(circuit, bits);
    assert_zero(circuit, value.sub(output.value.clone()));
    output
}

fn assert_lt<CS: Circuit>(circuit: &mut CS, modulus: Modulus, value: &UInt256<CS>) {
    let mut bound = modulus.words().0;
    let mut borrow = true;
    for word in &mut bound {
        let (difference, next_borrow) = word.overflowing_sub(u64::from(borrow));
        *word = difference;
        borrow = next_borrow;
    }
    debug_assert!(!borrow);
    let slack = lc_words::<CS>(&bound).sub(value.value.clone());
    let _ = uint_from_int::<CS, WIDTH, WORD_LIMBS>(circuit, slack, "range-check slack");
}

fn of_u<CS: Circuit>(circuit: &mut CS, modulus: Modulus, value: UInt256<CS>) -> Elem<CS> {
    assert_lt(circuit, modulus, &value);
    Elem { value }
}

fn of_elem<CS: Circuit>(value: &Elem<CS>) -> Rep<CS> {
    Rep {
        value: value.value.value.clone(),
        bound: 2,
    }
}

fn rep_constant<CS: Circuit>(value: &S) -> Rep<CS> {
    Rep {
        value: lc_constant(value),
        bound: 2,
    }
}

fn rep_u64<CS: Circuit>(value: u64) -> Rep<CS> {
    Rep {
        value: lc_u64(value),
        bound: 2,
    }
}

fn rep_add<CS: Circuit>(left: Rep<CS>, right: Rep<CS>) -> Rep<CS> {
    Rep {
        value: left.value.add(right.value),
        bound: left.bound + right.bound,
    }
}

fn rep_sub<CS: Circuit>(modulus: Modulus, left: Rep<CS>, right: Rep<CS>) -> Rep<CS> {
    let bias = modulus_multiple(modulus, right.bound);
    Rep {
        value: left.value.add(lc_words(&bias.0)).sub(right.value),
        bound: left.bound + right.bound,
    }
}

fn rep_scale<CS: Circuit>(coefficient: usize, value: Rep<CS>) -> Rep<CS> {
    Rep {
        value: value
            .value
            .scale_coefficient(P256Coefficient::<CS>::from(coefficient as u64)),
        bound: coefficient * value.bound,
    }
}

fn split_521<CS: Circuit>(
    circuit: &mut CS,
    bits: <CS::Bool as BoolWitness>::Repr<521, 9>,
) -> (UInt256<CS>, UInt<CS, 265, 5>) {
    let remainder = bits.slice::<256, 4>(0);
    let quotient = bits.slice::<265, 5>(WIDTH);
    (
        uint_from_repr::<CS, WIDTH, WORD_LIMBS>(circuit, remainder),
        uint_from_repr::<CS, 265, 5>(circuit, quotient),
    )
}

fn lazy_mul<CS: Circuit>(circuit: &mut CS, modulus: Modulus, x: Rep<CS>, y: Rep<CS>) -> Rep<CS> {
    let x_eval = x.value.capture();
    let y_eval = y.value.capture();
    let modulus_words = modulus.words();
    let bits = circuit.hint::<P256_Z_LIMBS, 521, 9, _>(move |context| {
        let a = Wide::<9>::from_evaluated(
            x_eval.evaluate_words(context),
            "lazy multiplication operand",
        )?;
        let b = Wide::<9>::from_evaluated(
            y_eval.evaluate_words(context),
            "lazy multiplication operand",
        )?;
        let product = multiply_wide(a, b);
        let (quotient, remainder) = product.div_rem(modulus_words);
        Ok(packed_wide_remainder_quotient(remainder, quotient))
    });
    let (r, q) = split_521(circuit, bits);
    let rhs = r.value.clone().add(q.value.clone().scale(modulus.value()));
    circuit.assert_r1c::<P256_Z_LIMBS>(x.value.z, y.value.z, rhs.z);
    Rep {
        value: r.value,
        bound: 2,
    }
}

fn lazy_mul_sub_to_elem<CS: Circuit>(
    circuit: &mut CS,
    modulus: Modulus,
    x: Rep<CS>,
    y: Rep<CS>,
    target: Rep<CS>,
) -> Elem<CS> {
    let bias = modulus_multiple(modulus, target.bound);
    let x_eval = x.value.capture();
    let y_eval = y.value.capture();
    let target_eval = target.value.capture();
    let modulus_words = modulus.words();
    let hint_bias = bias;
    let bits = circuit.hint::<P256_Z_LIMBS, 521, 9, _>(move |context| {
        let x = Wide::<9>::from_evaluated(x_eval.evaluate_words(context), "affine factor")?;
        let y = Wide::<9>::from_evaluated(y_eval.evaluate_words(context), "affine factor")?;
        let target = Wide::<9>::from_evaluated(
            target_eval.evaluate_words(context),
            "affine product target",
        )?;
        let mut shifted = multiply_wide(x, y);
        if shifted.add_assign(hint_bias) || !shifted.sub_assign(target) {
            return Err(HintError::new("invalid affine product dividend bounds"));
        }
        let (quotient, remainder) = shifted.div_rem(modulus_words);
        Ok(packed_wide_remainder_quotient(remainder, quotient))
    });
    let (r, q) = split_521(circuit, bits);
    let rhs = r
        .value
        .clone()
        .add(target.value.clone())
        .add(q.value.clone().scale(modulus.value()))
        .sub(lc_words(&bias.0));
    circuit.assert_r1c::<P256_Z_LIMBS>(x.value.z, y.value.z, rhs.z);
    Elem { value: r }
}

fn lazy_divide<CS: Circuit>(
    circuit: &mut CS,
    modulus: Modulus,
    denominator: Rep<CS>,
    numerator: Rep<CS>,
) -> Rep<CS> {
    let bias = modulus_multiple(modulus, numerator.bound);
    let denominator_eval = denominator.value.capture();
    let numerator_eval = numerator.value.capture();
    let modulus_words = modulus.words();
    let hint_bias = bias;
    let bits = circuit.hint::<P256_Z_LIMBS, 521, 9, _>(move |context| {
        let a = Wide::<9>::from_evaluated(
            denominator_eval.evaluate_words(context),
            "division denominator",
        )?;
        let b = Wide::<9>::from_evaluated(
            numerator_eval.evaluate_words(context),
            "division numerator",
        )?;
        let (_, denominator) = a.div_rem(modulus_words);
        let inverse = modular_inverse_u256(denominator, modulus_words)
            .ok_or_else(|| HintError::new("zero or noninvertible division denominator"))?;
        let (_, numerator) = b.div_rem(modulus_words);
        let inverse_product = multiply_wide(widen_u256(inverse), widen_u256(numerator));
        let (_, value) = inverse_product.div_rem(modulus_words);
        let mut shifted = multiply_wide(widen_u256(value), a);
        if shifted.add_assign(hint_bias) || !shifted.sub_assign(b) {
            return Err(HintError::new("invalid lazy division dividend bounds"));
        }
        let (quotient, remainder) = shifted.div_rem(modulus_words);
        debug_assert!(remainder.is_zero());
        Ok(packed_wide_remainder_quotient(value, quotient))
    });
    let (value, q) = split_521(circuit, bits);
    let rhs = numerator
        .value
        .clone()
        .add(q.value.clone().scale(modulus.value()))
        .sub(lc_words(&bias.0));
    circuit.assert_r1c::<P256_Z_LIMBS>(value.value.z.clone(), denominator.value.z, rhs.z);
    Rep {
        value: value.value,
        bound: 2,
    }
}

fn lazy_reduce<CS: Circuit>(circuit: &mut CS, modulus: Modulus, x: Rep<CS>) -> Elem<CS> {
    let x_eval = x.value.capture();
    let modulus_words = modulus.words();
    let bits = circuit.hint::<P256_Z_LIMBS, 521, 9, _>(move |context| {
        let value =
            Wide::<9>::from_evaluated(x_eval.evaluate_words(context), "lazy reduction operand")?;
        let (quotient, remainder) = value.div_rem(modulus_words);
        Ok(packed_wide_remainder_quotient(remainder, quotient))
    });
    let (r, q) = split_521(circuit, bits);
    let relation = x
        .value
        .sub(r.value.clone().add(q.value.scale(modulus.value())));
    assert_zero(circuit, relation);
    of_u(circuit, modulus, r)
}

fn lazy_reduce_scalar<CS: Circuit>(
    circuit: &mut CS,
    modulus: Modulus,
    x: Rep<CS>,
) -> ScalarElem<CS> {
    let x_eval = x.value.capture();
    let modulus_words = modulus.words();
    let bits = circuit.hint::<P256_Z_LIMBS, 521, 9, _>(move |context| {
        let value =
            Wide::<9>::from_evaluated(x_eval.evaluate_words(context), "lazy reduction operand")?;
        let (quotient, remainder) = value.div_rem(modulus_words);
        Ok(packed_wide_remainder_quotient(remainder, quotient))
    });
    let remainder = bits.slice::<256, 4>(0);
    let quotient = bits.slice::<265, 5>(WIDTH);
    let (r, int_bits) = uint_from_repr_with_lifts::<CS, 256, 4>(circuit, remainder);
    let q = uint_from_repr::<CS, 265, 5>(circuit, quotient);
    let relation = x
        .value
        .sub(r.value.clone().add(q.value.scale(modulus.value())));
    assert_zero(circuit, relation);
    assert_lt(circuit, modulus, &r);
    ScalarElem {
        elem: Elem { value: r },
        int_bits,
    }
}

fn lazy_assert_mul_eq<CS: Circuit>(
    circuit: &mut CS,
    modulus: Modulus,
    x: Rep<CS>,
    y: Rep<CS>,
    target: Rep<CS>,
) {
    let bias = modulus_multiple(modulus, target.bound);
    let x_eval = x.value.capture();
    let y_eval = y.value.capture();
    let target_eval = target.value.capture();
    let modulus_words = modulus.words();
    let hint_bias = bias;
    let bits = circuit.hint::<P256_Z_LIMBS, 265, 5, _>(move |context| {
        let x = Wide::<9>::from_evaluated(x_eval.evaluate_words(context), "relation factor")?;
        let y = Wide::<9>::from_evaluated(y_eval.evaluate_words(context), "relation factor")?;
        let target = Wide::<9>::from_evaluated(
            target_eval.evaluate_words(context),
            "modular relation target",
        )?;
        let mut shifted = multiply_wide(x, y);
        if shifted.add_assign(hint_bias) || !shifted.sub_assign(target) {
            return Err(HintError::new("invalid modular relation quotient bounds"));
        }
        let (quotient, remainder) = shifted.div_rem(modulus_words);
        debug_assert!(remainder.is_zero());
        Ok(packed_wide(quotient))
    });
    let q = uint_from_repr::<CS, 265, 5>(circuit, bits);
    let rhs = target
        .value
        .add(q.value.scale(modulus.value()))
        .sub(lc_words(&bias.0));
    circuit.assert_r1c::<P256_Z_LIMBS>(x.value.z, y.value.z, rhs.z);
}

fn relaxed_reduce_small<CS: Circuit>(circuit: &mut CS, modulus: Modulus, x: Lc<CS>) -> Elem<CS> {
    let x_eval = x.capture();
    let modulus_words = modulus.words();
    let bits = circuit.hint::<P256_Z_LIMBS, 258, 5, _>(move |context| {
        let value =
            Wide::<9>::from_evaluated(x_eval.evaluate_words(context), "relaxed modular dividend")?;
        let (quotient, remainder) = value.div_rem(modulus_words);
        Ok(packed_wide_remainder_quotient(remainder, quotient))
    });
    let r_bits = bits.slice::<256, 4>(0);
    let q_bits = bits.slice::<2, 1>(WIDTH);
    let r = uint_from_repr::<CS, 256, 4>(circuit, r_bits);
    let q = uint_from_repr::<CS, 2, 1>(circuit, q_bits);
    let relation = x.sub(r.value.clone().add(q.value.scale(modulus.value())));
    assert_zero(circuit, relation);
    Elem { value: r }
}

fn relaxed_mul<CS: Circuit>(
    circuit: &mut CS,
    modulus: Modulus,
    x: Elem<CS>,
    y: Elem<CS>,
) -> Elem<CS> {
    let x_eval = x.value.value.capture();
    let y_eval = y.value.value.capture();
    let modulus_words = modulus.words();
    let bits = circuit.hint::<P256_Z_LIMBS, 514, 9, _>(move |context| {
        let a = Wide::<9>::from_evaluated(
            x_eval.evaluate_words(context),
            "relaxed multiplication factor",
        )?;
        let b = Wide::<9>::from_evaluated(
            y_eval.evaluate_words(context),
            "relaxed multiplication factor",
        )?;
        let value = multiply_wide(a, b);
        let (quotient, remainder) = value.div_rem(modulus_words);
        Ok(packed_wide_remainder_quotient(remainder, quotient))
    });
    let r_bits = bits.slice::<256, 4>(0);
    let q_bits = bits.slice::<258, 5>(WIDTH);
    let r = uint_from_repr::<CS, 256, 4>(circuit, r_bits);
    let q = uint_from_repr::<CS, 258, 5>(circuit, q_bits);
    let rhs = r.value.clone().add(q.value.scale(modulus.value()));
    circuit.assert_r1c::<P256_Z_LIMBS>(x.value.value.z, y.value.value.z, rhs.z);
    Elem { value: r }
}

fn packed_evaluated<const N: usize, const M: usize>(
    words: &[u64],
    description: &str,
) -> HintResult<PackedBits<N, M>> {
    if words.last().is_some_and(|word| word >> 63 != 0) {
        return Err(HintError::new(format!("negative {description}")));
    }
    Ok(PackedBits::from_words(array::from_fn(|index| {
        words.get(index).copied().unwrap_or(0)
    })))
}

fn evaluated_is_one(words: &[u64]) -> bool {
    words.first() == Some(&1) && words[1..].iter().all(|word| *word == 0)
}

fn evaluated_usize(words: &[u64], description: &str) -> HintResult<usize> {
    if words.last().is_some_and(|word| word >> 63 != 0)
        || words
            .get(1..)
            .is_some_and(|words| words.iter().any(|word| *word != 0))
    {
        return Err(HintError::new(format!("invalid {description}")));
    }
    usize::try_from(words.first().copied().unwrap_or(0))
        .map_err(|_| HintError::new(format!("oversized {description}")))
}

fn packed_flag_inverse_u256(flag: bool, inverse: U256) -> PackedBits<257, 5> {
    let mut carry = u64::from(flag);
    PackedBits::from_words(array::from_fn(|index| {
        let word = inverse.0.get(index).copied().unwrap_or(0);
        let output = (word << 1) | carry;
        carry = word >> 63;
        output
    }))
}

/// An allocation-free integer used by P-256 hint arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct U256([u64; 4]);

impl U256 {
    const ZERO: Self = Self([0; 4]);

    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    fn is_even(self) -> bool {
        self.0[0] & 1 == 0
    }

    fn cmp_words(self, rhs: Self) -> Ordering {
        for index in (0..4).rev() {
            match self.0[index].cmp(&rhs.0[index]) {
                Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        Ordering::Equal
    }
}

impl<'a> TryFrom<&'a S> for U256 {
    type Error = ();

    fn try_from(value: &'a S) -> Result<Self, Self::Error> {
        let digits = value.to_u64_digits();
        if digits.len() > 4 {
            return Err(());
        }
        let mut words = [0; 4];
        words[..digits.len()].copy_from_slice(&digits);
        Ok(Self(words))
    }
}

impl From<U256> for S {
    fn from(v: U256) -> Self {
        let digits =
            v.0.into_iter()
                .flat_map(|word| [word as u32, (word >> 32) as u32]);
        Self::new(digits.collect())
    }
}

/// A little-endian unsigned integer for P-256 hint arithmetic. Nine limbs
/// match the circuit's signed Z capacity; eighteen hold a full product.
#[derive(Clone, Copy)]
struct Wide<const L: usize>([u64; L]);

impl<const L: usize> Wide<L> {
    fn from_evaluated(words: &[u64], description: &str) -> HintResult<Self> {
        if words.last().is_some_and(|word| word >> 63 != 0) {
            return Err(HintError::new(format!("negative {description}")));
        }
        if words.len() > L && words[L..].iter().any(|word| *word != 0) {
            return Err(HintError::new(format!("oversized {description}")));
        }
        let mut output = [0; L];
        let copied = words.len().min(L);
        output[..copied].copy_from_slice(&words[..copied]);
        Ok(Self(output))
    }

    fn add_assign<const K: usize>(&mut self, rhs: Wide<K>) -> bool {
        let mut carry = false;
        for index in 0..L {
            let word = rhs.0.get(index).copied().unwrap_or(0);
            let (sum, first_carry) = self.0[index].overflowing_add(word);
            let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
            self.0[index] = sum;
            carry = first_carry || second_carry;
        }
        carry
            || rhs
                .0
                .get(L..)
                .is_some_and(|words| words.iter().any(|word| *word != 0))
    }

    /// Subtracts `rhs`, returning false if the unsigned result would be
    /// negative. The wrapped output must be discarded in that case.
    fn sub_assign<const K: usize>(&mut self, rhs: Wide<K>) -> bool {
        let mut borrow = false;
        for index in 0..L {
            let word = rhs.0.get(index).copied().unwrap_or(0);
            let (difference, first_borrow) = self.0[index].overflowing_sub(word);
            let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
            self.0[index] = difference;
            borrow = first_borrow || second_borrow;
        }
        !borrow
            && !rhs
                .0
                .get(L..)
                .is_some_and(|words| words.iter().any(|word| *word != 0))
    }

    fn div_rem(self, modulus: U256) -> (Wide<L>, U256) {
        debug_assert!(modulus.0[3] >> 63 != 0);
        let mut quotient = [0; L];
        let mut dividend = self.0;
        let dividend_len = dividend
            .iter()
            .rposition(|word| *word != 0)
            .map_or(0, |index| index + 1);
        if dividend_len <= 4 {
            let mut remainder = [0; 4];
            remainder[..dividend_len].copy_from_slice(&dividend[..dividend_len]);
            let remainder = U256(remainder);
            if remainder.cmp_words(modulus) == Ordering::Less {
                return (Wide(quotient), remainder);
            }
        }

        // Knuth's Algorithm D. Both P-256 moduli are already normalized (the
        // top divisor bit is set), so no normalization shifts are needed.
        let divisor_high = modulus.0[3];
        let divisor_next = modulus.0[2];
        let quotient_len = dividend_len - 3;
        let mut extra_high = 0_u64;
        for j in (0..quotient_len).rev() {
            let active_len = 4 + j;
            let high = dividend[active_len - 1];
            let next = dividend[active_len - 2];
            let (mut estimate, mut estimate_remainder) = if extra_high < divisor_high {
                let numerator = (u128::from(extra_high) << 64) | u128::from(high);
                (
                    (numerator / u128::from(divisor_high)) as u64,
                    numerator % u128::from(divisor_high),
                )
            } else {
                debug_assert_eq!(extra_high, divisor_high);
                (u64::MAX, u128::from(extra_high) + u128::from(high))
            };

            while estimate_remainder <= u128::from(u64::MAX)
                && (estimate_remainder << 64) + u128::from(next)
                    < u128::from(estimate) * u128::from(divisor_next)
            {
                estimate -= 1;
                estimate_remainder += u128::from(divisor_high);
            }

            let mut offset_carry = u64::MAX;
            for (word, divisor) in dividend[j..j + 4].iter_mut().zip(modulus.0) {
                let offset_sum = (u128::from(u64::MAX) << 64) + u128::from(*word)
                    - u128::from(u64::MAX)
                    + u128::from(offset_carry)
                    - u128::from(divisor) * u128::from(estimate);
                *word = offset_sum as u64;
                offset_carry = (offset_sum >> 64) as u64;
            }
            let mut borrow = u64::MAX - offset_carry;
            if borrow > extra_high {
                estimate -= 1;
                let mut carry = false;
                for (word, divisor) in dividend[j..j + 4].iter_mut().zip(modulus.0) {
                    let (sum, first_carry) = word.overflowing_add(divisor);
                    let (sum, second_carry) = sum.overflowing_add(u64::from(carry));
                    *word = sum;
                    carry = first_carry || second_carry;
                }
                borrow -= u64::from(carry);
            }
            debug_assert_eq!(borrow, extra_high);
            quotient[j] = estimate;
            extra_high = dividend[active_len - 1];
        }

        let remainder = U256([dividend[0], dividend[1], dividend[2], extra_high]);
        debug_assert!(remainder.cmp_words(modulus) == Ordering::Less);
        (Wide(quotient), remainder)
    }
}

fn multiply_wide(left: Wide<9>, right: Wide<9>) -> Wide<18> {
    let mut output = [0; 18];
    let left_len = left
        .0
        .iter()
        .rposition(|word| *word != 0)
        .map_or(0, |index| index + 1);
    let right_len = right
        .0
        .iter()
        .rposition(|word| *word != 0)
        .map_or(0, |index| index + 1);
    for i in 0..left_len {
        let mut carry = 0_u128;
        for j in 0..right_len {
            let accumulated =
                u128::from(left.0[i]) * u128::from(right.0[j]) + u128::from(output[i + j]) + carry;
            output[i + j] = accumulated as u64;
            carry = accumulated >> 64;
        }
        output[i + right_len] = carry as u64;
    }
    Wide(output)
}

fn widen_u256<const L: usize>(value: U256) -> Wide<L> {
    assert!(L >= 4);
    let mut output = [0; L];
    output[..4].copy_from_slice(&value.0);
    Wide(output)
}

fn modulus_multiple(modulus: Modulus, factor: usize) -> Wide<9> {
    let factor = u64::try_from(factor).expect("P-256 representative bound exceeds u64");
    let mut output = [0; 9];
    let mut carry = 0_u128;
    for (index, word) in modulus.words().0.into_iter().enumerate() {
        let product = u128::from(word) * u128::from(factor) + carry;
        output[index] = product as u64;
        carry = product >> 64;
    }
    output[4] = carry as u64;
    Wide(output)
}

fn packed_wide<const N: usize, const M: usize, const L: usize>(value: Wide<L>) -> PackedBits<N, M> {
    PackedBits::from_words(array::from_fn(|index| {
        value.0.get(index).copied().unwrap_or(0)
    }))
}

fn packed_wide_remainder_quotient<const N: usize, const M: usize, const L: usize>(
    remainder: U256,
    quotient: Wide<L>,
) -> PackedBits<N, M> {
    PackedBits::from_words(array::from_fn(|index| {
        if index < 4 {
            remainder.0[index]
        } else {
            quotient.0.get(index - 4).copied().unwrap_or(0)
        }
    }))
}

fn modular_inverse_u256(u: U256, v: U256) -> Option<U256> {
    if u.cmp_words(v) != Ordering::Less {
        return None;
    }
    if u.is_zero() || v.is_zero() || v.is_even() {
        return None;
    }
    if u.0 == [1, 0, 0, 0] {
        return Some(u);
    }
    let value = CryptoU256::from_words(u.0);
    let modulus: Option<Odd<CryptoU256>> = Odd::new(CryptoU256::from_words(v.0)).into();
    let inverse: Option<CryptoU256> = value.invert_odd_mod_vartime(&modulus?).into();
    inverse.map(|inverse| U256(inverse.to_words()))
}

fn modular_inverse(value: &S, modulus: &S) -> Option<S> {
    let reduced;
    let value = if value < modulus {
        value
    } else {
        reduced = value % modulus;
        &reduced
    };
    modular_inverse_u256(U256::try_from(value).ok()?, U256::try_from(modulus).ok()?).map(S::from)
}

fn point_from_elems<CS: Circuit>(x: &Elem<CS>, y: &Elem<CS>) -> Point<CS> {
    Point {
        x: of_elem(x),
        y: of_elem(y),
        infinity: lc_u64(0),
    }
}

fn infinity<CS: Circuit>() -> Point<CS> {
    Point {
        x: rep_u64(0),
        y: rep_u64(0),
        infinity: lc_u64(1),
    }
}

fn and_bit<CS: Circuit>(circuit: &mut CS, x: Lc<CS>, y: Lc<CS>) -> Lc<CS> {
    let x_eval = x.capture();
    let y_eval = y.capture();
    let bits = circuit.hint::<P256_Z_LIMBS, 1, 1, _>(move |context| {
        let value = evaluated_is_one(x_eval.evaluate_words(context))
            && evaluated_is_one(y_eval.evaluate_words(context));
        Ok(PackedBits::from_array([value]))
    });
    let out = uint_from_repr::<CS, 1, 1>(circuit, bits).value;
    circuit.assert_r1c::<P256_Z_LIMBS>(x.z, y.z, out.z.clone());
    out
}

fn and3_bit<CS: Circuit>(circuit: &mut CS, x: Lc<CS>, y: Lc<CS>, z: Lc<CS>) -> Lc<CS> {
    let xy = and_bit(circuit, x, y);
    and_bit(circuit, z, xy)
}

fn lazy_zero_test<CS: Circuit>(circuit: &mut CS, modulus: Modulus, x: Rep<CS>) -> Lc<CS> {
    let x_eval = x.value.capture();
    let modulus_words = modulus.words();
    let bits = circuit.hint::<P256_Z_LIMBS, 257, 5, _>(move |context| {
        let a = Wide::<9>::from_evaluated(x_eval.evaluate_words(context), "zero-test operand")?;
        let (_, value) = a.div_rem(modulus_words);
        let is_zero = value.is_zero();
        let inverse = modular_inverse_u256(value, modulus_words).unwrap_or(U256::ZERO);
        Ok(packed_flag_inverse_u256(is_zero, inverse))
    });
    let z_bits = bits.slice::<1, 1>(0);
    let inverse_bits = bits.slice::<256, 4>(1);
    let z = uint_from_repr::<CS, 1, 1>(circuit, z_bits).value;
    let inverse = uint_from_repr::<CS, 256, 4>(circuit, inverse_bits);
    lazy_assert_mul_eq(
        circuit,
        modulus,
        x.clone(),
        of_elem(&Elem { value: inverse }),
        Rep {
            value: lc_u64::<CS>(1).sub(z.clone()),
            bound: 1,
        },
    );

    let z_eval = z.capture();
    let x_eval = x.value.capture();
    let modulus_words = modulus.words();
    let q_bits = circuit.hint::<P256_Z_LIMBS, 9, 1, _>(move |context| {
        let z = Wide::<9>::from_evaluated(z_eval.evaluate_words(context), "zero-test flag")?;
        let x = Wide::<9>::from_evaluated(
            x_eval.evaluate_words(context),
            "zero-test quotient operand",
        )?;
        let product = multiply_wide(z, x);
        let (quotient, remainder) = product.div_rem(modulus_words);
        debug_assert!(remainder.is_zero());
        Ok(packed_wide(quotient))
    });
    let q = uint_from_repr::<CS, 9, 1>(circuit, q_bits);
    circuit.assert_r1c::<P256_Z_LIMBS>(z.z.clone(), x.value.z, q.value.scale(modulus.value()).z);
    z
}

fn select_rep<CS: Circuit, const N: usize, const M: usize>(
    circuit: &mut CS,
    choose: Lc<CS>,
    when_one: Rep<CS>,
    when_zero: Rep<CS>,
    out_bound: usize,
) -> Rep<CS> {
    let choose_eval = choose.capture();
    let one_eval = when_one.value.capture();
    let zero_eval = when_zero.value.capture();
    let bits = circuit.hint::<P256_Z_LIMBS, N, M, _>(move |context| {
        let value = if evaluated_is_one(choose_eval.evaluate_words(context)) {
            one_eval.evaluate_words(context)
        } else {
            zero_eval.evaluate_words(context)
        };
        packed_evaluated(value, "selection")
    });
    let out = uint_from_repr::<CS, N, M>(circuit, bits).value;
    let difference = when_one.value.sub(when_zero.value.clone());
    let output_difference = out.clone().sub(when_zero.value);
    circuit.assert_r1c::<P256_Z_LIMBS>(choose.z, difference.z, output_difference.z);
    Rep {
        value: out,
        bound: out_bound,
    }
}

fn select_canonical<CS: Circuit>(
    circuit: &mut CS,
    choose: Lc<CS>,
    when_one: Rep<CS>,
    when_zero: Rep<CS>,
) -> Rep<CS> {
    select_rep::<CS, 256, 4>(circuit, choose, when_one, when_zero, 2)
}

fn select_formula<CS: Circuit>(
    circuit: &mut CS,
    choose: Lc<CS>,
    when_one: Rep<CS>,
    when_zero: Rep<CS>,
) -> Rep<CS> {
    select_rep::<CS, 262, 5>(circuit, choose, when_one, when_zero, 66)
}

fn double_complete<CS: Circuit>(circuit: &mut CS, point: Point<CS>) -> Point<CS> {
    let x2 = lazy_mul(circuit, Modulus::Base, point.x.clone(), point.x.clone());
    let numerator = rep_add(
        rep_sub(Modulus::Base, rep_scale(3, x2), rep_u64::<CS>(3)),
        Rep {
            value: point
                .infinity
                .clone()
                .scale_coefficient(P256Coefficient::<CS>::from(3_u64)),
            bound: 1,
        },
    );
    let denominator = rep_add(
        rep_scale(2, point.y.clone()),
        Rep {
            value: point.infinity.clone(),
            bound: 1,
        },
    );
    let slope = lazy_divide(circuit, Modulus::Base, denominator, numerator);
    let x3 = lazy_mul_sub_to_elem(
        circuit,
        Modulus::Base,
        slope.clone(),
        slope.clone(),
        rep_scale(2, point.x.clone()),
    );
    let x3_rep = of_elem(&x3);
    let y3 = lazy_mul_sub_to_elem(
        circuit,
        Modulus::Base,
        slope,
        rep_sub(Modulus::Base, point.x, x3_rep.clone()),
        point.y,
    );
    Point {
        x: x3_rep,
        y: of_elem(&y3),
        infinity: point.infinity,
    }
}

struct AddControl<CS: Circuit> {
    same_x: Lc<CS>,
    opposite_y: Lc<CS>,
    finite: Lc<CS>,
    double_case: Lc<CS>,
    active: Lc<CS>,
}

fn add_complete<CS: Circuit>(circuit: &mut CS, p: Point<CS>, q: Point<CS>) -> Point<CS> {
    let dx = rep_sub(Modulus::Base, q.x.clone(), p.x.clone());
    let y_sum = rep_add(p.y.clone(), q.y.clone());
    let same_x = lazy_zero_test(circuit, Modulus::Base, dx.clone());
    let opposite_y = lazy_zero_test(circuit, Modulus::Base, y_sum);
    let finite = and_bit(
        circuit,
        lc_u64::<CS>(1).sub(p.infinity.clone()),
        lc_u64::<CS>(1).sub(q.infinity.clone()),
    );
    let double_kind = and_bit(
        circuit,
        same_x.clone(),
        lc_u64::<CS>(1).sub(opposite_y.clone()),
    );
    let double_case = and_bit(circuit, finite.clone(), double_kind);
    let generic_case = and_bit(circuit, finite.clone(), lc_u64::<CS>(1).sub(same_x.clone()));
    let control = AddControl {
        same_x,
        opposite_y,
        finite,
        double_case: double_case.clone(),
        active: double_case.add(generic_case),
    };

    let dy = rep_sub(Modulus::Base, q.y.clone(), p.y.clone());
    let x2 = lazy_mul(circuit, Modulus::Base, p.x.clone(), p.x.clone());
    let double_numerator = rep_sub(Modulus::Base, rep_scale(3, x2), rep_u64::<CS>(3));
    let double_denominator = rep_scale(2, p.y.clone());
    let selected_numerator =
        select_formula(circuit, control.double_case.clone(), double_numerator, dy);
    let selected_denominator =
        select_formula(circuit, control.double_case.clone(), double_denominator, dx);
    let numerator = select_formula(
        circuit,
        control.active.clone(),
        selected_numerator,
        rep_u64(0),
    );
    let denominator = select_formula(
        circuit,
        control.active.clone(),
        selected_denominator,
        rep_u64(1),
    );
    let slope = lazy_divide(circuit, Modulus::Base, denominator, numerator);
    let candidate_x = lazy_mul_sub_to_elem(
        circuit,
        Modulus::Base,
        slope.clone(),
        slope.clone(),
        rep_add(p.x.clone(), q.x.clone()),
    );
    let candidate_x = of_elem(&candidate_x);
    let candidate_y = lazy_mul_sub_to_elem(
        circuit,
        Modulus::Base,
        slope,
        rep_sub(Modulus::Base, p.x.clone(), candidate_x.clone()),
        p.y.clone(),
    );
    let candidate_y = of_elem(&candidate_y);

    let inactive_x0 = select_canonical(circuit, q.infinity.clone(), p.x.clone(), rep_u64(0));
    let inactive_y0 = select_canonical(circuit, q.infinity.clone(), p.y.clone(), rep_u64(0));
    let inactive_x = select_canonical(circuit, p.infinity.clone(), q.x.clone(), inactive_x0);
    let inactive_y = select_canonical(circuit, p.infinity.clone(), q.y.clone(), inactive_y0);
    let x = select_canonical(circuit, control.active.clone(), candidate_x, inactive_x);
    let y = select_canonical(circuit, control.active, candidate_y, inactive_y);
    let both_infinity = and_bit(circuit, p.infinity, q.infinity);
    let finite_opposite = and3_bit(circuit, control.same_x, control.opposite_y, control.finite);
    Point {
        x,
        y,
        infinity: both_infinity.add(finite_opposite),
    }
}

fn materialize_multiples<CS: Circuit>(circuit: &mut CS, q: Point<CS>) -> Vec<Point<CS>> {
    let p1 = q;
    let mut output = Vec::with_capacity(16);
    output.push(infinity());
    output.push(p1.clone());
    let mut previous = p1.clone();
    for _ in 2..16 {
        previous = add_complete(circuit, previous, p1.clone());
        output.push(previous.clone());
    }
    output
}

fn indicators_impl<CS: Circuit, const N: usize, const M: usize>(
    circuit: &mut CS,
    digit: Lc<CS>,
) -> (UInt<CS, N, M>, Vec<Lc<CS>>) {
    let digit_eval = digit.capture();
    let bits = circuit.hint::<P256_Z_LIMBS, N, M, _>(move |context| {
        let digit = evaluated_usize(digit_eval.evaluate_words(context), "indicator digit")?;
        Ok(PackedBits::from_fn(|i| digit == i))
    });
    let bit_handles = repr_bits::<CS::Bool, N, M>(&bits);
    let mut lifted = Vec::with_capacity(N);
    let mut full = P256Z::<CS>::zero();
    let mut power = P256Coefficient::<CS>::one();
    for bit in &bit_handles {
        let z = circuit.bitz::<P256_Z_LIMBS>(bit.clone());
        full += z.clone() * power.clone();
        lifted.push(Lc { z });
        power += power.clone();
    }
    let out = UInt {
        bits,
        value: Lc { z: full },
    };
    let sum = lifted.iter().cloned().fold(lc_u64::<CS>(0), Lc::add);
    assert_zero(circuit, sum.sub(lc_u64(1)));
    let weighted = lifted
        .iter()
        .cloned()
        .enumerate()
        .fold(lc_u64::<CS>(0), |sum, (i, bit)| {
            sum.add(bit.scale_coefficient(P256Coefficient::<CS>::from(i as u64)))
        });
    assert_zero(circuit, weighted.sub(digit));
    (out, lifted)
}

fn lookup_rep<CS: Circuit>(
    circuit: &mut CS,
    digit: Lc<CS>,
    indicators: &[Lc<CS>],
    values: &[Rep<CS>],
) -> Rep<CS> {
    let digit_eval = digit.capture();
    let value_evals: Vec<_> = values.iter().map(|value| value.value.capture()).collect();
    let bits = circuit.hint::<P256_Z_LIMBS, 256, 4, _>(move |context| {
        let index = evaluated_usize(digit_eval.evaluate_words(context), "lookup digit")?;
        let expression = value_evals
            .get(index)
            .ok_or_else(|| HintError::new("lookup digit out of range"))?;
        packed_evaluated(expression.evaluate_words(context), "lookup value")
    });
    let out = uint_from_repr::<CS, 256, 4>(circuit, bits).value;
    for (indicator, value) in indicators.iter().zip(values) {
        circuit.assert_r1c::<P256_Z_LIMBS>(
            indicator.z.clone(),
            out.clone().sub(value.value.clone()).z,
            P256Z::<CS>::zero(),
        );
    }
    Rep {
        value: out,
        bound: 2,
    }
}

fn lookup_flag<CS: Circuit>(
    circuit: &mut CS,
    digit: Lc<CS>,
    indicators: &[Lc<CS>],
    flags: &[Lc<CS>],
) -> Lc<CS> {
    let digit_eval = digit.capture();
    let flag_evals: Vec<_> = flags.iter().map(Lc::capture).collect();
    let bits = circuit.hint::<P256_Z_LIMBS, 1, 1, _>(move |context| {
        let index = evaluated_usize(digit_eval.evaluate_words(context), "flag lookup digit")?;
        let flag = flag_evals
            .get(index)
            .ok_or_else(|| HintError::new("flag lookup digit out of range"))?;
        Ok(PackedBits::from_array([evaluated_is_one(
            flag.evaluate_words(context),
        )]))
    });
    let out = uint_from_repr::<CS, 1, 1>(circuit, bits).value;
    for (indicator, flag) in indicators.iter().zip(flags) {
        circuit.assert_r1c::<P256_Z_LIMBS>(
            indicator.z.clone(),
            out.clone().sub(flag.clone()).z,
            P256Z::<CS>::zero(),
        );
    }
    out
}

fn lookup_point<CS: Circuit>(circuit: &mut CS, digit: Lc<CS>, table: &[Point<CS>]) -> Point<CS> {
    let (_, indicator_bits) = indicators_impl::<CS, 16, 1>(circuit, digit.clone());
    let xs: Vec<_> = table.iter().map(|point| point.x.clone()).collect();
    let ys: Vec<_> = table.iter().map(|point| point.y.clone()).collect();
    let flags: Vec<_> = table.iter().map(|point| point.infinity.clone()).collect();
    Point {
        x: lookup_rep(circuit, digit.clone(), &indicator_bits, &xs),
        y: lookup_rep(circuit, digit.clone(), &indicator_bits, &ys),
        infinity: lookup_flag(circuit, digit, &indicator_bits, &flags),
    }
}

fn generator_table() -> &'static Vec<(S, S)> {
    static TABLE: OnceLock<Vec<(S, S)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let modulus = &*BASE_MODULUS;
        let mut table = Vec::with_capacity(256);
        table.push((S::zero(), S::zero()));
        let mut point = (GENERATOR_X.clone(), GENERATOR_Y.clone());
        table.push(point.clone());
        for _ in 2..256 {
            point = affine_add(&point, &(GENERATOR_X.clone(), GENERATOR_Y.clone()), modulus);
            table.push(point.clone());
        }
        table
    })
}

/// Materializes the fixed-base generator table used by witness generation.
/// Call this during benchmark or service setup to exclude one-time constant
/// initialization from latency measurements.
pub fn prepare() {
    let _ = generator_table();
}

fn affine_add(left: &(S, S), right: &(S, S), modulus: &S) -> (S, S) {
    let numerator = if left == right {
        (S::from(3_u64) * &left.0 * &left.0 + modulus - S::from(3_u64)) % modulus
    } else {
        (&right.1 + modulus - &left.1) % modulus
    };
    let denominator = if left == right {
        (S::from(2_u64) * &left.1) % modulus
    } else {
        (&right.0 + modulus - &left.0) % modulus
    };
    let slope = numerator * modular_inverse(&denominator, modulus).unwrap() % modulus;
    let x = (&slope * &slope + modulus * 2_u64 - &left.0 - &right.0) % modulus;
    let y = (&slope * ((&left.0 + modulus - &x) % modulus) + modulus - &left.1) % modulus;
    (x, y)
}

fn generator_coefficients<CS: Circuit>() -> Vec<(P256Coefficient<CS>, P256Coefficient<CS>)> {
    generator_table()
        .iter()
        .map(|(x, y)| (coefficient_for::<CS>(x), coefficient_for::<CS>(y)))
        .collect()
}

fn lookup_generator_byte<CS: Circuit>(
    circuit: &mut CS,
    digit: Lc<CS>,
    table: &[(P256Coefficient<CS>, P256Coefficient<CS>)],
) -> Point<CS> {
    let (_, indicators) = indicators_impl::<CS, 256, 4>(circuit, digit);
    let x = indicators
        .iter()
        .zip(table)
        .fold(lc_u64::<CS>(0), |sum, (bit, (x, _))| {
            sum.add(bit.clone().scale_coefficient(x.clone()))
        });
    let y = indicators
        .iter()
        .zip(table)
        .fold(lc_u64::<CS>(0), |sum, (bit, (_, y))| {
            sum.add(bit.clone().scale_coefficient(y.clone()))
        });
    Point {
        x: Rep { value: x, bound: 2 },
        y: Rep { value: y, bound: 2 },
        infinity: indicators[0].clone(),
    }
}

// A UInt retains the combined integer LC, but Lean's window expressions refer
// to individual intBits. Keep those lifts explicitly for scalar windows.
struct ScalarElem<CS: Circuit> {
    elem: Elem<CS>,
    int_bits: Vec<Lc<CS>>,
}

impl<CS: Circuit> Clone for ScalarElem<CS> {
    fn clone(&self) -> Self {
        Self {
            elem: self.elem.clone(),
            int_bits: self.int_bits.clone(),
        }
    }
}

fn lift_input_word<CS: Circuit>(
    circuit: &mut CS,
    bits: <CS::Bool as BoolWitness>::Repr<256, 4>,
) -> ScalarElem<CS> {
    let handles = repr_bits::<CS::Bool, 256, 4>(&bits);
    let mut lifted = Vec::with_capacity(256);
    let mut z = P256Z::<CS>::zero();
    let mut power = P256Coefficient::<CS>::one();
    for bit in handles {
        let bit_z = circuit.bitz::<P256_Z_LIMBS>(bit.clone());
        z += bit_z.clone() * power.clone();
        lifted.push(Lc { z: bit_z });
        power += power.clone();
    }
    ScalarElem {
        elem: Elem {
            value: UInt {
                bits,
                value: Lc { z },
            },
        },
        int_bits: lifted,
    }
}

fn scalar_window<CS: Circuit>(scalar: &ScalarElem<CS>, start: usize, width: usize) -> Lc<CS> {
    scalar.int_bits[start..start + width]
        .iter()
        .cloned()
        .enumerate()
        .fold(lc_u64::<CS>(0), |sum, (bit, value)| {
            sum.add(value.scale_coefficient(P256Coefficient::<CS>::from(1_u64 << bit)))
        })
}

fn joint_scalar_mul<CS: Circuit>(
    circuit: &mut CS,
    u1: ScalarElem<CS>,
    u2: ScalarElem<CS>,
    q: Point<CS>,
) -> Point<CS> {
    let q_table = materialize_multiples(circuit, q);
    let g_table = generator_coefficients::<CS>();
    let mut accumulator = infinity();
    for i in 0..32 {
        let d1 = scalar_window(&u1, 248 - 8 * i, 8);
        let d2_hi = scalar_window(&u2, 252 - 8 * i, 4);
        let d2_lo = scalar_window(&u2, 248 - 8 * i, 4);
        let q_hi = lookup_point(circuit, d2_hi, &q_table);
        let q_lo = lookup_point(circuit, d2_lo, &q_table);
        let g = lookup_generator_byte(circuit, d1, &g_table);
        for _ in 0..4 {
            accumulator = double_complete(circuit, accumulator);
        }
        accumulator = add_complete(circuit, accumulator, q_hi);
        for _ in 0..4 {
            accumulator = double_complete(circuit, accumulator);
        }
        accumulator = add_complete(circuit, accumulator, q_lo);
        accumulator = add_complete(circuit, accumulator, g);
    }
    accumulator
}

fn assert_on_curve<CS: Circuit>(circuit: &mut CS, x: &Elem<CS>, y: &Elem<CS>) {
    let x = of_elem(x);
    let y = of_elem(y);
    let x2 = lazy_mul(circuit, Modulus::Base, x.clone(), x.clone());
    let x3 = lazy_mul(circuit, Modulus::Base, x2, x.clone());
    let rhs = rep_sub(
        Modulus::Base,
        rep_add(x3, rep_constant(&CURVE_B)),
        rep_scale(3, x),
    );
    lazy_assert_mul_eq(circuit, Modulus::Base, y.clone(), y, rhs);
}

fn assert_elem_eq<CS: Circuit>(circuit: &mut CS, left: &Elem<CS>, right: &Elem<CS>) {
    assert_zero(
        circuit,
        left.value.value.clone().sub(right.value.value.clone()),
    );
}

/// Builds the standalone P-256 ECDSA verifier from seven little-endian
/// 256-bit input words ordered as digest, Q.x, Q.y, r, s, r^-1, s^-1.
pub fn verify_digest_circuit<CS: Circuit>(
    circuit: &mut CS,
    inputs: &[CS::Bool; VERIFY_DIGEST_INPUT_BITS],
) {
    let words: [<CS::Bool as BoolWitness>::Repr<256, 4>; 7] = array::from_fn(|slot| {
        <CS::Bool as BoolWitness>::Repr::from_array(array::from_fn(|bit| {
            inputs[slot * WIDTH + bit].clone()
        }))
    });
    // These seven U.fromWord calls are deliberately kept in input order.
    let mut words = words.into_iter();
    let digest = lift_input_word(circuit, words.next().unwrap());
    let qx = lift_input_word(circuit, words.next().unwrap());
    let qy = lift_input_word(circuit, words.next().unwrap());
    let r = lift_input_word(circuit, words.next().unwrap());
    let s = lift_input_word(circuit, words.next().unwrap());
    let r_inverse = lift_input_word(circuit, words.next().unwrap());
    let s_inverse = lift_input_word(circuit, words.next().unwrap());

    let qx = of_u(circuit, Modulus::Base, qx.elem.value);
    let qy = of_u(circuit, Modulus::Base, qy.elem.value);
    let r = of_u(circuit, Modulus::Scalar, r.elem.value);
    let s = of_u(circuit, Modulus::Scalar, s.elem.value);
    let r_inverse = of_u(circuit, Modulus::Scalar, r_inverse.elem.value);
    let s_inverse = of_u(circuit, Modulus::Scalar, s_inverse.elem.value);

    assert_on_curve(circuit, &qx, &qy);
    lazy_assert_mul_eq(
        circuit,
        Modulus::Scalar,
        of_elem(&r),
        of_elem(&r_inverse),
        rep_u64(1),
    );
    lazy_assert_mul_eq(
        circuit,
        Modulus::Scalar,
        of_elem(&s),
        of_elem(&s_inverse),
        rep_u64(1),
    );

    let z = relaxed_reduce_small(circuit, Modulus::Scalar, digest.elem.value.value);
    let u1_relaxed = relaxed_mul(circuit, Modulus::Scalar, z, s_inverse.clone());
    let u2_relaxed = relaxed_mul(circuit, Modulus::Scalar, r.clone(), s_inverse);
    let u1 = lazy_reduce_scalar(circuit, Modulus::Scalar, of_elem(&u1_relaxed));
    let u2 = lazy_reduce_scalar(circuit, Modulus::Scalar, of_elem(&u2_relaxed));

    let q = point_from_elems(&qx, &qy);
    let sum = joint_scalar_mul(circuit, u1, u2, q);
    assert_zero(circuit, sum.infinity);
    let x_canonical = lazy_reduce(circuit, Modulus::Base, sum.x);
    let x_mod_n = relaxed_reduce_small(circuit, Modulus::Scalar, x_canonical.value.value);
    assert_elem_eq(circuit, &x_mod_n, &r);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_products::StoredInteger;
    use crate::stats::{Dummy, LeanStats, Stats};
    use crate::witgen::{ProductWitgen, WitnessOnly};
    use num_bigint::BigInt;

    fn stored_bigint(value: &StoredInteger) -> BigInt {
        let bytes = value
            .words()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        BigInt::from_signed_bytes_le(&bytes)
    }

    fn words_biguint(words: &[u64]) -> S {
        S::new(
            words
                .iter()
                .flat_map(|word| [*word as u32, (*word >> 32) as u32])
                .collect(),
        )
    }

    fn valid_input() -> Box<[bool; VERIFY_DIGEST_INPUT_BITS]> {
        let r = GENERATOR_X.clone();
        let s = &r + S::one();
        let values = [
            S::one(),
            GENERATOR_X.clone(),
            GENERATOR_Y.clone(),
            r.clone(),
            s.clone(),
            modular_inverse(&r, &SCALAR_MODULUS).unwrap(),
            modular_inverse(&s, &SCALAR_MODULUS).unwrap(),
        ];
        let bits: Box<[bool]> = (0..VERIFY_DIGEST_INPUT_BITS)
            .map(|index| values[index / WIDTH].bit((index % WIDTH) as u64))
            .collect();
        bits.try_into().unwrap()
    }

    #[test]
    fn verifier_dimensions_exactly_match_lean() {
        let mut stats = Stats::new(VERIFY_DIGEST_INPUT_BITS);
        verify_digest_circuit(&mut stats, &[Dummy; VERIFY_DIGEST_INPUT_BITS]);
        assert_eq!(
            stats.lean_stats(),
            LeanStats {
                m_rows: 1_215_663,
                m_cols: 1_215_663,
                r1cs_rows: 7_061,
            }
        );
    }

    #[test]
    fn fixed_width_division_matches_biguint() {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        for modulus in [Modulus::Base, Modulus::Scalar] {
            for _ in 0..1_000 {
                let mut words = [0; 18];
                for word in &mut words {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *word = state;
                }
                let value = Wide(words);
                let (quotient, remainder) = value.div_rem(modulus.words());
                let value = words_biguint(&words);
                let expected_quotient = &value / modulus.value();
                let expected_remainder = value % modulus.value();
                assert_eq!(words_biguint(&quotient.0), expected_quotient);
                assert_eq!(S::from(remainder), expected_remainder);
            }
        }
    }

    #[test]
    fn valid_signature_generates_the_exact_sized_witness() {
        let inputs = valid_input();
        let mut witness_only =
            WitnessOnly::with_inputs_and_capacity(inputs.as_ref(), VERIFY_DIGEST_WITNESS_BITS);
        verify_digest_circuit(&mut witness_only, &inputs);
        assert_eq!(witness_only.witness().bit_len(), VERIFY_DIGEST_WITNESS_BITS);

        let mut witgen =
            ProductWitgen::with_inputs_and_capacity(inputs.as_ref(), VERIFY_DIGEST_WITNESS_BITS);
        verify_digest_circuit(&mut witgen, &inputs);
        assert_eq!(witness_only.witness(), witgen.witness());
        assert_eq!(witgen.witness().bit_len(), VERIFY_DIGEST_WITNESS_BITS);
        assert_eq!(
            witgen.integer_witness().bit_len(),
            VERIFY_DIGEST_INTEGER_WITNESS_BITS
        );
        assert_eq!(witgen.products().a_mw.len(), VERIFY_DIGEST_R1CS_ROWS);
        assert_eq!(witgen.products().b_mw.len(), VERIFY_DIGEST_R1CS_ROWS);
        assert_eq!(witgen.products().c_mw.len(), VERIFY_DIGEST_R1CS_ROWS);
        for ((a, b), c) in witgen
            .products()
            .a_mw
            .iter()
            .zip(&witgen.products().b_mw)
            .zip(&witgen.products().c_mw)
        {
            assert_eq!(stored_bigint(a) * stored_bigint(b), stored_bigint(c));
        }
    }
}
