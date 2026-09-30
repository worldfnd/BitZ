//! The protocol parameters: everything both sides fix before a claim exists.

use crate::{BitTable, BitzClaimField, Shape, TableError, VirtualMap};
use field::{
    F128, FqDefault, MAX_MODULUS_BITS,
    gf128::{MULT_ORDER, is_generator},
};
use num_traits::{CheckedMul, ToBytes};
use spongefish::Encoding;
use std::marker::PhantomData;

/// A parameter set one of the pre-claim gates rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsError {
    /// `(k_1 + 1)(Q - 1)` reaches `ord(g)`, so a sent fold and the honest
    /// exponent could collide in the exponent.
    FoldBoundExceeded,
    /// The generator's order is not the full group, so a fold is not the only
    /// exponent producing its image.
    GeneratorOrderNotFull,
    /// The fold gate leaves the fingerprint prime narrower than
    /// [`MIN_PRIME_BITS`]: the shape is too tall.
    PrimeTooNarrow,
}

/// The narrowest fingerprint prime a proof is run over: the width of the
/// default modulus. The PIOP's soundness error is of order `1/q`, so no shape
/// may push the prime below what the fixed modulus gave.
pub const MIN_PRIME_BITS: u32 = FqDefault::BITS;

/// Width of the fingerprint prime for `shape`: the largest `bits` such that
/// every prime in `[2^(bits-1), 2^bits)` passes the fold gate of
/// [`BitZParams::new`], capped by the field's [`MAX_MODULUS_BITS`].
///
/// The prime is drawn from this interval by both sides, so the width must
/// be fixed by the shape.
///
/// Wider is sounder, so the interval is the widest the gate admits. The gate
/// is the paper's `q < (|K| - 1) / k_1`, and with `k_1 = 2^log_rows` that is
/// exactly `128 - log_rows` bits.
pub fn prime_bits(shape: &Shape) -> Result<u32, ParamsError> {
    // `rows * q < ord(g)` admits `q <= (ord(g) - 1) / rows`.
    let max_modulus = (MULT_ORDER - 1) / shape.rows() as u128;
    // `2^bits - 1 <= max_modulus < 2^(bits + 1) - 1`.
    let bits = (max_modulus + 1).ilog2().min(MAX_MODULUS_BITS);
    if bits < MIN_PRIME_BITS {
        return Err(ParamsError::PrimeTooNarrow);
    }
    Ok(bits)
}

/// The shape, the modulus and the generator: what a proof is fixed against.
///
/// The two roles derive their own setups from this, so neither can be built
/// against parameters the other did not see.
///
/// The modulus is not a value here, it's accessible as `F::modulus()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitZParams<F> {
    shape: Shape,
    generator: F128,
    _phantom: PhantomData<F>,
}

impl<F: BitzClaimField> BitZParams<F> {
    /// Runs the gates that need only the parameters.
    pub fn new(shape: Shape, generator: F128) -> Result<Self, ParamsError> {
        // The field asserts its modulus is an odd prime below 2^126 on its own
        // behalf, so no modulus gate is needed here.

        // `g^{<f_j, gamma>} = g^{eta_j}` implies integer equality only if both
        // sides, each in `[0, k_1 (Q - 1)]`, differ by less than `ord(g)`.
        // The bound is the paper's `Q < (|K| - 1) / k_1`, under which Round 1
        // of 4.3. "The BitZ IOP for the core LinBitsRings relation" is
        // skipped: `k_1 Q < ord(g)`. With `k_1` a power of two it admits every
        // `(128 - log_rows)`-bit prime, so a prime of the width [`prime_bits`]
        // derives always passes.
        //
        // `ord(g) = 2^128 - 1` is a property of `F128`, established by the
        // generator gate below. A product that overflows exceeds it too.
        let f128_order = F::Integer::from(MULT_ORDER);
        let rows = F::Integer::from(shape.rows() as u64);
        if !rows
            .checked_mul(&F::modulus())
            .is_some_and(|reach| reach < f128_order)
        {
            return Err(ParamsError::FoldBoundExceeded);
        }
        if !is_generator(generator) {
            return Err(ParamsError::GeneratorOrderNotFull);
        }

        Ok(Self {
            shape,
            generator,
            _phantom: PhantomData,
        })
    }

    /// Views a packed witness through the configured shape.
    ///
    /// The only way to build a [`BitTable`], so a table can never be shaped by
    /// anything but a checked parameter set.
    pub fn table<'a>(&self, packed: &'a [F128]) -> Result<BitTable<'a>, TableError> {
        BitTable::new(self.shape, packed)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn generator(&self) -> F128 {
        self.generator
    }

    /// The largest fold the verifier may accept, `k_1 (Q - 1)`.
    ///
    /// [`Self::new`]'s gate puts it below `ord(g)`.
    pub fn fold_bound(&self) -> F::Integer {
        let rows = u64::try_from(self.shape.rows()).expect("Too many rows");
        let rows = F::Integer::from(rows);
        let max_f = F::max_value().lift();
        rows.checked_mul(&max_f).expect("Multiplication overflow")
    }
}

/// Every field is fixed width, the modulus at `F::Integer`'s, so distinct
/// parameter sets cannot encode alike.
impl<F: BitzClaimField> Encoding<[u8]> for BitZParams<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let modulus = F::modulus().to_le_bytes();
        let mut frame = Vec::with_capacity(32 + modulus.as_ref().len());
        frame.extend_from_slice(&(self.shape.log_rows() as u64).to_le_bytes());
        frame.extend_from_slice(&(self.shape.log_columns() as u64).to_le_bytes());
        frame.extend_from_slice(modulus.as_ref());
        frame.extend_from_slice(&self.generator.to_bytes());
        frame
    }
}

/// A parameter set one of the pre-claim gates rejects when the claim is about
/// a vector the oracle does not commit to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualParamsError {
    /// The map has no column for the constant-one coordinate.
    MissingConstantColumn,
    /// `h` has more coordinates than the claim shape indexes.
    ClaimShapeTooSmall,
    /// The map declares more bits of `f` than the committed shape holds.
    CommittedShapeTooSmall,
}

/// Parameters for a claim about `h` opened against a commitment to `f`.
///
/// Two shapes, because the two vectors have different lengths. The inherited
/// [`BitZParams`] shapes `h`: it is what the fold walks, what bounds the
/// exponent, and what the claim's weights are counted against. The committed
/// shape belongs to `f` alone and reaches only the table and the opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualParams<F> {
    claim: BitZParams<F>,
    committed: Shape,
}

impl<F: BitzClaimField> VirtualParams<F> {
    /// Checks both shapes against the map before either is used.
    ///
    /// Zero padding is what makes the inequalities rather than equalities: both
    /// vectors are padded up to their shape so that they have multilinear
    /// extensions, and padding contributes nothing.
    pub fn new(
        claim: BitZParams<F>,
        committed: Shape,
        map: &impl VirtualMap,
    ) -> Result<Self, VirtualParamsError> {
        if map.h_len() > 1 << claim.shape().log_bits() {
            return Err(VirtualParamsError::ClaimShapeTooSmall);
        }
        let committed_bits = map
            .f_len()
            .checked_sub(1)
            .ok_or(VirtualParamsError::MissingConstantColumn)?;
        if committed_bits > 1 << committed.log_bits() {
            return Err(VirtualParamsError::CommittedShapeTooSmall);
        }

        Ok(Self { claim, committed })
    }

    /// The parameters the fold and the reduction read, shaped to `h`.
    pub fn claim(&self) -> &BitZParams<F> {
        &self.claim
    }

    /// The shape of the committed bits.
    pub fn committed_shape(&self) -> &Shape {
        &self.committed
    }

    /// Views the committed witness, which is `f` and not the vector the claim
    /// is about.
    pub fn table<'a>(&self, packed: &'a [F128]) -> Result<BitTable<'a>, TableError> {
        BitTable::new(self.committed, packed)
    }
}

/// Distinct from a plain [`BitZParams`] frame by length, so a proof of one
/// cannot replay as a proof of the other.
impl<F: BitzClaimField> Encoding<[u8]> for VirtualParams<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut frame = self.claim.encode().as_ref().to_vec();
        frame.extend_from_slice(&(self.committed.log_rows() as u64).to_le_bytes());
        frame.extend_from_slice(&(self.committed.log_columns() as u64).to_le_bytes());
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::gf128::smallest_generator;
    use num_traits::{ConstOne, ConstZero};

    /// The largest prime below `2^114`, the top of the sampling range.
    const Q114: u128 = (1 << 114) - 11;
    type F = field::Fq<Q114>;

    fn params_at(shape: Shape) -> Result<BitZParams<F>, ParamsError> {
        BitZParams::new(shape, smallest_generator())
    }

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    /// A map of stated dimensions and nothing else, which is all the gates read.
    struct Dimensions {
        h_len: usize,
        f_len: usize,
    }

    impl VirtualMap for Dimensions {
        fn transpose(
            &self,
            _: &[F128],
        ) -> Result<crate::TransposedWeights, crate::VirtualMapError> {
            unreachable!("the parameter gates never apply the map")
        }

        fn digest(&self) -> [u8; 32] {
            [0; 32]
        }

        fn h_len(&self) -> usize {
            self.h_len
        }

        fn f_len(&self) -> usize {
            self.f_len
        }
    }

    fn virtual_params_for(
        h_len: usize,
        f_len: usize,
    ) -> Result<VirtualParams<F>, VirtualParamsError> {
        VirtualParams::new(
            params_at(shape()).unwrap(),
            shape(),
            &Dimensions { h_len, f_len },
        )
    }

    #[test]
    fn accepts_vectors_the_shapes_have_room_for() {
        // Both shapes index `2^22` coordinates; anything shorter is padded.
        assert!(virtual_params_for(1 << 22, (1 << 22) + 1).is_ok());
        assert!(virtual_params_for(3, 4).is_ok());
    }

    #[test]
    fn rejects_a_claim_vector_wider_than_its_shape() {
        assert_eq!(
            virtual_params_for((1 << 22) + 1, 4).err(),
            Some(VirtualParamsError::ClaimShapeTooSmall)
        );
    }

    #[test]
    fn rejects_committed_bits_wider_than_their_shape() {
        assert_eq!(
            virtual_params_for(4, (1 << 22) + 2).err(),
            Some(VirtualParamsError::CommittedShapeTooSmall)
        );
    }

    #[test]
    fn rejects_a_map_without_the_constant_column() {
        assert_eq!(
            virtual_params_for(4, 0),
            Err(VirtualParamsError::MissingConstantColumn)
        );
    }

    /// A shared frame would let a proof of one claim replay as a proof of the
    /// other.
    #[test]
    fn the_two_parameter_frames_cannot_collide() {
        let plain = params_at(shape()).unwrap();
        let virtualised = virtual_params_for(4, 4).unwrap();

        assert_ne!(
            plain.encode().as_ref().len(),
            virtualised.encode().as_ref().len()
        );
    }

    /// The committed shape, not the claim's, is what the witness is read
    /// through.
    #[test]
    fn the_table_is_shaped_by_the_committed_bits() {
        let claim = BitZParams::<F>::new(Shape::new(8, 15).unwrap(), smallest_generator()).unwrap();
        let committed = shape();
        let params =
            VirtualParams::new(claim, committed, &Dimensions { h_len: 4, f_len: 4 }).unwrap();
        let packed = vec![F128::ZERO; (1 << committed.log_bits()) / 128];

        assert_eq!(params.table(&packed).unwrap().shape(), &committed);
    }

    #[test]
    fn rejects_a_shape_the_modulus_is_too_large_for() {
        // `t = 14` is the widest row count this prime admits: `2^14 Q114` is
        // `2^128 - 11 * 2^14`, under `ord(g)`; `2^15 Q114` is not.
        assert!(params_at(Shape::new(14, 21).unwrap()).is_ok());
        assert_eq!(
            params_at(Shape::new(15, 20).unwrap()).err(),
            Some(ParamsError::FoldBoundExceeded)
        );
    }

    #[test]
    fn rejects_a_generator_of_partial_order() {
        assert_eq!(
            BitZParams::<F>::new(shape(), F128::ONE).err(),
            Some(ParamsError::GeneratorOrderNotFull)
        );
    }

    /// The gate of [`BitZParams::new`] at the top of a `bits`-bit interval.
    fn gate_admits(bits: u32, shape: &Shape) -> bool {
        let top = (1u128 << bits) - 1;
        top.checked_mul(shape.rows() as u128)
            .is_some_and(|reach| reach < MULT_ORDER)
    }

    /// One shape per admissible row count.
    fn shapes_by_rows() -> impl Iterator<Item = Shape> {
        use crate::shape::{MAX_LOG_BITS, MIN_LOG_BITS, PACK_BITS};
        (PACK_BITS as usize..=MAX_LOG_BITS)
            .map(|log_rows| Shape::new(log_rows, MIN_LOG_BITS.saturating_sub(log_rows)).unwrap())
    }

    #[test]
    fn the_prime_width_is_the_widest_interval_the_gate_admits() {
        for shape in shapes_by_rows() {
            match prime_bits(&shape) {
                Ok(bits) => {
                    assert!(bits >= MIN_PRIME_BITS, "{shape:?}");
                    assert!(gate_admits(bits, &shape), "{shape:?}");
                    assert!(
                        bits == MAX_MODULUS_BITS || !gate_admits(bits + 1, &shape),
                        "{shape:?}"
                    );
                }
                Err(error) => {
                    assert_eq!(error, ParamsError::PrimeTooNarrow, "{shape:?}");
                    assert!(!gate_admits(MIN_PRIME_BITS, &shape), "{shape:?}");
                }
            }
        }
    }

    #[test]
    fn the_prime_width_at_the_row_counts_in_use() {
        for (log_rows, bits) in [(7, 121), (14, 114), (21, 107), (28, 100)] {
            let shape = Shape::new(log_rows, 22usize.saturating_sub(log_rows)).unwrap();
            assert_eq!(prime_bits(&shape), Ok(bits), "{log_rows}");
        }
        let shape = Shape::new(29, 0).unwrap();
        assert_eq!(prime_bits(&shape), Err(ParamsError::PrimeTooNarrow));
    }

    /// `Q114` is admitted by exactly the shapes whose width reaches it.
    #[test]
    fn the_prime_width_agrees_with_the_gate() {
        for shape in shapes_by_rows() {
            let reaches = prime_bits(&shape).is_ok_and(|bits| bits >= F::BITS);
            assert_eq!(params_at(shape).is_ok(), reaches, "{shape:?}");
        }
    }

    #[test]
    fn the_fold_bound_is_the_widest_column_sum() {
        let params = params_at(shape()).unwrap();
        assert_eq!(params.fold_bound(), 128 * (Q114 - 1));
    }

    #[test]
    fn the_encoding_covers_every_parameter_and_no_derived_value() {
        let params = params_at(shape()).unwrap();
        let encoded = params.encode();
        let encoded = encoded.as_ref();

        assert_eq!(encoded.len(), 48);
        assert_eq!(&encoded[..8], &7u64.to_le_bytes());
        assert_eq!(&encoded[8..16], &15u64.to_le_bytes());
        assert_eq!(&encoded[16..32], &Q114.to_le_bytes());
        assert_eq!(&encoded[32..], &smallest_generator().to_bytes());
    }

    #[test]
    fn a_different_parameter_encodes_differently() {
        let narrow = params_at(shape()).unwrap();
        let wide = params_at(Shape::new(13, 9).unwrap()).unwrap();

        assert_ne!(
            narrow.encode().as_ref().to_vec(),
            wide.encode().as_ref().to_vec()
        );
    }
}
