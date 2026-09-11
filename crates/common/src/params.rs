//! The protocol parameters: everything both sides fix before a claim exists.

use field::{F128, gf128::is_generator};
use spongefish::Encoding;

use crate::{BitTable, Shape, TableError, VirtualMap};

/// A parameter set one of the pre-claim gates rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsError {
    /// `(k_1 + 1)(Q - 1)` reaches `ord(g)`, so two folds could collide in the
    /// exponent.
    FoldBoundExceeded,
    /// The generator's order is not the full group, so a fold is not the only
    /// exponent producing its image.
    GeneratorOrderNotFull,
}

/// The shape, the modulus and the generator: what a proof is fixed against.
///
/// The two roles derive their own setups from this, so neither can be built
/// against parameters the other did not see.
///
/// `Q` is a const parameter, not a field: the weights are `Fq<Q>`, whose
/// modulus lives in the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitZParams<const Q: u128> {
    shape: Shape,
    generator: F128,
}

impl<const Q: u128> BitZParams<Q> {
    /// Runs the gates that need only the parameters.
    pub fn new(shape: Shape, generator: F128) -> Result<Self, ParamsError> {
        // `Fq<Q>` asserts Q is an odd prime below 2^126 on its own behalf, so
        // no modulus gate is needed here.

        // `ord(g) > (k_1 + 1)(Q - 1)`, the paper's requisite. A fold is an
        // integer at most `k_1 (Q - 1)` while the value it is compared against
        // is at most `Q - 1`, so the two differ by at most the sum; the
        // exponent is only ever seen modulo `ord(g)`, and a gap that never
        // reaches the group order cannot close.
        //
        // Overflow is itself a rejection: past `2^128 - 1` there is no room
        // left. `ord(g)` is `u128::MAX`, the full order the generator gate
        // below establishes.
        let Some(gap) = (Q - 1).checked_mul(shape.rows() as u128 + 1) else {
            return Err(ParamsError::FoldBoundExceeded);
        };
        // A `u128` cannot exceed `ord(g)`, so equalling it is the only way
        // left to reach it.
        if gap == u128::MAX {
            return Err(ParamsError::FoldBoundExceeded);
        }
        if !is_generator(generator) {
            return Err(ParamsError::GeneratorOrderNotFull);
        }

        Ok(Self { shape, generator })
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
    pub fn fold_bound(&self) -> u128 {
        (self.shape.rows() as u128) * (Q - 1)
    }
}

/// Every field is fixed width, so distinct parameter sets cannot encode alike.
impl<const Q: u128> Encoding<[u8]> for BitZParams<Q> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut frame = [0u8; 48];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            frame[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };

        put(&(self.shape.log_rows() as u64).to_le_bytes());
        put(&(self.shape.log_columns() as u64).to_le_bytes());
        put(&Q.to_le_bytes());
        put(&self.generator.to_bytes());
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
    /// `f` has more coordinates than the committed shape indexes, so some
    /// bit the map reads was never committed.
    CommittedShapeTooSmall,
}

/// Parameters for a claim about `h` opened against a commitment to `f`.
///
/// Two shapes, because the two vectors have different lengths. The inherited
/// [`BitZParams`] shapes `h`: it is what the fold walks, what bounds the
/// exponent, and what the claim's weights are counted against. The committed
/// shape belongs to `f` alone and reaches only the table and the opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualParams<const Q: u128> {
    claim: BitZParams<Q>,
    committed: Shape,
}

impl<const Q: u128> VirtualParams<Q> {
    /// Checks both shapes against the map before either is used.
    ///
    /// Zero padding is what makes the inequalities rather than equalities: both
    /// vectors are padded up to their shape so that they have multilinear
    /// extensions, and padding contributes nothing.
    pub fn new(
        claim: BitZParams<Q>,
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
    pub fn claim(&self) -> &BitZParams<Q> {
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
impl<const Q: u128> Encoding<[u8]> for VirtualParams<Q> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut frame = [0u8; 64];
        frame[..48].copy_from_slice(self.claim.encode().as_ref());
        frame[48..56].copy_from_slice(&(self.committed.log_rows() as u64).to_le_bytes());
        frame[56..].copy_from_slice(&(self.committed.log_columns() as u64).to_le_bytes());
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

    fn params_at(shape: Shape) -> Result<BitZParams<Q114>, ParamsError> {
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
    ) -> Result<VirtualParams<Q114>, VirtualParamsError> {
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
        let claim =
            BitZParams::<Q114>::new(Shape::new(8, 15).unwrap(), smallest_generator()).unwrap();
        let committed = shape();
        let params =
            VirtualParams::new(claim, committed, &Dimensions { h_len: 4, f_len: 4 }).unwrap();
        let packed = vec![F128::ZERO; (1 << committed.log_bits()) / 128];

        assert_eq!(params.table(&packed).unwrap().shape(), &committed);
    }

    #[test]
    fn rejects_a_shape_the_modulus_is_too_large_for() {
        // `t = 13` is the widest row count this prime admits; 14 is not. The
        // `+1` is what separates the two: `k_1 (Q - 1)` alone would still fit
        // at `t = 14`.
        assert!(params_at(Shape::new(13, 22).unwrap()).is_ok());
        assert_eq!(
            params_at(Shape::new(14, 21).unwrap()).err(),
            Some(ParamsError::FoldBoundExceeded)
        );
    }

    #[test]
    fn rejects_a_generator_of_partial_order() {
        assert_eq!(
            BitZParams::<Q114>::new(shape(), F128::ONE).err(),
            Some(ParamsError::GeneratorOrderNotFull)
        );
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
