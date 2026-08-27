//! The protocol parameters: everything both sides fix before a claim exists.

use field::{F128, gf128::is_generator};
use spongefish::Encoding;

use crate::{BitTable, Shape, TableError};

/// `log2 |K|`. The extension is `F_2^128`, so a fold has 128 bits of room.
const EXTENSION_BITS: u32 = 128;

/// A parameter set one of the pre-claim gates rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsError {
    /// `Q` is at or above `|K| / k_1`, so two folds could collide in the
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
pub struct F2ZParams<const Q: u128> {
    shape: Shape,
    generator: F128,
}

impl<const Q: u128> F2ZParams<Q> {
    /// Runs the gates that need only the parameters.
    pub fn new(shape: Shape, generator: F128) -> Result<Self, ParamsError> {
        // `Fq<Q>` asserts Q is an odd prime below 2^126 on its own behalf, so
        // no modulus gate is needed here.

        // `Q < |K| / k_1`, equivalently `t + log2 Q < 128`: a fold is at most
        // `k_1 (Q - 1)` and is only ever seen modulo `ord(g)`, so it is unique
        // exactly while the range fits inside the group. A shift because
        // `2^128` does not fit a u128; `t >= 7` keeps it in range.
        if Q >> (EXTENSION_BITS - shape.t() as u32) != 0 {
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
impl<const Q: u128> Encoding<[u8]> for F2ZParams<Q> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut frame = [0u8; 48];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            frame[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };

        put(&(self.shape.t() as u64).to_le_bytes());
        put(&(self.shape.s() as u64).to_le_bytes());
        put(&Q.to_le_bytes());
        put(&self.generator.to_bytes());
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::gf128::smallest_generator;
    use num_traits::ConstOne;

    /// The largest prime below `2^114`, the top of the sampling range.
    const Q114: u128 = (1 << 114) - 11;

    fn params_at(shape: Shape) -> Result<F2ZParams<Q114>, ParamsError> {
        F2ZParams::new(shape, smallest_generator())
    }

    /// `m = 22`: 128 rows per column, 32768 columns.
    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    #[test]
    fn rejects_a_shape_the_modulus_is_too_large_for() {
        // `t = 14` is the widest row count this prime admits; 15 is not.
        assert!(params_at(Shape::new(14, 21).unwrap()).is_ok());
        assert_eq!(
            params_at(Shape::new(15, 20).unwrap()).err(),
            Some(ParamsError::FoldBoundExceeded)
        );
    }

    #[test]
    fn rejects_a_generator_of_partial_order() {
        assert_eq!(
            F2ZParams::<Q114>::new(shape(), F128::ONE).err(),
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
        let wide = params_at(Shape::new(14, 8).unwrap()).unwrap();

        assert_ne!(
            narrow.encode().as_ref().to_vec(),
            wide.encode().as_ref().to_vec()
        );
    }
}
