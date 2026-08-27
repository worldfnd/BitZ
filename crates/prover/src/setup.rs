//! The prover's derived setup.

use common::F2ZParams;
use field::FixedBasePow;

/// The parameters, with the comb table derived from them.
///
/// Deriving the comb here is what removes the pairing check the two used to
/// need: there is no second generator for the comb to disagree with.
#[derive(Debug)]
pub struct F2ZProver<const Q: u128> {
    params: F2ZParams<Q>,
    comb: FixedBasePow,
}

impl<const Q: u128> F2ZProver<Q> {
    /// Derives the comb over the parameters' generator.
    ///
    /// `window` trades the comb's size against multiplies per exponentiation.
    ///
    /// # Panics
    ///
    /// [`FixedBasePow::new`] requires `window` in `1..=16`.
    pub fn new(params: F2ZParams<Q>, window: u32) -> Self {
        let comb = FixedBasePow::new(params.generator(), window);
        Self { params, comb }
    }

    pub fn params(&self) -> &F2ZParams<Q> {
        &self.params
    }

    pub(crate) fn comb(&self) -> &FixedBasePow {
        &self.comb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::Shape;
    use field::gf128::smallest_generator;

    const Q114: u128 = (1 << 114) - 11;

    #[test]
    fn the_comb_is_built_on_the_generator_the_parameters_name() {
        // `pow(1)` reads the base straight out of the comb.
        let params =
            F2ZParams::<Q114>::new(Shape::new(7, 15).unwrap(), smallest_generator()).unwrap();
        let setup = F2ZProver::new(params, 8);

        assert_eq!(setup.comb().pow(1), params.generator());
    }
}
