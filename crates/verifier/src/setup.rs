//! The verifier's derived setup.

use common::F2ZParams;
use field::FixedBasePow;

/// The parameters, with the comb table and the fold bound derived from them.
///
/// The bound is precomputed here rather than on the parameters because the
/// verifier is its only reader: the prover never range-checks a fold it
/// produced itself.
#[derive(Debug)]
pub struct F2ZVerifier<const Q: u128> {
    params: F2ZParams<Q>,
    comb: FixedBasePow,
    fold_bound: u128,
}

impl<const Q: u128> F2ZVerifier<Q> {
    /// Derives the comb over the parameters' generator, and the fold bound.
    ///
    /// # Panics
    ///
    /// [`FixedBasePow::new`] requires `window` in `1..=16`.
    pub fn new(params: F2ZParams<Q>, window: u32) -> Self {
        let comb = FixedBasePow::new(params.generator(), window);
        let fold_bound = params.fold_bound();
        Self {
            params,
            comb,
            fold_bound,
        }
    }

    pub fn params(&self) -> &F2ZParams<Q> {
        &self.params
    }

    pub(crate) fn comb(&self) -> &FixedBasePow {
        &self.comb
    }

    /// The largest fold this verifier accepts, `k_1 (Q - 1)`.
    pub fn fold_bound(&self) -> u128 {
        self.fold_bound
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
        let setup = F2ZVerifier::new(params, 8);

        assert_eq!(setup.comb().pow(1), params.generator());
    }
}
