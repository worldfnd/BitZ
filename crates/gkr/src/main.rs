fn main() {}

type Field = i32;

// A circuit is defined by it's leaf value only because it is a balanced tree
struct Circuit {
    leafs: Vec<Field>,
}

impl Circuit {
    fn new(mut leafs: Vec<Field>) -> Self {
        leafs.resize(leafs.len().next_power_of_two(), 1);

        Circuit { leafs: leafs }
    }

    // Should an evaluation consume?
    fn eval(&self) -> CircuitEval {
        // Size can be precalculated
        let mut witnesses = Vec::with_capacity(self.leafs.len().ilog2() as usize);

        let mut prev_eval = self.leafs.clone();

        while prev_eval.len() > 1 {
            let mut eval = Vec::with_capacity(prev_eval.len() >> 1);
            for pair in prev_eval.chunks_exact(2) {
                eval.push(pair[0] * pair[1])
            }
            witnesses.push(prev_eval);
            prev_eval = eval;
        }

        witnesses.push(prev_eval);

        CircuitEval(witnesses)
    }
}

// Main purpose of the wrapper is to have the order denoted
struct CircuitEval(Vec<Vec<Field>>);

impl CircuitEval {
    // Return from the final layer back to the front
    // If circuit evaluation needs to stay an alternative is to wrap it in an iterator to keep track of the location
    fn pop(&mut self) -> Option<Vec<Field>> {
        self.0.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // Pairwise multiplication is associative, so folding-by-multiply any
        // witness layer (including the leaves, padded with the multiplicative
        // identity 1) must equal folding the original input.
        #[test]
        fn eval_preserves_product_across_layers(leaves in prop::collection::vec(-3i32..=3, 0..12)) {
            let expected = leaves.iter().fold(1, |acc, &x| acc * x);

            let mut eval = Circuit::new(leaves).eval();

            let mut layers_checked = 0;
            while let Some(layer) = eval.pop() {
                let folded = layer.iter().fold(1, |acc, &x| acc * x);
                prop_assert_eq!(folded, expected);
                layers_checked += 1;
            }
            prop_assert!(layers_checked > 0);
        }
    }
}
