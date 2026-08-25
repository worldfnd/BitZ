use prove_playground::Mle;

fn main() {}

fn prove(input: Vec<Field>) {
    let circuit = Circuit::new(input);
    let witnesses = circuit.eval();
}

/// Grand product GKR
fn gpgkr_prove(eval: CircuitEval) {
    prove_layer()
}

fn prove_layer(
    point: &[Field],
    wnext: &[Field],
    challenges: &[Field],
) -> ((Field, Field), Vec<(Field, Field)>) {
    assert_eq!(wnext.len(), 2 << challenges.len());
    assert_eq!(point.len(), challenges.len());
    let mut suffix_table = SuffixTable::new(point);
    let mut factor = 1;

    let mut sumcheck_transcript = vec![];
    let mut mle_wnext = Mle::new(wnext);

    for (r, z) in challenges.iter().zip(point) {
        let eq = suffix_table.pop().unwrap();
        let mut sum_lo = 0;
        // let mut sum_hi = 0;
        let mut sum_inf = 0;
        let h = mle_wnext.len() / 2;

        for i in 0..eq.len() {
            let (l0, r0) = (mle_wnext[2 * i], mle_wnext[2 * i + 1]);
            let (l1, r1) = (mle_wnext[h + 2 * i], mle_wnext[h + 2 * i + 1]);
            sum_lo += eq[i] * l0 * r0;
            // sum_hi += eq[i] * l1 * r1;
            sum_inf += eq[i] * (l1 - l0) * (r1 - r0)
        }

        sumcheck_transcript.push((factor * sum_lo, factor * sum_inf));

        factor *= r * z + (1 - z) * (1 - r);
        mle_wnext.fix_variable(*r);
    }

    ((mle_wnext[0], mle_wnext[1]), sumcheck_transcript)
}

fn gpgkr_verify() {}

struct SuffixTable(Vec<Vec<Field>>);

impl SuffixTable {
    // This option might get annoying
    // Can be allocated as a single table directly.
    // Maybe vectors can be reused? This could be reused with sumcheck layer. The evaluation layer can be reused each gkr
    // Might be better in combination with split_eq table
    // currently has to be consumed in reverse order -> pop. Which is fine
    fn new(point: &[Field]) -> SuffixTable {
        // We do not need to build the table for the first challenge.
        let mut table = Vec::with_capacity(point.len());
        let mut prev = Vec::from([1]);

        for z in point.iter().skip(1).rev() {
            let size = prev.len() << 1;
            let mut entry = vec![0; size];
            let (low, hi) = entry.split_at_mut(size >> 1);

            // Does the lead to a range check?
            for (i, e) in prev.iter().enumerate() {
                let tmp = z * e;
                // (1-z)*e, z*e
                (low[i], hi[i]) = (e - tmp, tmp)
            }

            table.push(prev);
            prev = entry;
        }
        table.push(prev);
        SuffixTable(table)
    }

    fn pop(&mut self) -> Option<Vec<Field>> {
        self.0.pop()
    }
}

type Field = i32;

// A circuit is defined by it's leaf value only because it is a balanced tree
// TODO: Optimise for circuits that are padded.
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
// TODO: This can be a single vec that is allocated at the start.
// The vec vec structure only has to come back for a padded implementation.
// There can be a traded off between space usage and computation reuse. This one choses speed for memory.
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
