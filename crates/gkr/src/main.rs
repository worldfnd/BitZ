use prove_playground::Mle;

fn main() {}

fn prove(input: Vec<Field>) {
    let circuit = Circuit::new(input);
    let witnesses = circuit.eval();
    gpgkr_prove(_, _, witnesses);
}

// TODO capture the reverse running through eval in it's own iter

/// Grand product GKR
fn gpgkr_prove(
    // Captured by a transcript implementation
    challenges: Vec<Vec<Field>>,
    line_challenges: Vec<Field>,
    //
    mut eval: CircuitEval,
) -> Vec<((Field, Field), Vec<(Field, Field)>)> {
    let mut transcripts = vec![];
    let mut point = vec![];
    let _last_value = eval.pop().unwrap();
    for ((wnext, challenge_layer), line_challenge) in
        eval.0.iter().rev().zip(challenges).zip(line_challenges)
    {
        let (line, transcript) = prove_layer(&point, wnext, &challenge_layer);
        transcripts.push((line, transcript));

        point = challenge_layer;
        point.push(line_challenge);
    }
    transcripts
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
    // TODO: wnext is taken by reference, so this clones the whole (largest) table.
    // Taking `wnext: Vec<Field>` by value and moving it in would avoid the copy.
    let mut mle_wnext = Mle::new(wnext);

    for (r, z) in challenges.iter().zip(point) {
        // Last table to be popped is just 1. Feels like that can be optimised. Would save two multiplications.
        // TODO: special-case eq.len() == 1 (final round) to skip the `eq[i] *`
        // multiplications below entirely.
        let eq = suffix_table.pop().unwrap();
        let mut sum_0 = 0;
        let mut sum_inf = 0;
        let h = mle_wnext.len() / 2; // Same as eq.len()?

        // Could this loop be combined with fix_variable of r*?
        // TODO: yes — this loop and mle_wnext.fix_variable(*r) below both read
        // l0/r0/l1/r1 for every gate, i.e. two full passes over mle_wnext per
        // round. Fuse them: compute the folded values
        // (l0 + r*(l1-l0), r0 + r*(r1-r0)) right here and write into a
        // half-sized buffer, then drop the separate fix_variable call.
        //
        // TODO: independent across i, so this loop is embarrassingly
        // parallel — a rayon par_iter + reduction would scale it across cores.
        for i in 0..eq.len() {
            // Two sequential cache access lines
            // TODO: mle_wnext[2*i..] and mle_wnext[h+2*i..] are h elements
            // apart, so every iteration bounces between two distant cache
            // lines. Inherent to the split-at-midpoint MLE layout
            // (Mle::fix_variable splits at n/2) — fixing it means changing
            // the table layout, not just this loop.
            let (l0, r0) = (mle_wnext[2 * i], mle_wnext[2 * i + 1]);
            let (l1, r1) = (mle_wnext[h + 2 * i], mle_wnext[h + 2 * i + 1]);
            sum_0 += eq[i] * l0 * r0;
            sum_inf += eq[i] * (l1 - l0) * (r1 - r0)
        }

        sumcheck_transcript.push((factor * sum_0, factor * sum_inf));

        factor *= r * z + (1 - z) * (1 - r);
        mle_wnext.fix_variable(*r);
    }

    ((mle_wnext[0], mle_wnext[1]), sumcheck_transcript)
}

fn verify_layer(check_value: Field, point: &[Field], challenges: &[Field]) {}

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
