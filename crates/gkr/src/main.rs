use prove_playground::*;

fn main() {}

fn prove(input: Vec<Field>) {
    let circuit = Circuit::new(input);
    let witnesses = circuit.eval();

    // gpgkr_prove samples one more challenge per layer than the last
    // (0, 1, 2, ... over `m` layers), so the total is the triangular
    // number m*(m+1)/2.
    let m = circuit.leafs.len().ilog2() as usize;
    let c = Challenge::with_capacity(m * (m + 1) / 2);

    gpgkr_prove(&c, witnesses);
}

/// Grand product GKR
///
/// Returns two preallocated buffers that are only ever appended to (no
/// reallocation once the loop below starts):
/// - `round01[i]` is layer `i`'s terminal `(w0, w1)` pair, one per layer.
/// - `sumcheck` is every layer's sumcheck transcript back to back, widening
///   as it goes: layer `i`'s point has `i` challenges, so it contributes `i`
///   entries, right after layer `i - 1`'s `i - 1` entries. Total size is the
///   triangular number `m * (m - 1) / 2`.
fn gpgkr_prove(c: &Challenge, mut eval: CircuitEval) -> (Vec<(Field, Field)>, Vec<(Field, Field)>) {
    let _last_value = eval.pop().unwrap();
    let m = eval.len();

    let mut round01 = Vec::with_capacity(m);
    let mut sumcheck = Vec::with_capacity(m.saturating_sub(1) * m / 2);

    let mut point = c.new_frame();
    for wnext in eval.iter() {
        round01.push(prove_layer(&point, wnext, c, &mut sumcheck));

        // sample extra challenge for the next round
        let _ = c.get_challenge();
        point = c.new_frame();
    }
    (round01, sumcheck)
}

fn prove_layer(
    point: &[Field],
    wnext: &[Field],
    c: &Challenge,
    sumcheck: &mut Vec<(Field, Field)>,
) -> (Field, Field) {
    let mut suffix_table = SuffixTable::new(point);
    let mut factor = 1;

    // TODO: wnext is taken by reference, so this clones the whole (largest) table.
    // Taking `wnext: Vec<Field>` by value and moving it in would avoid the copy.
    let mut mle_wnext = Mle::new(wnext);

    for z in point {
        let r = c.get_challenge();
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

        sumcheck.push((factor * sum_0, factor * sum_inf));

        factor *= r * z + (1 - z) * (1 - r);
        mle_wnext.fix_variable(r);
    }

    (mle_wnext[0], mle_wnext[1])
}

fn gpgkr_verify(
    final_value: Field,
    circuit: Circuit,
    challenges: Vec<Vec<Field>>,
    round01: Vec<(Field, Field)>,
    sumcheck: Vec<(Field, Field)>,
) {
    let mut cursor = 0;
    for (i, (c, (w0, w1))) in challenges.iter().zip(round01).enumerate() {
        // Layer i's point has i challenges, so its slice of `sumcheck` is i
        // entries wide, right after layer i - 1's slice.
        let t = &sumcheck[cursor..cursor + i];
        cursor += i;

        // split c in init and last
        // The init is for checking the transscript.
        // The values in the transcript are s0 and sinf so these need to work with final value
        //
        // The sumcheck if for checking the sum. And the w0 and w1 are the new claims that need to be checked. Which the next step will combine into a single one.
        //
        // w0 and w1 are the terminal values / leaves that remain after folding the wnext mle
        // w0 = w_next(challenges, 0) | w1 = w_next(challenges, 1)
        // last together with these values make for the next claim.
    }
}

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

    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> impl Iterator<Item = &Vec<Field>> {
        self.0.iter().rev()
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
