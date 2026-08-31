use prove_playground::*;

fn main() {}

fn prove(input: Vec<Field>, groups: usize) {
    let circuit = Circuit::new(input);
    let witnesses = circuit.batched_eval(groups);

    // gpgkr_prove samples one more challenge per layer than the last
    // (0, 1, 2, ... over `m` layers), so the total is the triangular
    // number m*(m+1)/2.
    let m = circuit.leafs.len().ilog2() as usize;
    let c = Challenge::with_capacity(m, groups);

    gpgkr_prove(groups, &c, witnesses);
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
fn gpgkr_prove(
    groups: usize,
    c: &Challenge,
    mut eval: CircuitEval,
) -> (Vec<(Field, Field)>, TriangularArray<(Field, Field)>) {
    let _last_value = eval.pop().unwrap();
    let m = eval.len();

    let mut round01 = Vec::with_capacity(m);
    let mut sumcheck = TriangularArray::with_capacity(m.saturating_sub(1), groups);

    let first_point = [];
    let mut point: Point<'_, Field> = Point::new(&first_point);
    for (i, wnext) in eval.into_iter().enumerate() {
        round01.push(prove_layer(point, wnext, c, &mut sumcheck));

        // sample extra challenge for the next round
        let _ = c.get_challenge();
        point = c.build_point(i);
    }
    (round01, sumcheck)
}

fn prove_layer(
    point: Point<Field>,
    mut wnext: Vec<Field>,
    c: &Challenge,
    sumcheck: &mut TriangularArray<(Field, Field)>,
) -> (Field, Field) {
    let mut suffix_table = SuffixTable::new(point);
    let mut factor = 1;

    let mid = wnext.len() / 2;
    let (mut mle_l, mut mle_r) = wnext.split_at_mut(mid);

    for z in point {
        let r = c.get_challenge();
        // Last table to be popped is just 1. Feels like that can be optimised. Would save two multiplications.
        // TODO: special-case eq.len() == 1 (final round) to skip the `eq[i] *`
        // multiplications below entirely.
        let eq = suffix_table.pop().unwrap();
        let mut sum_0 = 0;
        let mut sum_inf = 0;
        let h = mle_l.len() / 2; // Same as mle.next/2?

        // Fused loop of TODO find different way of writing this
        for i in 0..eq.len() {
            // Two sequential cache access lines
            let (l0, r0) = (mle_l[i], mle_r[i]);
            let (l1, r1) = (mle_l[i + h], mle_r[i + h]);
            sum_0 += eq[i] * l0 * r0;
            sum_inf += eq[i] * (l1 - l0) * (r1 - r0);

            // MLE folding
            mle_l[i] = mle_l[i] + r * (mle_l[h + i] - mle_l[i]);
            mle_r[i] = mle_r[i] + r * (mle_r[h + i] - mle_r[i])
        }
        mle_l = &mut mle_l[..h];
        mle_r = &mut mle_r[..h];

        // should factor be included? because aren't we just reintroducing the problem?
        sumcheck.push((factor * sum_0, factor * sum_inf));

        factor *= r * z + (1 - z) * (1 - r);
    }

    (mle_l[0], mle_r[0])
}

fn gpgkr_verify(
    final_value: Field,
    circuit: Circuit,
    challenges: Challenge,
    round01: Vec<(Field, Field)>,
    sumcheck: TriangularArray<(Field, Field)>,
) -> bool {
    let mut claim = final_value;
    // TODO: this only verifies the connections between layers 0..m-2. The
    // deepest layer's (w0, w1) — round01[m-1] — is never read, and nothing
    // evaluates the actual leaf MLE at the final `point` to cross-check it.
    // As-is, a prover can claim any (w0, w1) for the leaf layer and this
    // still returns true.
    let rounds = circuit.leafs.len().ilog2() as usize;

    let start_point = [];
    let mut point: Point<Field> = Point::new(&start_point);
    let start_sumcheck = [];
    let mut sumcheck_round: &[(Field, Field)] = &start_sumcheck;

    for i in 0..rounds {
        let next_point = challenges.build_point(i);
        let (r, sumcheck_challenge) = next_point.sc();

        let (factor, sumcheck_final) =
            verify_round(claim, point, sumcheck_round, sumcheck_challenge);

        let claim_lr = round01[i];
        // Check if line polynomial hits same spot as sumcheck check
        if (factor * claim_lr.0 * claim_lr.1) != sumcheck_final {
            return false;
        }

        // Reduce both claims to a single claim
        claim = claim_lr.0 + r * (claim_lr.1 - claim_lr.0);
        point = next_point;
        if i < rounds - 1 {
            sumcheck_round = sumcheck.round(i);
        }
    }

    // Deal with input layer
    let leaf_check = mle(circuit.leafs, point);

    leaf_check == claim
}

fn verify_round(
    mut claim: Field,
    point: Point<Field>,
    sumcheck: &[(Field, Field)],
    sumcheck_challenge: &[Field],
) -> (Field, Field) {
    let mut prefix = 1;
    assert_eq!(sumcheck_challenge.len(), point.len());

    for ((r, z), (sum0, suminf)) in sumcheck_challenge.iter().zip(point).zip(sumcheck) {
        let eqjsum0 = (1 - z) * sum0;
        let eqjsum1 = claim - eqjsum0;
        let sum1 = eqjsum1 / z;

        // TODO rewrite
        let factor = r * z + (1 - r) * (1 - z);
        // TODO factor r
        claim = factor * (sum0 + r * (sum1 - sum0) + r * (r - 1) * suminf);

        prefix *= factor;
    }
    (prefix, claim)
}

struct SuffixTable(Vec<Vec<Field>>);

impl SuffixTable {
    // This option might get annoying
    // Can be allocated as a single table directly.
    // Maybe vectors can be reused? This could be reused with sumcheck layer. The evaluation layer can be reused each gkr
    // Might be better in combination with split_eq table
    // currently has to be consumed in reverse order -> pop. Which is fine
    fn new<'a>(point: &Point<'a, Field>) -> SuffixTable {
        // We do not need to build the table for the first challenge.
        let mut table = Vec::with_capacity(point.len());
        let mut prev = Vec::from([1]);

        // The selector is the first entry of the points and we need to skip that. So this works
        let (_s, c) = point.sc();

        // for z in point.skip(1).rev() {
        for z in c {
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
    fn batched_eval(&self, groups: usize) -> CircuitEval {
        let mut witnesses = Vec::with_capacity(self.leafs.len().ilog2() as usize);

        let mut prev_eval = self.leafs.clone();

        // Once we hit number of groups we have one per group
        while prev_eval.len() > groups {
            let mut eval = Vec::with_capacity(prev_eval.len() >> 1);
            let mid = prev_eval.len() / 2;
            let (l, r) = prev_eval.split_at(mid);
            for pair in l.iter().zip(r) {
                eval.push(pair.0 * pair.1)
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

    fn into_iter(self) -> impl Iterator<Item = Vec<Field>> {
        self.0.into_iter().rev()
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
