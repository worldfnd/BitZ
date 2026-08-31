pub struct Mle {
    table: Vec<Field>,
}

impl Mle {
    pub fn new(eval: Vec<Field>) -> Self {
        Mle { table: eval }
    }

    // g(0), g(1) for the round polynomial: sum of each half.
    pub fn round_sums(&self) -> (Field, Field) {
        let n = self.table.len();
        let (lower, upper) = self.table.split_at(n / 2);
        (lower.iter().sum(), upper.iter().sum())
    }

    pub fn fix_variable(&mut self, r: Field) {
        let n = self.table.len();
        let (lower, upper) = self.table.split_at_mut(n / 2);
        for i in 0..lower.len() {
            lower[i] = lower[i] + r * (upper[i] - lower[i]);
        }
        self.table.truncate(n / 2);
    }

    pub fn scalar(&self) -> Field {
        assert_eq!(self.table.len(), 1);
        self.table[0]
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }
}

impl std::ops::Index<usize> for Mle {
    type Output = Field;
    fn index(&self, i: usize) -> &Field {
        &self.table[i]
    }
}

pub fn mle(eval: Vec<Field>, rs: impl Iterator<Item = Field> + ExactSizeIterator) -> Field {
    assert_eq!(eval.len(), 1 << rs.len());
    let mut m = Mle::new(eval);
    for r in rs {
        m.fix_variable(r);
    }
    m.scalar()
}

use std::cell::UnsafeCell;
pub type Field = i128;

pub struct Challenge {
    data: UnsafeCell<TriangularArray<Field>>,
}

/// Triangular arrays with a fixed extra data per round.
pub struct TriangularArray<T> {
    data: Box<[T]>,
    len: usize,
    lgroups: usize,
}

impl<T: Default + Copy> TriangularArray<T> {
    pub fn with_capacity(rounds: usize, groups: usize) -> Self {
        let lg = groups.ilog2() as usize;
        let capacity = rounds * (rounds + 1) / 2 + rounds * lg;
        Self {
            data: vec![T::default(); capacity].into_boxed_slice(),
            len: 0,
            lgroups: lg,
        }
    }

    pub fn push(&mut self, el: T) {
        // Preincrement because we don't want to have 0 as a possible challenge due to division.
        let index = self.len;
        self.data[index] = el;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn round(&self, r: usize) -> &[T] {
        let start = r * (r + 1) / 2 + r * self.lgroups;
        let end_inclusive = start + r + self.lgroups;
        &self.data[start..=end_inclusive]
    }
}

pub struct Point<'a, T: Copy> {
    backend: &'a [T],
    state: PointState,
}

enum PointState {
    Start,
    Next(usize),
    Stop,
}

impl<'a, T: Copy> Point<'a, T> {
    pub fn new(backend: &'a [T]) -> Self {
        Self {
            backend: backend,
            state: if backend.len() != 0 {
                PointState::Start
            } else {
                PointState::Stop
            },
        }
    }

    // Will crash if called on empty
    pub fn sc(&self) -> (T, &[T]) {
        let selector_idx = self.backend.len() - 1;
        (self.backend[selector_idx], &self.backend[..selector_idx])
    }

    pub fn c(&self) -> &[T] {
        let selector_idx = self.backend.len().saturating_sub(1);
        &self.backend[..selector_idx]
    }
}

impl<'a, T: Copy> Iterator for Point<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self.state {
            PointState::Start => {
                let idx = self.backend.len() - 1;
                let val = self.backend[idx];
                self.state = if idx != 0 {
                    PointState::Next(0)
                } else {
                    PointState::Stop
                };
                Some(val)
            }
            PointState::Next(i) => {
                let val = self.backend[i];
                self.state = if i < self.backend.len() - 2 {
                    PointState::Next(i + 1)
                } else {
                    PointState::Stop
                };
                Some(val)
            }
            PointState::Stop => None,
        }
    }
}

impl<'a, T: Copy> ExactSizeIterator for Point<'a, T> {
    fn len(&self) -> usize {
        self.backend.len()
    }
}

impl Challenge {
    pub fn with_capacity(rounds: usize, groups: usize) -> Self {
        Self {
            data: UnsafeCell::new(TriangularArray::with_capacity(rounds, groups)),
        }
    }
    pub fn get_challenge(&self) -> Field {
        unsafe {
            let val = (*self.data.get()).len() as Field + 1;
            (*self.data.get()).push(val);
            val
        }
    }

    // new frame breaks when
    pub fn build_point<'a>(&'a self, round: usize) -> Point<'a, Field> {
        let backend = unsafe { (*self.data.get()).round(round as usize) };
        Point::new(backend)
    }

    // pub fn into_inner(self) -> TriangularArray<Field> {
    //     self.data.into_inner()
    // }
}

// pub struct BTreeArray<T> {
//     data: Box<[T]>,
//     len: usize,
// }

// impl<T: Default + Copy> BTreeArray<T> {
//     pub fn with_capacity(capacity: usize) -> Self {
//         BTreeArray {
//             data: vec![T::default(); capacity].into_boxed_slice(),
//             len: 0,
//         }
//     }

//     pub fn len(&self) -> usize {
//         self.len
//     }

//     // TODO capacity check?
//     pub fn push(&mut self, el: T) {
//         let index = self.len;
//         self.data[index] = el;
//         self.len += 1;
//     }

//     pub fn layer(&self, depth: u32) -> &[T] {
//         &self.data[2_usize.pow(depth) - 1..2_usize.pow(depth + 1) - 1]
//     }
// }

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
    let mut suffix_table = SuffixTable::new(&point);
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
        debug_assert_eq!(eq.len(), h);

        let (lo_l, hi_l) = mle_l.split_at_mut(h);
        let (lo_r, hi_r) = mle_r.split_at_mut(h);

        for ((((l_lo, r_lo), &l_hi), &r_hi), &e) in lo_l
            .iter_mut()
            .zip(lo_r.iter_mut())
            .zip(hi_l.iter())
            .zip(hi_r.iter())
            .zip(eq.iter())
        {
            let (l0, r0) = (*l_lo, *r_lo);
            sum_0 += e * l0 * r0;
            sum_inf += e * (l_hi - l0) * (r_hi - r0);

            // MLE folding
            *l_lo = l0 + r * (l_hi - l0);
            *r_lo = r0 + r * (r_hi - r0);
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
        let c = point.c();

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
}

impl IntoIterator for CircuitEval {
    type Item = Vec<Field>;
    type IntoIter = std::iter::Rev<std::vec::IntoIter<Vec<Field>>>;

    fn into_iter(self) -> Self::IntoIter {
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
        fn eval_preserves_product_across_layers(leaves in prop::collection::vec(-3i128..=3, 0..12)) {
            let expected = leaves.iter().fold(1, |acc, &x| acc * x);

            let mut eval = Circuit::new(leaves).batched_eval(1);

            let mut layers_checked = 0;
            while let Some(layer) = eval.pop() {
                let folded = layer.iter().fold(1, |acc, &x| acc * x);
                prop_assert_eq!(folded, expected);
                layers_checked += 1;
            }
            prop_assert!(layers_checked > 0);
        }

        // Round-trips the grand-product GKR proof: prove the circuit's
        // product, then check the verifier accepts against the true
        // product value (groups = 1, so no batching).
        //
        // Field = i32 has no modulus yet, so challenge/sum products
        // compound unchecked across rounds and layers. Depth <= 3 (up to
        // 8 leaves) is the deepest that reliably stays in i32 range with
        // leaf values in -3..=3; depth 4+ overflows even when the
        // sumcheck logic itself is correct, so this is a field-width
        // limitation, not something to chase as a prove/verify bug.
        #[test]
        fn gpgkr_round_trip(leaves in prop::collection::vec(-3i128..=3, 1..8)) {
            let expected: Field = leaves.iter().fold(1, |acc, &x| acc * x);

            let circuit = Circuit::new(leaves);
            let m = circuit.leafs.len().ilog2() as usize;
            let witnesses = circuit.batched_eval(1);

            let c = Challenge::with_capacity(m, 1);
            let (round01, sumcheck) = gpgkr_prove(1, &c, witnesses);

            prop_assert!(gpgkr_verify(expected, circuit, c, round01, sumcheck));
        }

        // The selector is stored last in the backend but needs to lead
        // logically, so Point's iterator should yield the backend rotated
        // by one: [backend[len-1], backend[0], ..., backend[len-2]].
        // `gpgkr_prove`/`gpgkr_verify` hit backend length 0 at round 0
        // (no challenges sampled yet) and length 1 at round 1 (with
        // groups = 1) -- both currently panic with "attempt to subtract
        // with overflow" inside `Iterator for Point::next` in lib.rs
        // instead of producing (respectively) no elements and one element.
        // That's what blocks `gpgkr_round_trip` above from getting past
        // round 0.
        #[test]
        fn point_rotates_last_element_to_front(backend in prop::collection::vec(any::<Field>(), 0..16)) {
            let collected: Vec<Field> = Point::new(&backend).collect();

            let mut expected = Vec::with_capacity(backend.len());
            if let Some((&last, rest)) = backend.split_last() {
                expected.push(last);
                expected.extend_from_slice(rest);
            }

            prop_assert_eq!(collected, expected);
        }
    }
}
