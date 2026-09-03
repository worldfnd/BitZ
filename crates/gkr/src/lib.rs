use std::collections::VecDeque;

use field::{F128, Wide256};
use num_traits::{ConstOne, ConstZero};
use poly::DenseMultilinearExtension;
use rayon::prelude::*;
use transcript::{ProverState, VerifierState};

pub type Field = F128;

// TODO modify densemultilinearextension such that no point reversal is necessary
fn mle(eval: Vec<Field>, rs: &Point) -> Field {
    let num_vars = rs.len();
    let mut point: Vec<Field> = rs.iter().cloned().collect();
    point.reverse();

    // TODO DenseMultilinearExtension::from_evaluations doesn't need a num_vars it already checks based on evaluation size.
    // Possibly we could even do zerro padding, but that means memory allocation. Better to have a check beforehand of power of two.
    // Direction should be a parameter
    DenseMultilinearExtension::from_evaluations(num_vars, eval)
        .unwrap()
        .evaluate(&point)
        .unwrap()
}

type Point = VecDeque<Field>;

// Functions for stand alone use
fn prove(input: Vec<Field>, groups: usize) -> (Vec<Field>, transcript::Proof) {
    let circuit = Circuit::new(input);
    let (last_value, witnesses) = circuit.batched_eval(groups);

    // TODO instance is a bit loose value and should be replaced by PCS
    let instance = (last_value.clone(), circuit.leafs);

    let mut prover = transcript::build_prover("gkr", &instance);

    gpgkr_prove(&mut prover, groups.ilog2() as u8, witnesses);
    (last_value, prover.finish())
}

fn verify(input: Vec<Field>, output: Vec<Field>, proof: transcript::Proof) -> bool {
    let circuit = Circuit::new(input);
    let instance = (&output, &circuit.leafs);
    let mut verifier = transcript::build_verifier("gkr", &instance, &proof);

    let ok = gpgkr_verify(&mut verifier, output, circuit);

    ok && verifier.check_eof().is_ok()
}

pub fn gpgkr_prove(
    ps: &mut ProverState,
    log_groups: u8,
    // All the intermediate witnesses + the input layer. Doesn't contain the output layer
    witnesses: LayerWitnesses,
) {
    // Edge cases
    // - witnesses < log groups (configuration mistake) panic
    // - empty witnesses -> single constant circuit -> one verifier message that permutes the proof state, but a single constant can't have an MLE
    let groups = 1usize << log_groups;
    // The layer directly below the output must be exactly twice the output
    // layer's size -- that's the batching invariant `batched_eval` maintains.
    assert_eq!(witnesses.next_layer_len(), Some(groups * 2));

    // Extension point chosen by the verifier to evaluate the output layer
    let mut point: Point = (0..log_groups).map(|_| ps.verifier_message()).collect();

    for wnext in witnesses.into_iter() {
        point = prove_layer(ps, point, wnext);
    }
}

fn prove_layer(ps: &mut ProverState, point: Point, mut wnext: Vec<Field>) -> Point {
    let mut suffix_table = SuffixTable::new(&point);
    let mut factor = Field::ONE;

    let mid = wnext.len() / 2;
    // Tree is encoded in LSB order
    let (mut mle_l, mut mle_r) = wnext.split_at_mut(mid);

    // TODO: use a double buffer or override approach? Now there is a point allocation each layer
    // Can go up to ~21 allocations assuming input of 2^35 and 6:4 split
    let mut next_point = VecDeque::with_capacity(point.len() + 1);

    for z in point {
        // TODO: special-case eq.len() == 1 (final round) to skip the `eq[i] *`
        // multiplications below entirely.
        // TODO: unwrap will be dealt with in upcoming approach to SuffixTable
        let eq = suffix_table.pop().unwrap();
        let h = mle_l.len() / 2;
        debug_assert_eq!(eq.len(), h);

        let (lo_l, hi_l) = mle_l.split_at_mut(h);
        let (lo_r, hi_r) = mle_r.split_at_mut(h);

        let (sum_0, sum_inf) = lo_l
            .par_iter_mut()
            .zip(lo_r.par_iter_mut())
            .zip(hi_l.par_iter())
            .zip(hi_r.par_iter())
            .zip(eq.par_iter())
            .with_min_len(PARALLEL_MIN_LANES)
            .fold(
                || (Wide256::zero(), Wide256::zero()),
                |(mut sum_0, mut sum_inf), ((((l_lo, r_lo), &l_hi), &r_hi), &e)| {
                    let (d_l, d_r) = (l_hi - *l_lo, r_hi - *r_lo);

                    // `e * l0 * r0` and `e * (l_hi-l0) * (r_hi-r0)`: each is
                    // two multiplications in a row, deferred into the
                    // running wide sums by `mul3_wide`
                    sum_0 += mul3_wide(e, *l_lo, *r_lo);
                    sum_inf += mul3_wide(e, d_l, d_r);

                    (sum_0, sum_inf)
                },
            )
            .reduce(
                || (Wide256::zero(), Wide256::zero()),
                |(a0, ainf), (b0, binf)| (a0 + b0, ainf + binf),
            );

        ps.prover_message(&[factor * sum_0.reduce(), factor * sum_inf.reduce()]);

        let r = ps.verifier_message();
        next_point.push_back(r);

        lo_l.par_iter_mut()
            .zip(lo_r.par_iter_mut())
            .zip(hi_l.par_iter())
            .zip(hi_r.par_iter())
            .with_min_len(PARALLEL_MIN_LANES)
            .for_each(|(((l_lo, r_lo), &l_hi), &r_hi)| {
                // Recalculating delta rather than writing it in the top half of the array in the previous loop to save on writes.
                // No benchmarking has been done to check the difference.
                let (d_l, d_r) = (l_hi - *l_lo, r_hi - *r_lo);

                // narrow multiply as the next round will multiply with this result
                *l_lo = *l_lo + r * d_l;
                *r_lo = *r_lo + r * d_r;
            });
        mle_l = &mut mle_l[..h];
        mle_r = &mut mle_r[..h];

        factor *= eq_factor(r, z);
    }

    ps.prover_message(&[mle_l[0], mle_r[0]]);
    next_point.push_front(ps.verifier_message());
    next_point
}

/// `eq(r, z) = r*z + (1 - r)*(1 - z)`. Expanding gives
/// `1 + r + z + 2*r*z`, and in characteristic 2 `2*r*z = r*z + r*z = 0`, so
/// this is just `1 + r + z` -- no multiplication at all, and so nothing for
/// widemul to help with.
fn eq_factor(r: Field, z: Field) -> Field {
    poly::eq::eq_eval(&[r], &[z])
}

/// `a * b * c`: two multiplications in a row. The first is reduced -- it has
/// to come back down to a field element to feed the second carryless
/// multiply -- but the second is left unreduced, so callers can batch its
/// reduction with the rest of a running wide sum instead of paying for it on
/// every term.
fn mul3_wide(a: Field, b: Field, c: Field) -> Wide256 {
    Wide256::mul(Wide256::mul(a, b).reduce(), c)
}

// TODO SuffixTable becomes a wrapper around a preallocated vector that is large enough for all rounds.
// SuffixTable can be 'created' each round / destroyed to ensure proper truncation
struct SuffixTable(Vec<Vec<Field>>);

impl SuffixTable {
    /// Allocates all directly as it is as much space as what a double buffer approach would take.
    fn new(point: &Point) -> SuffixTable {
        let mut table = Vec::with_capacity(1 << point.len().saturating_sub(1));
        let mut prev = Vec::from([Field::ONE]);

        // The selector is the first entry of the points and we need to skip
        // that -- except when `point` is itself empty, in which case there
        // is no selector and `c` must stay empty too (`min(1)` keeps the
        // range start in-bounds there instead of panicking).
        let c = point.range(point.len().min(1)..);

        // Suffix table is in the reverse order of the points
        for &z in c.rev() {
            let size = prev.len() << 1;
            let mut entry = vec![Field::ZERO; size];
            let (low, hi) = entry.split_at_mut(size >> 1);

            for (i, &e) in prev.iter().enumerate() {
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

/// TODO replace this with a proper approach
///  Matches `poly`'s `PARALLEL_FOLD_THRESHOLD`, chosen for the same per-round
/// MLE-fold shape.
const PARALLEL_MIN_LANES: usize = 1 << 12;

fn gpgkr_verify(vs: &mut VerifierState, last_value: Vec<Field>, circuit: Circuit) -> bool {
    // Edge cases
    // | circuit | last value |
    //    0 0 -> valid no circuit has no output
    //    0 1 -> false
    //    1 0 -> false
    //    1 1 -> check if last value is smaller but might not even be necessary -> does it lead to an invalid proof already
    //    circuit but no output or is it just a short circuit?

    let log_groups = last_value.len().ilog2();
    let mut point: Point = (0..log_groups).map(|_| vs.verifier_message()).collect();

    let rounds = circuit.leafs.len().ilog2() - last_value.len().ilog2();

    let mut claim = mle(last_value, &point);

    for _i in 0..rounds {
        match verify_round(vs, claim, point) {
            Some(ret) => {
                (point, claim) = ret;
            }
            None => return false,
        }
    }

    // TODO replace by PCS
    let leaf_check = mle(circuit.leafs, &mut point);

    leaf_check == claim
}

fn verify_round(vs: &mut VerifierState, mut claim: Field, point: Point) -> Option<(Point, Field)> {
    let mut prefix = Field::ONE;

    let mut next_point: Point = VecDeque::new();

    for z in point {
        let [sum0, suminf]: [Field; 2] = vs.prover_message().unwrap();
        let eqjsum0 = (Field::ONE - z) * sum0;
        let eqjsum1 = claim - eqjsum0;
        let sum1 = eqjsum1 / z;

        let r = vs.verifier_message();
        next_point.push_back(r);
        let factor = eq_factor(r, z);

        // `claim = factor * (sum0 + r*(sum1 - sum0) + r*(r - 1)*suminf)`.
        let bracket = (sum1 - sum0) + (r - Field::ONE) * suminf;
        claim = factor * (sum0 + r * bracket);

        prefix *= factor;
    }

    let elem_lr: [Field; 2] = vs.prover_message().unwrap();
    // Check if line polynomial hits same spot as sumcheck check
    if (prefix * elem_lr[0] * elem_lr[1]) != claim {
        None
    } else {
        let r = vs.verifier_message();
        next_point.push_front(r);

        // Reduce both claims to a single claim
        claim = elem_lr[0] + r * (elem_lr[1] - elem_lr[0]);

        Some((next_point, claim))
    }
}

//TODO circuit and circuit eval can't have there innards directly available as that would break power of 2 requirements for the rest.
// A circuit is defined by it's leaf value only because it is a balanced tree
// TODO: Optimise for circuits that are padded.
struct Circuit {
    leafs: Vec<Field>,
}

impl Circuit {
    // Fails if the leafs are 0.
    fn new(mut leafs: Vec<Field>) -> Self {
        if leafs.len() > 0 {
            leafs.resize(leafs.len().next_power_of_two(), Field::ONE);
        }
        Circuit { leafs: leafs }
    }

    // Returns the final evaluation and the witnesses of the intermediate layers
    // Can't consume the input as the circuit is necessary for the initialisation of fiat shamir
    fn batched_eval(&self, groups: usize) -> (Vec<Field>, LayerWitnesses) {
        // +1 to deal with the possible case that the leafs are empty. Given that otherwise the constructor padded it to a power of two and ilog rounds it down it becomes a noop
        let mut witnesses = Vec::with_capacity((self.leafs.len() + 1).ilog2() as usize);

        // TODO expensive clone going to get replaced by leaf lookups
        let mut prev_eval = self.leafs.clone();

        // Stop when there is one output per group
        while prev_eval.len() > groups {
            let mut eval = Vec::with_capacity(prev_eval.len() >> 1);
            let mid = prev_eval.len() / 2;
            let (l, r) = prev_eval.split_at(mid);
            for (&a, &b) in l.iter().zip(r) {
                eval.push(a * b)
            }
            witnesses.push(prev_eval);
            prev_eval = eval;
        }

        (prev_eval, LayerWitnesses(witnesses))
    }
}

// TODO: This can be a single vec that is allocated at the start.
pub struct LayerWitnesses(Vec<Vec<Field>>);

impl LayerWitnesses {
    fn pop(&mut self) -> Option<Vec<Field>> {
        self.0.pop()
    }

    /// Length of the layer `into_iter` yields first -- the one directly
    /// below the output layer.
    fn next_layer_len(&self) -> Option<usize> {
        self.0.last().map(Vec::len)
    }
}

impl IntoIterator for LayerWitnesses {
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

    fn field() -> impl Strategy<Value = Field> {
        any::<u128>().prop_map(Field::from)
    }

    // Regression test for a real bug in `SuffixTable::new`: it folded
    // `point.c()` front-to-back and handed tables out LIFO (largest/most-
    // complete first), which drops elements off the *back* of `c` as rounds
    // proceed. But round `k`'s eq-weight is owed to the challenges *not yet
    // consumed*, which are `c`'s later elements (rounds consume `c[0]`,
    // `c[1]`, ... in order) -- so tables need to drop from the *front*
    // instead, which folding `c` in reverse produces. With `c.len() <= 1`
    // (a point of length <= 2) there's no distinguishable front/back of a
    // single element, so this stayed invisible: `gpgkr_round_trip`'s
    // original leaf range never drove `prove_layer`/`verify_round` past a
    // 2-element point. This isolates the pair from the multi-layer
    // claim-chaining entirely -- one arbitrary layer of length `2^(l+1)`, an
    // arbitrary `l`-length point, an honestly-seeded `claim` -- and checks
    // `factor * elem_lr[0] * elem_lr[1] == verify_round`'s output across `l`
    // from 1 up through 8, well past the 2-element point where the bug
    // first showed up. `elem_lr` is read back off the transcript rather than
    // taken as `prove_layer`'s return value, the same way a real verifier
    // would, since `prove_layer` now writes it to `ps` instead of returning
    // it directly.
    #[test]
    fn suffix_table_handles_points_past_two_elements() {
        for l in 1usize..=8 {
            let n = 1usize << (l + 1);
            let wnext: Vec<Field> = (1u128..=(n as u128)).map(Field::from).collect();
            let z_vals: Vec<Field> = (100u128..100 + l as u128).map(Field::from).collect();
            let point: Point = z_vals.iter().cloned().collect();
            let half = wnext.len() / 2;
            let (lo, hi) = wnext.split_at(half);
            let above: Vec<Field> = lo.iter().zip(hi).map(|(&a, &b)| a * b).collect();
            let claim = mle(above, &point);

            let mut prover = transcript::build_prover("suffix-table-test", &z_vals);
            prove_layer(&mut prover, point.clone(), wnext.clone());
            let proof = prover.finish();

            let mut verifier = transcript::build_verifier("suffix-table-test", &z_vals, &proof);
            let result = verify_round(&mut verifier, claim, point);

            assert!(result.is_some(), "l = {l}");
            verifier.check_eof().unwrap();
        }
    }

    proptest! {
        // Pairwise multiplication is associative, so folding-by-multiply any
        // witness layer (including the leaves, padded with the multiplicative
        // identity 1) must equal folding the original input.
        #[test]
        fn eval_preserves_product_across_layers(leaves in prop::collection::vec(field(), 0..12)) {
            let expected = leaves.iter().fold(Field::ONE, |acc, &x| acc * x);

            let (top, mut witnesses) = Circuit::new(leaves).batched_eval(1);

            let folded = top.iter().fold(Field::ONE, |acc, &x| acc * x);
            prop_assert_eq!(folded, expected);

            let mut layers_checked = 1;
            while let Some(layer) = witnesses.pop() {
                let folded = layer.iter().fold(Field::ONE, |acc, &x| acc * x);
                prop_assert_eq!(folded, expected);
                layers_checked += 1;
            }
            prop_assert!(layers_checked > 0);
        }

        // Round-trips the grand-product GKR proof: prove the circuit's
        // product, then check the verifier accepts against the top layer
        // the prover actually produced (groups = 1, so no batching -- the
        // top layer is just the scalar product, checked against `expected`
        // independently before handing it to the verifier). Range goes up to
        // 33 leaves (padded to 64, k = 6) rather than a narrower one so this
        // regularly drives some layer's `prove_layer`/`verify_round` call
        // past a 2-element point -- see
        // `suffix_table_handles_points_past_two_elements`'s doc comment for
        // why that boundary matters. Starts at 2 rather than 1: a single
        // leaf pads to a single-element circuit with no witness layers at
        // all, the "single constant can't have an MLE" case `gpgkr_prove`
        // explicitly doesn't support.
        #[test]
        fn gpgkr_round_trip(leaves in prop::collection::vec(field(), 2..33)) {
            let expected: Field = leaves.iter().fold(Field::ONE, |acc, &x| acc * x);

            let (last_value, proof) = prove(leaves.clone(), 1);
            prop_assert_eq!(last_value.clone(), vec![expected]);

            prop_assert!(verify(leaves, last_value, proof));
        }

        // `batched_eval` stops descending the tree once a layer has `groups`
        // entries left, so the surviving top layer should hold one partial
        // product per group rather than the full product. Each step
        // multiplies the *low* half of the current layer by the *high*
        // half (`split_at` at the midpoint, same MSB-first split
        // `prove_layer`'s sumcheck fold uses), so the bits that survive
        // uncollapsed are the low `log2(groups)` bits of the original leaf
        // index: group `j`'s surviving entry is the product of every
        // (power-of-two-padded) leaf whose index is congruent to `j` modulo
        // `groups`. This is the one part of batching that's already fully
        // implemented -- `Circuit::batched_eval` on its own, with nothing
        // downstream of it yet -- so this pins it as a correctness
        // invariant to build the rest of batching on top of.
        #[test]
        fn batched_eval_top_layer_is_interleaved_group_products(
            leaves in prop::collection::vec(field(), 1..32),
            groups_log2 in 0usize..4,
        ) {
            let padded_len = leaves.len().next_power_of_two();
            let groups_log2 = groups_log2.min(padded_len.ilog2() as usize);
            let groups = 1usize << groups_log2;

            let mut padded = leaves.clone();
            padded.resize(padded_len, Field::ONE);

            let circuit = Circuit::new(leaves);
            let (top, _witnesses) = circuit.batched_eval(groups);
            prop_assert_eq!(top.len(), groups);

            for (j, &value) in top.iter().enumerate() {
                let expected = padded
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % groups == j)
                    .fold(Field::ONE, |acc, (_, &x)| acc * x);
                prop_assert_eq!(value, expected);
            }
        }

        // Round-trips batched (`groups > 1`) grand-product GKR: `gpgkr_prove`
        // hands back the `groups`-sized top layer alongside the proof, and
        // `gpgkr_verify` takes it directly as public input, seeding its
        // starting claim as that layer's own MLE evaluated at the same
        // `lgroups` group-selector challenges `gpgkr_prove` used -- mirroring
        // how `mle(circuit.leafs, point)` already checks the leaf layer at
        // the other end. Leaves go up to 33 (padded to 64) and groups up to
        // 8 so `m = k - lgroups` regularly exceeds 2, same reasoning as
        // `gpgkr_round_trip`'s range. Clamped strictly below `padded_len`'s
        // log2 (not just `<=`) so `groups` never reaches the padded leaf
        // count -- `groups == padded_len` leaves zero witness layers per
        // group, the same unsupported single-constant case
        // `gpgkr_round_trip` avoids by starting its range at 2.
        #[test]
        fn gpgkr_round_trip_batched(
            leaves in prop::collection::vec(field(), 4..33),
            groups_log2 in 1usize..4,
        ) {
            let padded_len = leaves.len().next_power_of_two();
            let groups_log2 = groups_log2.min(padded_len.ilog2() as usize - 1).max(1);
            let groups = 1usize << groups_log2;

            let (last_value, proof) = prove(leaves.clone(), groups);

            prop_assert!(verify(leaves, last_value, proof));
        }

        // Regression guard for a real bug: `verify` used to discard
        // `gpgkr_verify`'s boolean result entirely and only checked that the
        // proof's byte-lengths lined up (`check_eof`), so it accepted any
        // proof shaped correctly regardless of whether the sumcheck
        // equations actually held. Tampering with the claimed output after
        // proving changes what `verify` absorbs into the transcript's
        // instance tag, desyncing every challenge derived from it, so
        // verification must fail. Starts at 2 leaves for the same reason as
        // `gpgkr_round_trip`'s range: a single leaf hits the unsupported
        // no-witness-layers case.
        #[test]
        fn gpgkr_verify_rejects_tampered_output(leaves in prop::collection::vec(field(), 2..33)) {
            let (last_value, proof) = prove(leaves.clone(), 1);

            let mut tampered = last_value;
            tampered[0] = tampered[0] + Field::ONE;

            prop_assert!(!verify(leaves, tampered, proof));
        }
    }
}
