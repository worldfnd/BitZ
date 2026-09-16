use std::collections::VecDeque;

use field::{F128, Wide256};
use num_traits::{ConstOne, ConstZero};
use rayon::prelude::*;
use transcript::{ProverState, VerifierState};

pub type Field = F128;

type Point = VecDeque<Field>;

/// Proves the layer-by-layer sumcheck reduction from a claim at `point`
/// (an evaluation point on the output layer) down to a claim on the leaves.
// TODO #[must_use], requires changing the test suite
pub fn gpgkr_prove(
    ps: &mut ProverState,
    point: &[F128],
    // All the intermediate witnesses + the input layer. Doesn't contain the output layer
    witnesses: LayerWitnesses,
) -> (Vec<F128>, Field) {
    // Edge cases
    // - empty witnesses -> single constant circuit -> one verifier message that permutes the proof state, but a single constant can't have an MLE
    let mut point = point.to_owned();
    point.reverse();

    let mut point = VecDeque::from(point);

    let mut claim = Field::ZERO;
    for wnext in witnesses.into_iter() {
        (point, claim) = prove_layer(ps, point, wnext);
    }

    let mut point = Vec::from(point);
    point.reverse();
    (point, claim)
}

fn prove_layer(ps: &mut ProverState, point: Point, mut wnext: Vec<Field>) -> (Point, Field) {
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

        // At z = 0 the incoming claim determines the value at zero, so send
        // the value at one. Otherwise send the value at zero as usual.
        let send_one = z == Field::ZERO;
        let (sum_endpoint, sum_inf) = lo_l
            .par_iter_mut()
            .zip(lo_r.par_iter_mut())
            .zip(hi_l.par_iter())
            .zip(hi_r.par_iter())
            .zip(eq.par_iter())
            .with_min_len(PARALLEL_MIN_LANES)
            .fold(
                || (Wide256::zero(), Wide256::zero()),
                |(mut sum_endpoint, mut sum_inf), ((((l_lo, r_lo), &l_hi), &r_hi), &e)| {
                    let (d_l, d_r) = (l_hi - *l_lo, r_hi - *r_lo);
                    let (l_endpoint, r_endpoint) = if send_one {
                        (l_hi, r_hi)
                    } else {
                        (*l_lo, *r_lo)
                    };

                    // The endpoint product and `e * (l_hi-l0) * (r_hi-r0)`: each is
                    // two multiplications in a row, deferred into the
                    // running wide sums by `mul3_wide`
                    sum_endpoint += mul3_wide(e, l_endpoint, r_endpoint);
                    sum_inf += mul3_wide(e, d_l, d_r);

                    (sum_endpoint, sum_inf)
                },
            )
            .reduce(
                || (Wide256::zero(), Wide256::zero()),
                |(a0, ainf), (b0, binf)| (a0 + b0, ainf + binf),
            );

        ps.prover_message(&[factor * sum_endpoint.reduce(), factor * sum_inf.reduce()]);

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
                *l_lo += r * d_l;
                *r_lo += r * d_r;
            });
        mle_l = &mut mle_l[..h];
        mle_r = &mut mle_r[..h];

        factor *= eq_factor(r, z);
    }

    ps.prover_message(&[mle_l[0], mle_r[0]]);
    let r = ps.verifier_message();
    next_point.push_front(r);
    let claim = mle_l[0] + r * (mle_r[0] - mle_l[0]);
    (next_point, claim)
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

// TODO:  SuffixTable becomes a wrapper around a preallocated vector that is large enough for all rounds.
//          SuffixTable can be 'created' each round / destroyed to ensure proper truncation of the underlying vector
// TODO: Split suffix table
struct SuffixTable(Vec<Vec<Field>>);

impl SuffixTable {
    /// Allocates all directly as it is as much space as a double buffer approach would take.
    fn new(point: &Point) -> SuffixTable {
        let mut table = Vec::with_capacity(point.len().max(1));
        let mut prev = Vec::from([Field::ONE]);

        // The selector is the first entry of the point and we need to skip
        // that -- except when `point` is itself empty, in which case there
        // is no selector and `c` must stay empty too (`min(1)` keeps the
        // range start in-bounds there instead of panicking).
        let c = point.range(point.len().min(1)..);

        // Suffix table is in the reverse order of the point
        for &z in c.rev() {
            let size = prev.len() << 1;
            let mut entry = Field::zeroed_vec(size);
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

#[must_use]
pub fn gpgkr_verify(
    vs: &mut VerifierState,
    mut claim: Field,
    point: &[F128],
    rounds: u32,
) -> Option<(Vec<F128>, Field)> {
    // Edge cases around input lenghts, 0 meaning empty
    // | circuit | last value |
    //    0 0 -> valid, no circuit has no output
    //    0 1 -> false
    //    1 0 -> false
    // last value equal to circuit
    //    direct comparison of the two
    // last value being larger than circuit
    //    -> false
    //
    let mut point = point.to_owned();
    point.reverse();
    let mut point = VecDeque::from(point);

    for _i in 0..rounds {
        (point, claim) = verify_layer(vs, claim, point)?
    }

    let mut point = Vec::from(point);
    point.reverse();

    Some((point, claim))
}

/// Point's orientation is the reverse of gpgkr_verify
fn verify_layer(vs: &mut VerifierState, mut claim: Field, point: Point) -> Option<(Point, Field)> {
    let mut prefix = Field::ONE;

    let mut next_point: Point = VecDeque::new();

    for z in point {
        let [sum_endpoint, suminf]: [Field; 2] = vs.prover_message().ok()?;
        // claim = (1 - z) * sum0 + z * sum1. At z = 0, sum0 is known
        // and the prover supplies sum1; otherwise recover sum1 by division.
        let (sum0, sum1) = if z == Field::ZERO {
            (claim, sum_endpoint)
        } else {
            let eqjsum0 = (Field::ONE - z) * sum_endpoint;
            (sum_endpoint, (claim - eqjsum0) / z)
        };

        let r = vs.verifier_message();
        next_point.push_back(r);
        let factor = eq_factor(r, z);

        // `claim = factor * (sum0 + r*(sum1 - sum0) + r*(r - 1)*suminf)`.
        let bracket = (sum1 - sum0) + (r - Field::ONE) * suminf;
        claim = factor * (sum0 + r * bracket);

        prefix *= factor;
    }

    let elem_lr: [Field; 2] = vs.prover_message().ok()?;
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

//TODO circuit and circuit eval can't have their innards directly available as that would break power of 2 requirements for the rest.
// A circuit is defined by its leaf value only because it is a balanced tree
// TODO: Optimise for circuits that are padded.
pub struct GrandProductCircuit {
    leafs: Vec<Field>,
}

impl GrandProductCircuit {
    // Fails if the leafs are 0.
    pub fn new(mut leafs: Vec<Field>) -> Self {
        if !leafs.is_empty() {
            leafs.resize(leafs.len().next_power_of_two(), Field::ONE);
        }
        GrandProductCircuit { leafs }
    }

    // Returns the final evaluation and the witnesses of the intermediate layers
    // Can't consume the input as the circuit is necessary for the initialisation of fiat shamir
    // TODO: replace with leaf lookups and add multithreading
    pub fn batched_eval(&self, groups: usize) -> (Vec<Field>, LayerWitnesses) {
        // +1 to deal with the possible case that the leafs are empty. Given that otherwise the constructor padded it to a power of two, and ilog rounds it down, it becomes a noop
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
    pub fn pop(&mut self) -> Option<Vec<Field>> {
        self.0.pop()
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

    /// Product of `xs`, folding from `1` -- the invariant every grand-product
    /// test below checks against.
    fn product(xs: &[Field]) -> Field {
        xs.iter().fold(Field::ONE, |acc, &x| acc * x)
    }

    /// Proves a grand-product circuit for `leaves` batched into `2^log_groups`
    /// groups, then checks the verifier accepts the proof against the
    /// prover's own claimed output layer. Returns that output layer so
    /// callers can assert further properties of it.
    fn prove_and_verify(
        leaves: Vec<Field>,
        log_groups: usize,
    ) -> Result<Vec<Field>, TestCaseError> {
        let (last_value, proof) = prove(leaves.clone(), log_groups);
        prop_assert!(verify(leaves, last_value.clone(), proof));
        Ok(last_value)
    }

    proptest! {
        // batched_eval, unbatched (groups = 1): pairwise multiplication is
        // associative, so folding-by-multiply any witness layer (including
        // the leaves, padded with the multiplicative identity 1) must equal
        // folding the original input. Checks every intermediate layer, not
        // just the top.
        #[test]
        fn eval_preserves_product_across_layers(leaves in prop::collection::vec(field(), 0..12)) {
            let expected = product(&leaves);

            let (top, mut witnesses) = GrandProductCircuit::new(leaves).batched_eval(1);

            prop_assert_eq!(product(&top), expected);

            while let Some(layer) = witnesses.pop() {
                prop_assert_eq!(product(&layer), expected);
            }
        }


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

            let circuit = GrandProductCircuit::new(leaves);
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

        // Full protocol, unbatched: proves and verifies through `prove`/
        // `verify`, then checks the output against an independently computed
        // product
        // Starts at 1 since Dense MLE can't handle zero
        #[test]
        fn gpgkr_round_trip(leaves in prop::collection::vec(field(), 1..100)) {
            let expected = product(&leaves);

            let last_value = prove_and_verify(leaves, 0)?;
            prop_assert_eq!(last_value, vec![expected]);
        }

        // Full protocol, batched (`groups > 1`)
        #[test]
        fn gpgkr_round_trip_batched(
            leaves in prop::collection::vec(field(), 4..33),
            groups_log2 in 1usize..4,
        ) {
            let padded_len = leaves.len().next_power_of_two();
            let groups_log2 = groups_log2.min(padded_len.ilog2() as usize - 1).max(1);

            prove_and_verify(leaves, groups_log2)?;
        }

        // negative/soundness test
        #[test]
        fn gpgkr_verify_rejects_tampered_output(leaves in prop::collection::vec(field(), 2..33)) {
            let (last_value, proof) = prove(leaves.clone(), 0);

            let mut tampered = last_value;
            tampered[0] += Field::ONE;

            prop_assert!(!verify(leaves, tampered, proof));
        }
    }

    #[test]
    fn gpgkr_round_trip_with_zero_coordinates() {
        let leaves: Vec<Field> = (1u128..=16).map(Field::from).collect();
        let nonzero = Field::from(5u128);
        for point in [
            [Field::ZERO, Field::ZERO],
            [Field::ZERO, nonzero],
            [nonzero, Field::ZERO],
            [Field::ZERO, Field::ONE],
            [Field::ONE, Field::ZERO],
        ] {
            let circuit = GrandProductCircuit::new(leaves.clone());
            let (output, witnesses) = circuit.batched_eval(4);
            let claim = mle(output, &point);
            let instance = (leaves.clone(), point.to_vec());
            let mut prover = transcript::build_prover("gkr-zero", &instance);
            let terminal = gpgkr_prove(&mut prover, &point, witnesses);
            assert_eq!(terminal.1, mle(leaves.clone(), &terminal.0));
            let proof = prover.finish();

            let mut verifier = transcript::build_verifier("gkr-zero", &instance, &proof);
            assert_eq!(
                gpgkr_verify(&mut verifier, claim, &point, 2),
                Some(terminal)
            );
            verifier.check_eof().unwrap();

            let mut verifier = transcript::build_verifier("gkr-zero", &instance, &proof);
            assert!(gpgkr_verify(&mut verifier, claim + Field::ONE, &point, 2).is_none());
        }
    }

    #[test]
    fn gpgkr_verify_rejects_every_truncated_prefix() {
        let leaves: Vec<Field> = (1u128..=8).map(Field::from).collect();
        for log_groups in [0, 1] {
            let (output, proof) = prove(leaves.clone(), log_groups);
            assert!(verify(leaves.clone(), output.clone(), proof.clone()));

            // Cover partial field elements and missing messages in every layer.
            for len in 0..proof.narg_string.len() {
                let mut truncated = proof.clone();
                truncated.narg_string.truncate(len);
                assert!(
                    !verify(leaves.clone(), output.clone(), truncated),
                    "accepted {len} bytes with log_groups = {log_groups}",
                );
            }
        }
    }

    pub fn prove(input: Vec<Field>, log_groups: usize) -> (Vec<Field>, transcript::Proof) {
        let circuit = GrandProductCircuit::new(input);
        let groups = 1usize << log_groups;
        let (last_value, witnesses) = circuit.batched_eval(groups);

        // TODO instance is a bit loose and should be replaced by PCS
        let instance = (last_value.clone(), circuit.leafs);

        let mut prover = transcript::build_prover("gkr", &instance);

        // Extension point chosen by the verifier to evaluate the output layer.
        // Mirrors `verify`'s `output.len().max(1).ilog2()` exactly, since both
        // sides must draw the same number of challenges here.
        let log_groups = last_value.len().max(1).ilog2();
        let point: Vec<Field> = (0..log_groups).map(|_| prover.verifier_message()).collect();

        gpgkr_prove(&mut prover, &point, witnesses);
        (last_value, prover.finish())
    }

    pub fn verify(input: Vec<Field>, output: Vec<Field>, proof: transcript::Proof) -> bool {
        let circuit = GrandProductCircuit::new(input);
        let instance = (&output, &circuit.leafs);
        let mut verifier = transcript::build_verifier("gkr", &instance, &proof);

        let log_groups = output.len().max(1).ilog2();
        let point: Vec<Field> = (0..log_groups)
            .map(|_| verifier.verifier_message())
            .collect();
        let claim = mle(output, &point);
        let log_groups = point.len() as u32;
        let log_leafs = circuit.leafs.len().max(1).ilog2();
        let rounds = log_leafs.saturating_sub(log_groups);

        match gpgkr_verify(&mut verifier, claim, &point, rounds) {
            Some((point, claim)) => {
                let leaf_check = mle(circuit.leafs, &point);

                let ok = leaf_check == claim;

                ok && verifier.check_eof().is_ok()
            }
            None => false,
        }
    }

    use poly::DenseMultilinearExtension;
    fn mle(eval: Vec<Field>, point: &[Field]) -> Field {
        let num_vars = point.len();

        // TODO DenseMultilinearExtension::from_evaluations doesn't need a num_vars; it already checks based on evaluation size.
        // TODO MLE should be able to handle empty point when given a single evaluation
        DenseMultilinearExtension::from_evaluations(num_vars, eval)
            .unwrap()
            .evaluate(point)
            .unwrap()
    }
}
