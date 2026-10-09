use std::collections::VecDeque;

use common::{BitMatrix, BitTable, matrix::StepIter};
use field::{F128, Wide256};
use num_traits::{ConstOne, ConstZero};
use rayon::prelude::*;
use transcript::{ProverState, VerifierState};

pub type Field = F128;

type Point = VecDeque<Field>;

/// Proves the layer-by-layer sumcheck reduction from a claim at `point`
/// (an evaluation point on the output layer) down to a claim on the leaves.
// TODO #[must_use], requires changing the test suite
#[tracing::instrument(name = "Prove GKR", skip_all)]
pub fn gpgkr_prove(
    ps: &mut ProverState,
    log_bits: usize,
    point: &[F128],
    // All the intermediate witnesses + the input layer, from the top down. Doesn't contain the output layer
    witnesses: impl IntoIterator<Item = Vec<Field>>,
) -> (Vec<F128>, Field) {
    // Edge cases
    // - empty witnesses -> single constant circuit -> one verifier message that permutes the proof state, but a single constant can't have an MLE
    let mut point = point.to_owned();
    point.reverse();

    let mut point = VecDeque::from(point);

    let mut claim = Field::ZERO;
    let mut suffix_storage = SuffixTable::alloc_storage(log_bits);
    for wnext in witnesses.into_iter() {
        (point, claim) = prove_layer(ps, &mut suffix_storage, point, wnext);
    }

    let mut point = Vec::from(point);
    point.reverse();
    (point, claim)
}

fn prove_layer(
    ps: &mut ProverState,
    storage: &mut [Field],
    mut point: Point,
    mut wnext: Vec<Field>,
) -> (Point, Field) {
    let suffix_table = SuffixTable::new(storage, &point);
    let mut factor = Field::ONE;

    let mid = wnext.len() / 2;
    // Tree is encoded in LSB order
    let (mut mle_l, mut mle_r) = wnext.split_at_mut(mid);

    let rounds = point.len();
    let mut next_point = VecDeque::with_capacity(rounds + 1);

    // `SuffixTable::layer(i)` has length `2^i`, so `h` is even on every round
    // except the last (`i == 0`). Peel that round out instead of checking
    // for it on every iteration.
    let last_z = point.pop_back();

    for (i, z) in (1..rounds).rev().zip(point) {
        let eq = suffix_table.layer(i);
        let h = mle_l.len() / 2;
        debug_assert_eq!(eq.len(), h);
        debug_assert_eq!(h % 2, 0);

        let (lo_l, hi_l) = mle_l.split_at_mut(h);
        let (lo_r, hi_r) = mle_r.split_at_mut(h);

        // `send_one` is constant for the whole round: at z = 0 the incoming
        // claim determines the value at zero, so this round sends the value
        // at one instead. Branch on it once here -- SEND_ONE below is a
        // const generic, so each instantiation gets only its own branch,
        // not a per-element check inside the hot fold.
        let send_one = z == Field::ZERO;
        let (sum_endpoint, sum_inf) = if send_one {
            reduce_sumcheck_round::<true>(lo_l, lo_r, hi_l, hi_r, eq)
        } else {
            reduce_sumcheck_round::<false>(lo_l, lo_r, hi_l, hi_r, eq)
        };

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

    if let Some(z) = last_z {
        // i == 0, h == 1: `SuffixTable::new` always seeds the base layer
        // (`eq[0]`) with `Field::ONE`, so the `eq[i] *` multiplication every
        // other round needs is the identity here -- skipped rather than
        // spent multiplying by one.
        debug_assert_eq!(mle_l.len(), 2);
        let (l_lo, l_hi) = (mle_l[0], mle_l[1]);
        let (r_lo, r_hi) = (mle_r[0], mle_r[1]);
        let (d_l, d_r) = (l_hi - l_lo, r_hi - r_lo);

        // Same `send_one` swap as the main loop above: at z = 0 the incoming
        // claim determines the value at zero, so send the value at one.
        let (l_endpoint, r_endpoint) = if z == Field::ZERO {
            (l_hi, r_hi)
        } else {
            (l_lo, r_lo)
        };
        let sum_endpoint = Wide256::mul(l_endpoint, r_endpoint);
        let sum_inf = Wide256::mul(d_l, d_r);

        ps.prover_message(&[factor * sum_endpoint.reduce(), factor * sum_inf.reduce()]);

        let r = ps.verifier_message();
        next_point.push_back(r);

        mle_l[0] = l_lo + r * d_l;
        mle_r[0] = r_lo + r * d_r;
        mle_l = &mut mle_l[..1];
        mle_r = &mut mle_r[..1];

        factor *= eq_factor(r, z);
    }

    ps.prover_message(&[mle_l[0], mle_r[0]]);
    let r = ps.verifier_message();
    next_point.push_front(r);
    let claim = mle_l[0] + r * (mle_r[0] - mle_l[0]);
    (next_point, claim)
}

fn reduce_sumcheck_round<const SEND_ONE: bool>(
    lo_l: &[Field],
    lo_r: &[Field],
    hi_l: &[Field],
    hi_r: &[Field],
    eq: &[Field],
) -> (Wide256, Wide256) {
    // chunking reduces variance in benchmarking
    lo_l.par_chunks_exact(2)
        .zip(lo_r.par_chunks_exact(2))
        .zip(hi_l.par_chunks_exact(2))
        .zip(hi_r.par_chunks_exact(2))
        .zip(eq.par_chunks_exact(2))
        // Chunked by 2, so the element-count threshold below is halved in
        // terms of chunks.
        .with_min_len(PARALLEL_MIN_LANES / 2)
        .fold(
            || (Wide256::zero(), Wide256::zero()),
            |(mut sum_endpoint, mut sum_inf), ((((l_lo, r_lo), l_hi), r_hi), e)| {
                let (d_l0, d_r0) = (l_hi[0] - l_lo[0], r_hi[0] - r_lo[0]);
                let (d_l1, d_r1) = (l_hi[1] - l_lo[1], r_hi[1] - r_lo[1]);

                let (l_e0, r_e0) = if SEND_ONE {
                    (l_hi[0], r_hi[0])
                } else {
                    (l_lo[0], r_lo[0])
                };
                let (l_e1, r_e1) = if SEND_ONE {
                    (l_hi[1], r_hi[1])
                } else {
                    (l_lo[1], r_lo[1])
                };

                let (a0, b0) = (e[0] * l_e0, e[0] * d_l0);
                let (a1, b1) = (e[1] * l_e1, e[1] * d_l1);

                sum_endpoint += Wide256::mul(a0, r_e0);
                sum_inf += Wide256::mul(b0, d_r0);
                sum_endpoint += Wide256::mul(a1, r_e1);
                sum_inf += Wide256::mul(b1, d_r1);

                (sum_endpoint, sum_inf)
            },
        )
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(a0, ainf), (b0, binf)| (a0 + b0, ainf + binf),
        )
}

/// `eq(r, z) = r*z + (1 - r)*(1 - z)`. Expanding gives
/// `1 + r + z + 2*r*z`, and in characteristic 2 `2*r*z = r*z + r*z = 0`, so
/// this is just `1 + r + z` -- no multiplication at all, and so nothing for
/// widemul to help with.
fn eq_factor(r: Field, z: Field) -> Field {
    Field::ONE + r + z
}

// TODO: Split suffix table
struct SuffixTable<'a> {
    storage: &'a [Field],
}

impl<'a> SuffixTable<'a> {
    fn layer(&self, i: usize) -> &[Field] {
        let start = (1 << i) - 1;
        let end = (1 << (i + 1)) - 1;
        &self.storage[start..end]
    }
}

impl<'a> SuffixTable<'a> {
    /// Allocates all directly as it is as much space as a double buffer approach would take.
    #[inline(never)]
    fn new(storage: &'a mut [Field], point: &Point) -> SuffixTable<'a> {
        // Establishes the whole table's space budget up front so LLVM can
        // prove the `low`/`hi` writes in the loop below are in bounds
        let needed = (1usize << point.len()).saturating_sub(1).max(1);
        assert!(storage.len() >= needed);

        storage[0] = F128::ONE;
        let (mut prev, mut remaining) = storage.split_at_mut(1);
        // The selector is the first entry of the point and we need to skip
        // that -- except when `point` is itself empty, in which case there
        // is no selector and `c` must stay empty too (`min(1)` keeps the
        // range start in-bounds there instead of panicking).
        let c = point.range(point.len().min(1)..);

        // Suffix table is in the reverse order of the point
        for &z in c.rev() {
            let size = prev.len() << 1;
            let (entry, next) = remaining.split_at_mut(size);
            let (low, hi) = entry.split_at_mut(entry.len() >> 1);

            for ((l, h), &e) in low.iter_mut().zip(hi.iter_mut()).zip(prev.iter()) {
                let tmp = z * e;
                // (1-z)*e, z*e
                (*l, *h) = (e - tmp, tmp)
            }

            prev = entry;
            remaining = next;
        }
        SuffixTable { storage }
    }

    fn alloc_storage(log_bits: usize) -> Vec<Field> {
        Field::zeroed_vec(1 << (log_bits.saturating_sub(1)).max(1))
    }
}

/// TODO replace this with a proper approach
///  Matches `poly`'s `PARALLEL_FOLD_THRESHOLD`, chosen for the same per-round
/// MLE-fold shape.
const PARALLEL_MIN_LANES: usize = 1 << 12;

#[must_use]
#[tracing::instrument(name = "Verify GKR", skip_all)]
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

// TODO: Optimise for circuits that are padded.
pub struct GrandProductCircuit<'a> {
    matrix: BitMatrix<'static>,
    images: &'a [F128],
}

impl<'a> GrandProductCircuit<'a> {
    pub fn new(row_images: &'a [F128], bittable: &BitTable) -> GrandProductCircuit<'a> {
        assert_eq!(
            row_images.len(),
            bittable.shape().rows(),
            "one image per row of the table"
        );
        // Leaf `(b << s) | c` is `row_images[b]` where `bit(c, b)` is set and
        // one elsewhere: columns occupy the low index bits, so each product
        // tree reduces one column, folding the row bits from the top.
        //
        // The bottom three layers are looked up in 16-entry tables, one per
        // four rows, so the top two row bits are split off and moved beside
        // the columns. After the transpose, row group `b' < rows/4` is one
        // contiguous run, and in it the bits of rows `b'`, `b' + rows/4`,
        // `b' + 2 rows/4` and `b' + 3 rows/4` of one column sit next to each
        // other as a nibble: the four leaves under node `(b' << s) | c` of
        // layer 2, columns in ascending order.
        let matrix = bittable.as_matrix();
        let log_dim2 = matrix.log_dim2() - 2;
        let matrix = matrix.reshape(log_dim2).transpose();

        Self {
            matrix,
            images: row_images,
        }
    }

    /// Returns the output layer, the product of each column of the table, and
    /// the witnesses of the layers below it.
    // TODO: add multithreading
    #[tracing::instrument(name = "Evaluate grand-product circuit", level = "debug", skip_all)]
    pub fn batched_eval(self) -> (Vec<Field>, LayerWitnesses) {
        // Lower
        /// `lut[s]` is the product of the inputs whose bit is set in `s`, `a`
        /// on bit 0 through `d` on bit 3 (`0b0000 -> 1`, `0b0001 -> a`,
        /// `0b0011 -> a*b`, ...), so a 4-bit lookup string indexes its
        /// multilinear value directly.
        ///
        /// Built by doubling: entries `2^i..2^(i+1)` are entries `0..2^i`
        /// times the `i`-th input. The first entry of each new half is the
        /// input itself, so it is copied rather than multiplied by one,
        /// leaving 11 multiplications -- one per entry with two or more
        /// factors, which is the minimum.
        fn create_lut(c: [Field; 4]) -> [Field; 16] {
            let mut lut = [Field::ONE; 16];
            for (i, x) in c.into_iter().enumerate() {
                let half = 1 << i;
                lut[half] = x;
                for k in 1..half {
                    lut[half + k] = lut[k] * x;
                }
            }
            lut
        }

        // `new` checked there is one image per row, so four per row group.
        let quart = self.matrix.dim1();
        // Each LUT takes one image from each quarter of the rows.
        let (a, rest) = self.images.split_at(quart);
        let (b, rest) = rest.split_at(quart);
        let (c, d) = rest.split_at(quart);
        // LUT for the bottom three layers (0,1,2)
        let lower_lut: Vec<[Field; 16]> = a
            .iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .map(|(((&a, &b), &c), &d)| create_lut([a, b, c, d]))
            .collect();

        // Layers 0 to 2 are all in the tables, so the first layer built in
        // full is layer 3. Its node `k` multiplies layer-2 nodes `k` and
        // `k + n/2`, each a whole nibble looked up, so the matrix and the
        // tables split in the middle into the left and the right child.
        // Read as `u64`s: on a little-endian target, bit `i` of the words is
        // bit `i` of the `u64`s too.
        const { assert!(cfg!(target_endian = "little")) };
        let bitstring: &[u64] = bytemuck::cast_slice(&self.matrix[..]);
        let (left, right) = bitstring.split_at(bitstring.len() / 2);
        let (left_luts, right_luts) = lower_lut.split_at(quart / 2);

        // Each table serves one run of whole words, so a pair of tables is
        // picked once and its nibbles are read a word at a time.
        // A table's run is one `xs[i1]` of the matrix, a nibble per node.
        const NIBBLES_IN_WORD: usize = u64::BITS as usize / 4;
        let lut_words = left.len() / left_luts.len(); // This introduces a bound. StepIterator can prevent that but is more expensive than doing it this way

        assert!(lut_words > 0, "each table needs at least a word of nibbles");
        let runs = left
            .chunks_exact(lut_words)
            .zip(right.chunks_exact(lut_words))
            .zip(left_luts.iter().zip(right_luts));

        // Access nible
        let nibble = |word: u64, i: usize| (word >> (4 * i)) as usize & 15;

        // One node per nibble of each left table.
        let mut layer3 = Vec::with_capacity(left_luts.len() * lut_words * NIBBLES_IN_WORD);
        for ((l_words, r_words), (l_lut, r_lut)) in runs {
            for (&l, &r) in l_words.iter().zip(r_words) {
                // `extend` from a `Range` knows its length, so there is no
                // capacity check or length update per node.
                layer3.extend(
                    (0..NIBBLES_IN_WORD).map(|i| l_lut[nibble(l, i)] * r_lut[nibble(r, i)]),
                );
            }
        }
        // Upper
        // A table serves one node of every product tree, so there is a tree
        // per nibble of a table.
        let trees = lut_words * NIBBLES_IN_WORD;
        let (output, upper_nodes) = fold_layers(layer3, trees);

        (
            output,
            LayerWitnesses {
                matrix: self.matrix,
                lower_lut,
                upper_nodes,
            },
        )
    }
}

/// Each layer-2 node of a run of row groups as the table of its row group and
/// the nibble of its four leaves, in node order: node `k = (b' << s) | c`
/// reads nibble `k`, and all `2^s` columns of row group `b'` share its table.
/// `nibbles` and `luts` cover the same row groups.
fn lut_nodes<'a>(
    nibbles: &'a [u128],
    luts: &'a [[Field; 16]],
) -> impl ExactSizeIterator<Item = (&'a [Field; 16], usize)> {
    let nibbles = StepIter::<4>::new(nibbles.iter());
    // Every table serves the same run of nibbles, one per column.
    // TODO check assembly that this isn't reloaded constantly. Or maybe it is unavoidable but might still save a shift instruction. Also might generate too much indirection.
    let log_columns = (nibbles.len() / luts.len()).ilog2();
    nibbles
        .enumerate()
        .map(move |(k, bs)| (&luts[k >> log_columns], bs as usize))
}

/// Multiplies `layer` pairwise, left half against right half, until one
/// value per group remains. Returns that output layer and every layer before
/// it, `layer` first.
fn fold_layers(mut layer: Vec<Field>, groups: usize) -> (Vec<Field>, Vec<Vec<Field>>) {
    let rounds = (layer.len() / groups).checked_ilog2().unwrap_or(0);
    let mut witnesses = Vec::with_capacity(rounds as usize);

    // Stop when there is one output per group
    while layer.len() > groups {
        let (l, r) = layer.split_at(layer.len() / 2);
        let next = l.iter().zip(r).map(|(&a, &b)| a * b).collect();
        witnesses.push(layer);
        layer = next;
    }

    (layer, witnesses)
}

/// The layers below the output, yielded from the top down. The bottom three
/// are kept as the tables they are looked up in, the rest in full.
// TODO: This can be a single vec that is allocated at the start.
pub struct LayerWitnesses {
    /// The nibbles of the layer-2 nodes; see [`GrandProductCircuit::new`].
    matrix: BitMatrix<'static>,
    /// One table per row group, shared by all of its columns.
    lower_lut: Vec<[Field; 16]>,
    /// Layer 3 and up, layer 3 first.
    upper_nodes: Vec<Vec<Field>>,
}

impl LayerWitnesses {
    /// The leaves and layers 1 and 2, looked up in full. A node is the
    /// product of the nibble bits its subtree covers: leaf quarter `q` is bit
    /// `q` alone, each half of layer 1 pairs the quarters two apart, and
    /// layer 2 takes all four.
    fn lower_layers(&self) -> [Vec<Field>; 3] {
        let nodes = lut_nodes(&self.matrix, &self.lower_lut);
        let n = nodes.len();
        let mut leaves = Field::zeroed_vec(4 * n);
        let mut layer1 = Field::zeroed_vec(2 * n);
        let mut layer2 = Field::zeroed_vec(n);

        for (k, (lut, bs)) in nodes.enumerate() {
            for q in 0..4 {
                leaves[q * n + k] = lut[bs & (1 << q)];
            }
            layer1[k] = lut[bs & 0b0101];
            layer1[n + k] = lut[bs & 0b1010];
            layer2[k] = lut[bs];
        }

        [leaves, layer1, layer2]
    }
}

// The sumcheck still reads every layer in full, so for now the bottom three
// are rebuilt from the tables here; layer 3 and up are already in
// `upper_nodes`.
impl IntoIterator for LayerWitnesses {
    type Item = Vec<Field>;
    type IntoIter = std::iter::Rev<std::vec::IntoIter<Vec<Field>>>;

    fn into_iter(self) -> Self::IntoIter {
        let mut layers = Vec::from(self.lower_layers());
        layers.extend(self.upper_nodes);
        layers.into_iter().rev()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{BitZParams, Shape};
    use field::gf128::smallest_generator;
    use proptest::prelude::*;

    fn field() -> impl Strategy<Value = Field> {
        any::<u128>().prop_map(Field::from)
    }

    /// Product of `xs`, folding from `1` -- the invariant every grand-product
    /// test below checks against.
    fn product(xs: &[Field]) -> Field {
        xs.iter().fold(Field::ONE, |acc, &x| acc * x)
    }

    /// `leaves` padded with ones to a power of two, so they have an MLE. The
    /// sumcheck tests below take any leaves, not only those a bit table
    /// gives, so they build the layers with `fold_layers` directly.
    fn pad(mut leaves: Vec<Field>) -> Vec<Field> {
        if !leaves.is_empty() {
            leaves.resize(leaves.len().next_power_of_two(), Field::ONE);
        }
        leaves
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
        // fold_layers, unbatched (groups = 1): pairwise multiplication is
        // associative, so folding-by-multiply any witness layer (including
        // the leaves, padded with the multiplicative identity 1) must equal
        // folding the original input. Checks every intermediate layer, not
        // just the top.
        #[test]
        fn eval_preserves_product_across_layers(leaves in prop::collection::vec(field(), 0..12)) {
            let expected = product(&leaves);

            let (top, witnesses) = fold_layers(pad(leaves), 1);

            prop_assert_eq!(product(&top), expected);

            for layer in witnesses {
                prop_assert_eq!(product(&layer), expected);
            }
        }


        #[test]
        fn fold_layers_top_layer_is_interleaved_group_products(
            leaves in prop::collection::vec(field(), 1..32),
            groups_log2 in 0usize..4,
        ) {
            let padded_len = leaves.len().next_power_of_two();
            let groups_log2 = groups_log2.min(padded_len.ilog2() as usize);
            let groups = 1usize << groups_log2;

            let mut padded = leaves.clone();
            padded.resize(padded_len, Field::ONE);

            let (top, _witnesses) = fold_layers(pad(leaves), groups);
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
            let (output, witnesses) = fold_layers(leaves.clone(), 4);
            let claim = mle(output, &point);
            let instance = (leaves.clone(), point.to_vec());
            let mut prover = transcript::build_prover("gkr-zero", &instance);
            let terminal = gpgkr_prove(
                &mut prover,
                leaves.len().ilog2() as usize,
                &point,
                witnesses.into_iter().rev(),
            );
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

    /// Small enough that `BitZParams` accepts every shape below.
    const Q: u128 = (1 << 100) - 15;

    /// The bottom three layers looked up in the tables, and every layer
    /// above, match the circuit built one leaf at a time from the same table.
    #[test]
    fn lut_layers_match_the_leaf_construction() {
        // At t = 7 and 8 the transposed row groups are shorter than a `u128`.
        for (log_rows, log_columns) in [(7, 15), (8, 14), (9, 13), (15, 7)] {
            let shape = Shape::new(log_rows, log_columns).unwrap();
            let params = BitZParams::<Q>::new(shape, smallest_generator()).unwrap();
            let packed: Vec<F128> = (0..(1u128 << shape.log_bits()) / 128)
                .map(|i| {
                    F128::from(i.wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15) ^ 0xA5A5)
                })
                .collect();
            let table = params.table(&packed).unwrap();
            let row_images: Vec<F128> = (0..shape.rows() as u128)
                .map(|b| F128::from((b + 1) * 0x9E37_79B9_7F4A_7C15 + 7))
                .collect();

            // Leaf `(b << s) | c`, as `GrandProductCircuit::new` describes it.
            let mut leaves = vec![Field::ONE; 1 << shape.log_bits()];
            for (b, &image) in row_images.iter().enumerate() {
                for c in 0..shape.columns() {
                    if table.bit(c, b) {
                        leaves[(b << log_columns) | c] = image;
                    }
                }
            }
            let (expected_output, expected_layers) = fold_layers(leaves, shape.columns());

            let (output, witnesses) = GrandProductCircuit::new(&row_images, &table).batched_eval();
            // Witnesses come top-down; the reference is bottom-up.
            let mut layers: Vec<Vec<Field>> = witnesses.into_iter().collect();
            layers.reverse();

            assert!(output == expected_output, "output, t = {log_rows}");
            assert_eq!(layers.len(), expected_layers.len(), "t = {log_rows}");
            for (i, (layer, expected)) in layers.iter().zip(&expected_layers).enumerate() {
                assert!(layer == expected, "layer {i}, t = {log_rows}");
            }
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
        let leaves = pad(input);
        // Fake hashing
        let n = leaves.len() as u128;

        let log_bits = leaves.len().max(1).ilog2() as usize;
        let groups = 1usize << log_groups;
        let (last_value, witnesses) = fold_layers(leaves, groups);

        // TODO instance is a bit loose and should be replaced by PCS
        let instance = (last_value.clone(), n);

        let mut prover = transcript::build_prover("gkr", &instance);

        // Extension point chosen by the verifier to evaluate the output layer.
        // Mirrors `verify`'s `output.len().max(1).ilog2()` exactly, since both
        // sides must draw the same number of challenges here.
        let log_groups = last_value.len().max(1).ilog2();
        let point: Vec<Field> = (0..log_groups).map(|_| prover.verifier_message()).collect();

        gpgkr_prove(&mut prover, log_bits, &point, witnesses.into_iter().rev());
        (last_value, prover.finish())
    }

    pub fn verify(input: Vec<Field>, output: Vec<Field>, proof: transcript::Proof) -> bool {
        let leaves = pad(input);
        // Fake hashing
        let instance = (&output, leaves.len() as u128);
        let mut verifier = transcript::build_verifier("gkr", &instance, &proof);

        let log_groups = output.len().max(1).ilog2();
        let point: Vec<Field> = (0..log_groups)
            .map(|_| verifier.verifier_message())
            .collect();
        let claim = mle(output, &point);
        let log_groups = point.len() as u32;
        let log_leafs = leaves.len().max(1).ilog2();
        let rounds = log_leafs.saturating_sub(log_groups);

        match gpgkr_verify(&mut verifier, claim, &point, rounds) {
            Some((point, claim)) => {
                let leaf_check = mle(leaves, &point);

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
