use common::{BitMatrix, BitTable, matrix::StepIter};
use field::{F128, Wide256};
use num_traits::{ConstOne, ConstZero};
use rayon::prelude::*;
use transcript::{ProverState, VerifierState};

pub type Field = F128;

/// Proves the layer-by-layer sumcheck reduction from a claim at `point`
/// (an evaluation point on the output layer) down to a claim on the leaves.
///
/// Points are little-endian, as in `poly`'s tables: `point[i]` is the
/// coordinate of index bit `i`. Every layer adds one coordinate, at the top.
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
    let mut claim = Field::ZERO;
    let mut eq_storage = EqTables::alloc_storage(log_bits);
    for children in witnesses {
        (point, claim) = prove_layer(ps, &mut eq_storage, &point, children);
    }
    (point, claim)
}

/// Proves `c = Σ_x eq(z, x) · L(x) · R(x)` for the halves `L`, `R` of
/// `children`, one round per coordinate of `point`, top first. Round `i`
/// sends the sum `sum_i` at one endpoint and its leading coefficient, where
/// `s_i(X) = eq(z_i, X) · sum_i(X)` and `sum_i` carries
/// `eq_bound = Π_{j > i} eq(z_j, r_j)` from the rounds already bound.
/// Returns the point `(r, t)` on `children` and the claim
/// `W(r, t) = L(r) + t · (R(r) - L(r))`.
fn prove_layer(
    ps: &mut ProverState,
    storage: &mut [Field],
    point: &[Field],
    mut children: Vec<Field>,
) -> (Vec<Field>, Field) {
    let n = point.len();
    let eq_tables = EqTables::new(storage, point);
    // The top index bit of `children` picks the half: `L`, then `R`.
    let mid = children.len() / 2;
    let (mut left, mut right) = children.split_at_mut(mid);
    let mut next_point = vec![Field::ZERO; n + 1];
    let mut eq_bound = Field::ONE;

    for (i, &z) in point.iter().enumerate().rev() {
        // `x_i = 0` in the low half of each, `x_i = 1` in the high half.
        let half = 1 << i;
        let (left0, left1) = left.split_at_mut(half);
        let (right0, right1) = right.split_at_mut(half);

        // Branch once per round: `round_sums` takes the endpoint as a const
        // generic, so the hot fold has no per-element check.
        let eq = eq_tables.below(i);
        let (sum_end, sum_inf) = if sends_one(z) {
            round_sums::<true>(left0, left1, right0, right1, eq)
        } else {
            round_sums::<false>(left0, left1, right0, right1, eq)
        };
        ps.prover_message(&[eq_bound * sum_end.reduce(), eq_bound * sum_inf.reduce()]);

        let r = ps.verifier_message();
        bind(r, left0, left1, right0, right1);
        left = &mut left[..half];
        right = &mut right[..half];

        next_point[i] = r;
        eq_bound *= eq_factor(r, z);
    }

    let (left_r, right_r) = (left[0], right[0]);
    ps.prover_message(&[left_r, right_r]);
    let t = ps.verifier_message();
    next_point[n] = t;
    (next_point, interpolate(left_r, right_r, t))
}

/// `Σ eq · L(e) · R(e)` and `Σ eq · ΔL · ΔR` over the free variables: the
/// round's sum at the endpoint `e` (`1` if `AT_ONE`) and its leading
/// coefficient, without `eq_bound` and left unreduced.
fn round_sums<const AT_ONE: bool>(
    left0: &[Field],
    left1: &[Field],
    right0: &[Field],
    right1: &[Field],
    eq: &[Field],
) -> (Wide256, Wide256) {
    let at = |v0: Field, v1: Field| if AT_ONE { v1 } else { v0 };

    // The last round has a single summand, and `eq` there is the empty
    // product `1`: no multiplication by it, and too few for two lanes.
    if let [_] = eq {
        let (l0, l1, r0, r1) = (left0[0], left1[0], right0[0], right1[0]);
        return (
            Wide256::mul(at(l0, l1), at(r0, r1)),
            Wide256::mul(l1 - l0, r1 - r0),
        );
    }

    // Two lanes per step: independent multiplication chains. Chunking also
    // reduces variance in benchmarking.
    left0
        .par_chunks_exact(2)
        .zip(left1.par_chunks_exact(2))
        .zip(right0.par_chunks_exact(2))
        .zip(right1.par_chunks_exact(2))
        .zip(eq.par_chunks_exact(2))
        // Chunked by 2, so the element-count threshold below is halved in
        // terms of chunks.
        .with_min_len(PARALLEL_MIN_LANES / 2)
        .fold(
            || (Wide256::zero(), Wide256::zero()),
            |(mut end, mut inf), ((((l0, l1), r0), r1), e)| {
                // `e · L(e) · R(e)` into `end` and `e · ΔL · ΔR` into `inf`,
                // with `e` joining the left factor through a reduced multiply
                // so that the right one can stay wide.
                //
                // Steps, products, then sums, each for both lanes: this order
                // puts every load of the step ahead of the first multiply.
                // Lane by lane is 1.1-1.4% slower single-threaded.
                let (d_l0, d_r0) = (l1[0] - l0[0], r1[0] - r0[0]);
                let (d_l1, d_r1) = (l1[1] - l0[1], r1[1] - r0[1]);
                let (a0, b0) = (e[0] * at(l0[0], l1[0]), e[0] * d_l0);
                let (a1, b1) = (e[1] * at(l0[1], l1[1]), e[1] * d_l1);
                end += Wide256::mul(a0, at(r0[0], r1[0]));
                inf += Wide256::mul(b0, d_r0);
                end += Wide256::mul(a1, at(r0[1], r1[1]));
                inf += Wide256::mul(b1, d_r1);
                (end, inf)
            },
        )
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(a0, ainf), (b0, binf)| (a0 + b0, ainf + binf),
        )
}

/// Binds the variable that splits each pair of halves to `r`, writing the
/// result over the low halves.
fn bind(r: Field, left0: &mut [Field], left1: &[Field], right0: &mut [Field], right1: &[Field]) {
    left0
        .par_iter_mut()
        .zip(right0.par_iter_mut())
        .zip(left1.par_iter())
        .zip(right1.par_iter())
        .with_min_len(PARALLEL_MIN_LANES)
        .for_each(|(((l0, r0), &l1), &r1)| {
            // Recalculating the step rather than writing it in the high half
            // in `round_sums` to save on writes. No benchmarking has been done
            // to check the difference.
            let (d_l, d_r) = (l1 - *l0, r1 - *r0);

            // `interpolate`, spelled out: as `*r0 = interpolate(*r0, r1, r)`
            // the compiler loads `*r0` as a vector and moves it back to
            // general registers for the subtraction, one extra instruction
            // per pair on the way into the multiply (~3.7% single-threaded).
            *l0 += r * d_l;
            *r0 += r * d_r;
        });
}

/// The line through `at0` (at `0`) and `at1` (at `1`), at `r`: a multilinear
/// function of one variable.
fn interpolate(at0: Field, at1: Field, r: Field) -> Field {
    // Narrow multiply, as the next round will multiply with this result.
    at0 + r * (at1 - at0)
}

/// The quadratic through `at0` (at `0`) and `at1` (at `1`) with leading
/// coefficient `lead`, at `r`: `at0 + r·(at1 - at0) + r·(r - 1)·lead`, with
/// `r` factored out.
fn quadratic_at(at0: Field, at1: Field, lead: Field, r: Field) -> Field {
    at0 + r * ((at1 - at0) + (r - Field::ONE) * lead)
}

/// Whether round `i` sends `sum_i(1)` rather than `sum_i(0)`. The verifier
/// recovers the other one from `c = (1 - z_i) · sum_i(0) + z_i · sum_i(1)`,
/// which divides by `z_i` to get `sum_i(1)`; at `z_i = 0` it gives
/// `sum_i(0) = c` directly instead.
fn sends_one(z: Field) -> bool {
    z == Field::ZERO
}

/// `eq(r, z) = r*z + (1 - r)*(1 - z)`. Expanding gives
/// `1 + r + z + 2*r*z`, and in characteristic 2 `2*r*z = r*z + r*z = 0`, so
/// this is just `1 + r + z` -- no multiplication at all, and so nothing for
/// widemul to help with.
fn eq_factor(r: Field, z: Field) -> Field {
    Field::ONE + r + z
}

/// `eq(z_{<i}, ·)` for every round `i` of a layer, in one buffer.
struct EqTables<'a> {
    storage: &'a [Field],
}

impl<'a> EqTables<'a> {
    /// `eq(z_{<i}, x)` for `x ∈ {0,1}^i`, in the order of
    /// `poly::eq::eq_table(&z[..i])`. Table `i` sits at
    /// `2^i - 1 .. 2^(i+1) - 1`.
    fn below(&self, i: usize) -> &[Field] {
        let start = (1 << i) - 1;
        let end = (1 << (i + 1)) - 1;
        &self.storage[start..end]
    }

    /// Allocates all directly as it is as much space as a double buffer approach would take.
    #[inline(never)]
    fn new(storage: &'a mut [Field], point: &[Field]) -> EqTables<'a> {
        // Establishes the whole table's space budget up front so LLVM can
        // prove the `low`/`hi` writes in the loop below are in bounds
        let needed = (1usize << point.len()).saturating_sub(1).max(1);
        assert!(storage.len() >= needed);

        storage[0] = F128::ONE;
        let (mut prev, mut remaining) = storage.split_at_mut(1);
        // Round `i` reads only the coordinates below `i`, so the top one
        // never enters a table.
        let below_top = &point[..point.len().saturating_sub(1)];

        // Each coordinate doubles the table, as its new top index bit.
        for &z in below_top {
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
        EqTables { storage }
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
    layers: u32,
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
    for _ in 0..layers {
        (point, claim) = verify_layer(vs, claim, &point)?
    }
    Some((point, claim))
}

/// Per round, recovers the other endpoint from
/// `c = (1 - z_i) · sum_i(0) + z_i · sum_i(1)` and moves the claim to
/// `s_i(r_i)`.
/// Then checks `c = eq(z, r) · L(r) · R(r)` and continues at
/// `W(r, t) = L(r) + t · (R(r) - L(r))`.
fn verify_layer(
    vs: &mut VerifierState,
    mut claim: Field,
    point: &[Field],
) -> Option<(Vec<Field>, Field)> {
    let n = point.len();
    let mut next_point = vec![Field::ZERO; n + 1];
    let mut eq_bound = Field::ONE;

    for (i, &z) in point.iter().enumerate().rev() {
        let [sum_end, sum_inf]: [Field; 2] = vs.prover_message().ok()?;
        // `claim = (1 - z) * sum0 + z * sum1`.
        let (sum0, sum1) = if sends_one(z) {
            (claim, sum_end)
        } else {
            (sum_end, (claim - (Field::ONE - z) * sum_end) / z)
        };

        let r = vs.verifier_message();
        next_point[i] = r;
        let eq_r = eq_factor(r, z);
        claim = eq_r * quadratic_at(sum0, sum1, sum_inf, r);
        eq_bound *= eq_r;
    }

    // Every coordinate is bound now: `eq_bound = eq(z, r)`.
    let [left_r, right_r]: [Field; 2] = vs.prover_message().ok()?;
    if eq_bound * left_r * right_r != claim {
        return None;
    }
    let t = vs.verifier_message();
    next_point[n] = t;
    Some((next_point, interpolate(left_r, right_r, t)))
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
    ///
    /// Layers 0 to 2 stay as lookup tables, layer 3 is the first one built in
    /// full, and the layers above it fold as usual.
    #[tracing::instrument(name = "Evaluate grand-product circuit", level = "debug", skip_all)]
    pub fn batched_eval(self) -> (Vec<Field>, LayerWitnesses) {
        let lower_lut = lower_luts(self.images);
        let layer3 = layer3(&self.matrix, &lower_lut);
        // A table's run holds one nibble per product tree.
        let trees = self.matrix.dim2() / NIBBLE_BITS;
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

const NIBBLE_BITS: usize = 4;
const NIBBLES_IN_WORD: usize = u64::BITS as usize / NIBBLE_BITS;

/// Layers 0 to 2 as tables, one per four images: `lut[s]` is the layer-2 node
/// whose four leaves are the bits of `s`.
///
/// Each table takes one image from each quarter of `images`.
fn lower_luts(images: &[Field]) -> Vec<[Field; 16]> {
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

    let quart = images.len() / 4;
    let (a, rest) = images.split_at(quart);
    let (b, rest) = rest.split_at(quart);
    let (c, d) = rest.split_at(quart);
    a.iter()
        .zip(b)
        .zip(c)
        .zip(d)
        .map(|(((&a, &b), &c), &d)| create_lut([a, b, c, d]))
        .collect()
}

/// Layer 3, the first layer built in full: node `k` multiplies layer-2 nodes
/// `k` and `k + n/2`, each a table entry looked up by its nibble.
///
/// The matrix and the tables split in the middle into the left and the right
/// child.
fn layer3(matrix: &BitMatrix, lower_lut: &[[Field; 16]]) -> Vec<Field> {
    // Read as `u64`s: on a little-endian target, bit `i` of the words is
    // bit `i` of the `u64`s too.
    const { assert!(cfg!(target_endian = "little")) };
    let bitstring: &[u64] = bytemuck::cast_slice(&matrix[..]);
    let (left, right) = bitstring.split_at(bitstring.len() / 2);
    let (left_luts, right_luts) = lower_lut.split_at(lower_lut.len() / 2);

    // Each table serves one run of whole words, so a pair of tables is
    // picked once and its nibbles are read a word at a time.
    // A table's run is one `xs[i1]` of the matrix, a nibble per node.
    let lut_words = left.len() / left_luts.len(); // This introduces a bound. StepIterator can prevent that but is more expensive than doing it this way

    assert!(lut_words > 0, "each table needs at least a word of nibbles");
    let runs = left
        .chunks_exact(lut_words)
        .zip(right.chunks_exact(lut_words))
        .zip(left_luts.iter().zip(right_luts));

    // The layer-2 nodes whose four leaves are the nibbles of `word`.
    fn layer2(lut: &[Field; 16], word: u64) -> impl Iterator<Item = Field> + '_ {
        (0..NIBBLES_IN_WORD).map(move |i| lut[(word >> (NIBBLE_BITS * i)) as usize & 15])
    }

    // One node per nibble of each left table.
    let mut layer3 = Vec::with_capacity(left_luts.len() * lut_words * NIBBLES_IN_WORD);
    for ((l_words, r_words), (l_lut, r_lut)) in runs {
        for (&l, &r) in l_words.iter().zip(r_words) {
            fold_into(&mut layer3, layer2(l_lut, l), layer2(r_lut, r));
        }
    }
    layer3
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
        let mut next = Vec::with_capacity(l.len());
        fold_into(&mut next, l.iter().copied(), r.iter().copied());
        witnesses.push(layer);
        layer = next;
    }

    (layer, witnesses)
}

/// One layer up, appended to `next`: node `k` multiplies node `k` of the left
/// half and node `k` of the right half below it.
fn fold_into(
    next: &mut Vec<Field>,
    left: impl Iterator<Item = Field>,
    right: impl Iterator<Item = Field>,
) {
    // `extend` from zipped ranges or slices knows its length, so there is no
    // capacity check or length update per node.
    next.extend(left.zip(right).map(|(a, b)| a * b));
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
    /// `q` alone, layer-1 half `h` pairs bits `h` and `h + 2`, and layer 2
    /// takes all four.
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
            for h in 0..2 {
                layer1[h * n + k] = lut[bs & (0b0101 << h)];
            }
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
