use common::{BitMatrix, BitTable, matrix::StepIter};
use field::{F128, Wide256};
use num_traits::{ConstOne, ConstZero};
use rayon::prelude::*;
use std::ops::Range;
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
    witnesses: LayerWitnesses,
) -> (Vec<F128>, Field) {
    // Edge cases
    // - empty witnesses -> single constant circuit -> one verifier message that permutes the proof state, but a single constant can't have an MLE
    let mut point = point.to_owned();
    let mut eq_memory = EqTables::alloc_storage(log_bits);
    let LayerWitnesses {
        lut_layers,
        upper_nodes,
    } = witnesses;
    for children in upper_nodes.into_iter().rev() {
        (point, _) = prove_layer(ps, &mut eq_memory, &point, children);
    }

    // Layers 2, 1 and 0 stay in their lookup tables. Round 1 of each writes the
    // children it has bound, `[left′ | right′]`, into one allocation sized
    // for the leaves'.
    // number of nodes at the leaf level is lut_layer.nodes() * 4. We only need half of that since we materialise after round 1.
    let mut bound_children = Field::zeroed_vec(2 * lut_layers.nodes());
    let (point, _) =
        prove_lut_layer::<LAYER2>(ps, &mut eq_memory, &point, &lut_layers, &mut bound_children);
    let (point, _) =
        prove_lut_layer::<LAYER1>(ps, &mut eq_memory, &point, &lut_layers, &mut bound_children);
    prove_lut_layer::<LEAVES>(ps, &mut eq_memory, &point, &lut_layers, &mut bound_children)
}

/// Proves `c = Σ_x eq(z, x) · L(x) · R(x)` for the halves `L`, `R` of
/// `children`, one round per coordinate of `point`, top first. Round `i`
/// sends `sum_i(b)` and its leading coefficient `sum_i(∞)`, where `b` is the
/// boolean point that [`sends_one`] picks, `s_i(X) = eq(z_i, X) · sum_i(X)`,
/// and `sum_i` carries `eq_bound = Π_{j > i} eq(z_j, r_j)` from the rounds
/// already bound. Returns the point `(r, t)` on `children` and the claim
/// `W(r, t) = L(r) + t · (R(r) - L(r))`.
fn prove_layer(
    ps: &mut ProverState,
    eq_memory: &mut [Field],
    point: &[Field],
    mut children: Vec<Field>,
) -> (Vec<Field>, Field) {
    let eq_tables = EqTables::new(eq_memory, point);
    // The top index bit of `children` picks the half: `L`, then `R`.
    let mid = children.len() / 2;
    let (left, right) = children.split_at_mut(mid);
    prove_rounds(ps, &eq_tables, point, None, left, right)
}

/// The dense rounds of a layer, then its last message. A LUT-backed layer
/// has already proved round 1, which binds the top coordinate of `point`, and
/// passes that round's challenge as `round_one`; its `left` and `right` are
/// then the bound halves. A dense layer passes `None` and proves every round
/// here.
fn prove_rounds(
    ps: &mut ProverState,
    eq_tables: &EqTables,
    point: &[Field],
    round_one: Option<Field>,
    mut left: &mut [Field],
    mut right: &mut [Field],
) -> (Vec<Field>, Field) {
    let n = point.len();
    let mut next_point = vec![Field::ZERO; n + 1];
    // The coordinates whose rounds run here, and `eq` of those already bound.
    let (remaining, mut eq_bound) = match round_one {
        Some(r) => {
            let top = n - 1;
            next_point[top] = r;
            (&point[..top], eq_factor(r, point[top]))
        }
        None => (point, Field::ONE),
    };

    for (i, &z) in remaining.iter().enumerate().rev() {
        // `x_i = 0` in the low half of each, `x_i = 1` in the high half.
        let half = 1 << i;
        let (left0, left1) = left.split_at_mut(half);
        let (right0, right1) = right.split_at_mut(half);

        // Branch once per round: `round_sums` takes `b` as a const generic,
        // so the hot fold has no per-element check.
        let eq = eq_tables.below(i);
        let (sum_b, sum_inf) = if sends_one(z) {
            round_sums::<true>(left0, left1, right0, right1, eq)
        } else {
            round_sums::<false>(left0, left1, right0, right1, eq)
        };
        ps.prover_message(&[eq_bound * sum_b.reduce(), eq_bound * sum_inf.reduce()]);

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

/// `Σ eq · L(b) · R(b)` and `Σ eq · ΔL · ΔR` over the free variables: the
/// round's sum at the boolean point `b` (`1` if `AT_ONE`) and its leading
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
            |(mut sum_b, mut sum_inf), ((((l0, l1), r0), r1), e)| {
                // `e · L(b) · R(b)` into `sum_b` and `e · ΔL · ΔR` into
                // `sum_inf`, with `e` joining the left factor through a reduced
                // multiply so that the right one can stay wide.
                //
                // Steps, products, then sums, each for both lanes: this order
                // puts every load of the step ahead of the first multiply.
                // Lane by lane is 1.1-1.4% slower single-threaded.
                let (d_l0, d_r0) = (l1[0] - l0[0], r1[0] - r0[0]);
                let (d_l1, d_r1) = (l1[1] - l0[1], r1[1] - r0[1]);
                let (a0, b0) = (e[0] * at(l0[0], l1[0]), e[0] * d_l0);
                let (a1, b1) = (e[1] * at(l0[1], l1[1]), e[1] * d_l1);
                sum_b += Wide256::mul(a0, at(r0[0], r1[0]));
                sum_inf += Wide256::mul(b0, d_r0);
                sum_b += Wide256::mul(a1, at(r0[1], r1[1]));
                sum_inf += Wide256::mul(b1, d_r1);
                (sum_b, sum_inf)
            },
        )
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(x_b, x_inf), (y_b, y_inf)| (x_b + y_b, x_inf + y_inf),
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

/// The LUT-backed layers, by their index in the tree. Each names the
/// sumcheck whose children it is.
const LEAVES: usize = 0;
const LAYER1: usize = 1;
const LAYER2: usize = 2;

/// Where round 1 of a LUT-backed sumcheck reads its operands `l0, l1, r0,
/// r1`, by the layer of its children: operand `o` at position `k` is nibble
/// `k + STREAM[o] · stride`, masked by `MASK[o]`, in that nibble's lookup
/// table. [`LutLayers::pairs`] says why.
const STREAM: [[usize; 4]; 3] = [[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 2, 3]];
const MASK: [[usize; 4]; 3] = [
    [0b0001, 0b0010, 0b0100, 0b1000],
    [0b0101, 0b0101, 0b1010, 0b1010],
    [0b1111; 4],
];

/// Layers 0 to 2 in their lookup tables, as round 1 of their sumchecks reads
/// them: layer-2 node `k = (g << s) | c` is nibble `k` of the matrix, looked
/// up in the lookup table of node position `g`, which all `2^s` trees share.
struct LutLayers {
    /// The nibbles of the layer-2 nodes; see [`GrandProductCircuit::new`].
    matrix: BitMatrix<'static>,
    luts: Vec<[Field; 16]>,
    /// `s`: a lookup table's run of nibbles holds one per tree.
    log_trees: usize,
}

impl LutLayers {
    fn new(matrix: BitMatrix<'static>, luts: Vec<[Field; 16]>) -> Self {
        let log_trees = (matrix.dim2() / NIBBLE_BITS).ilog2() as usize;
        // `pairs` reads whole words of one lookup table's run.
        assert!(
            1 << log_trees >= NIBBLES_IN_WORD,
            "each lookup table needs at least a word of nibbles"
        );
        Self {
            matrix,
            luts,
            log_trees,
        }
    }

    /// The nibbles, a word at a time.
    fn words(&self) -> &[u64] {
        // Read as `u64`s: on a little-endian target, bit `i` of the words is
        // bit `i` of the `u64`s too.
        const { assert!(cfg!(target_endian = "little")) };
        bytemuck::cast_slice(&self.matrix[..])
    }

    /// The layer-2 nodes, one per nibble.
    fn nodes(&self) -> usize {
        self.words().len() * NIBBLES_IN_WORD
    }

    /// The positions of round 1 handled as one piece: a lookup table's run,
    /// or a part of it, so that a piece never spans two lookup tables.
    fn piece(&self) -> usize {
        (1 << self.log_trees).min(PARALLEL_MIN_LANES)
    }

    /// Round 1's positions `0..half` in pieces, as a parallel iterator
    /// taking at least `PARALLEL_MIN_LANES` positions per task, whatever the
    /// number of trees.
    fn pieces(&self, half: usize) -> impl IndexedParallelIterator<Item = Range<usize>> + use<> {
        let piece = self.piece();
        (0..half / piece)
            .into_par_iter()
            .with_min_len(PARALLEL_MIN_LANES / piece)
            .map(move |p| p * piece..(p + 1) * piece)
    }

    /// The pairs that round 1 binds at `positions`, which lie in one lookup
    /// table's run and start and end on a word: per position, the nibbles of
    /// its operands `l0, l1, r0, r1`, unmasked, and each operand's lookup
    /// table.
    ///
    /// Round 1 binds the top bit of the parent's index, so each node pairs
    /// with its partner `stride` nodes later, where `stride` is the number of
    /// round-1 positions:
    /// - Layer 2: a nibble is one node. The left pair is nibbles `k` and
    ///   `k + stride`, and the right pair the two half the layer on,
    ///   `k + 2·stride` and `k + 3·stride`.
    /// - Layer 1: a nibble holds a left node, bits 0 and 2, and a right node,
    ///   bits 1 and 3. Both of their partners sit in nibble `k + stride`.
    /// - Leaves: a nibble holds both pairs, bits 0 and 1 for the left child
    ///   and bits 2 and 3 for the right, so every operand reads nibble `k`.
    ///
    /// The words each operand reads are zipped, and every tuple of words
    /// gives its 16 positions, one item per word. Callers zip the words with
    /// 16-position chunks of their own data and loop over the positions
    /// inside: flattened here and zipped per position, the loop becomes a
    /// per-position state machine that LLVM does not unroll.
    fn pairs<const LAYER: usize>(
        &self,
        stride: usize,
        positions: Range<usize>,
    ) -> (
        [&[Field; 16]; 4],
        impl Iterator<Item = impl Iterator<Item = [usize; 4]>> + '_,
    ) {
        let (words, len) = (self.words(), positions.len() / NIBBLES_IN_WORD);
        let at = STREAM[LAYER].map(|q| positions.start + q * stride);
        let luts = at.map(|k| &self.luts[k >> self.log_trees]);
        let [l0, l1, r0, r1] = at.map(|k| &words[k / NIBBLES_IN_WORD..][..len]);
        let pairs = l0
            .iter()
            .zip(l1)
            .zip(r0)
            .zip(r1)
            .map(|(((&l0, &l1), &r0), &r1)| {
                (0..NIBBLES_IN_WORD).map(move |i| {
                    [l0, l1, r0, r1].map(|word| (word >> (NIBBLE_BITS * i)) as usize & 0xF)
                })
            });
        (luts, pairs)
    }
}

/// The operands `[l0, l1, r0, r1]` of one position of round 1 of layer
/// `LAYER`, from their lookup tables and nibbles.
fn operands<const LAYER: usize>(luts: &[&[Field; 16]; 4], nibbles: [usize; 4]) -> [Field; 4] {
    std::array::from_fn(|o| luts[o][nibbles[o] & MASK[LAYER][o]])
}

/// Proves the layer whose children are layer `LAYER` of the lookup tables,
/// like [`prove_layer`]. Round 1 reads its operands from the lookup tables
/// and writes only the children it has bound, `[left′ | right′]`, into
/// `bound_children`; the rounds after it are dense.
fn prove_lut_layer<const LAYER: usize>(
    ps: &mut ProverState,
    eq_memory: &mut [Field],
    point: &[Field],
    lut_layers: &LutLayers,
    bound_children: &mut [Field],
) -> (Vec<Field>, Field) {
    let eq_tables = EqTables::new(eq_memory, point);
    let top = point.len() - 1;

    // Nothing is bound before round 1, so its sums go out as they are.
    let (sum_b, sum_inf) = if sends_one(point[top]) {
        round_one_sums::<LAYER, true>(lut_layers, &eq_tables, point)
    } else {
        round_one_sums::<LAYER, false>(lut_layers, &eq_tables, point)
    };
    ps.prover_message(&[sum_b.reduce(), sum_inf.reduce()]);

    let r = ps.verifier_message();
    let half = 1 << top;
    let (left, right) = bound_children[..2 * half].split_at_mut(half);
    lut_bind::<LAYER>(r, lut_layers, left, right);
    prove_rounds(ps, &eq_tables, point, Some(r), left, right)
}

/// Round 1's sum at the boolean point `b` (`1` if `AT_ONE`) and its leading
/// coefficient, unreduced, as [`round_sums`] gives them. The leaves and
/// layer 1 sum their eq weights in buckets. Layer 2 pairs whole nibbles of
/// two lookup tables, 256 keys per product, which costs more to finish than
/// a piece saves at small `s`, so it works node by node.
fn round_one_sums<const LAYER: usize, const AT_ONE: bool>(
    lut_layers: &LutLayers,
    eq_tables: &EqTables,
    point: &[Field],
) -> (Wide256, Wide256) {
    match LAYER {
        LEAVES => bucket_sums(lut_layers, eq_tables, point, leaf_buckets::<AT_ONE>),
        LAYER1 => bucket_sums(lut_layers, eq_tables, point, layer1_buckets::<AT_ONE>),
        _ => layer2_sums::<LAYER, AT_ONE>(lut_layers, eq_tables.below(point.len() - 1)),
    }
}

/// Round 1's sums a piece at a time, with `buckets` summing each piece's eq
/// weights by the nibble bits that `L·R` reads and multiplying each bucket
/// once. The weight splits as `eq(z_{<top}, k) = eq_hi[g] · eq_lo[c]` for
/// `k = (g << s) | c`, so the buckets sum `eq_lo`, which stays in cache, and
/// `eq_hi` joins once per piece.
fn bucket_sums(
    lut_layers: &LutLayers,
    eq_tables: &EqTables,
    point: &[Field],
    buckets: impl Fn(&LutLayers, usize, Range<usize>, &[Field]) -> (Wide256, Wide256) + Sync,
) -> (Wide256, Wide256) {
    let top = point.len() - 1;
    let (half, s) = (1 << top, lut_layers.log_trees);
    let eq_lo = eq_tables.below(s);
    let eq_hi = poly::eq_table(&point[s..top]);
    lut_layers
        .pieces(half)
        .map(|positions| {
            let h = eq_hi[positions.start >> s];
            let eq_lo = &eq_lo[positions.start % (1 << s)..][..positions.len()];
            let (sum_b, sum_inf) = buckets(lut_layers, half, positions, eq_lo);
            (
                Wide256::mul(h, sum_b.reduce()),
                Wide256::mul(h, sum_inf.reduce()),
            )
        })
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(x_b, x_inf), (y_b, y_inf)| (x_b + y_b, x_inf + y_inf),
        )
}

/// The piece of round 1 for the leaves at `positions`, its weights in `eq`.
/// Every operand is one bit of the same nibble, so both sums are functions of
/// the nibble: each position adds its weight into its nibble's bucket, and
/// each of the 16 buckets is multiplied once.
fn leaf_buckets<const AT_ONE: bool>(
    lut_layers: &LutLayers,
    stride: usize,
    positions: Range<usize>,
    eq: &[Field],
) -> (Wide256, Wide256) {
    let ([lut, ..], pairs) = lut_layers.pairs::<LEAVES>(stride, positions);
    let mut bucket = [Field::ZERO; 16];
    for (word, eq) in pairs.zip(eq.chunks_exact(NIBBLES_IN_WORD)) {
        for ([a, ..], &e) in word.zip(eq) {
            bucket[a] += e;
        }
    }
    // `L(b)·R(b)` is the node of bits 0 and 2, or 1 and 3, and `L(∞)·R(∞)` is
    // `(l1 − l0)(r1 − r0)`.
    let b_mask = if AT_ONE { 0b1010 } else { 0b0101 };
    let at_b = std::array::from_fn(|a| lut[a & b_mask]);
    let at_inf = std::array::from_fn(|a| {
        (lut[a & 0b0010] - lut[a & 0b0001]) * (lut[a & 0b1000] - lut[a & 0b0100])
    });
    (dot(&bucket, &at_b), dot(&bucket, &at_inf))
}

/// The piece of round 1 for layer 1 at `positions`, its weights in `eq`:
/// nibble `n0` and the nibble `n1` half the layer later. In characteristic 2,
/// `(l1 − l0)(r1 − r0)` is `l0·r0 + l1·r1 + l1·r0 + l0·r1`, and each product
/// reads two disjoint 2-bit masks: 16 buckets each, keyed by the bits it
/// reads.
fn layer1_buckets<const AT_ONE: bool>(
    lut_layers: &LutLayers,
    stride: usize,
    positions: Range<usize>,
    eq: &[Field],
) -> (Wide256, Wide256) {
    // `l0` and `r0` read nibble `n0` in its lookup table, `l1` and `r1`
    // nibble `n1`.
    let ([lut0, lut1, ..], pairs) = lut_layers.pairs::<LAYER1>(stride, positions);
    let mut bucket = [[Field::ZERO; 16]; 4];
    // A position's keys in the buckets of `l0·r0`, `l1·r1`, `l1·r0` and
    // `l0·r1`: bits 0 and 2 from the nibble of the `l` factor, bits 1 and 3
    // from that of the `r` factor.
    for (word, eq) in pairs.zip(eq.chunks_exact(NIBBLES_IN_WORD)) {
        for ([n0, n1, ..], &e) in word.zip(eq) {
            bucket[0][n0] += e;
            bucket[1][n1] += e;
            bucket[2][(n1 & 0b0101) | (n0 & 0b1010)] += e;
            bucket[3][(n0 & 0b0101) | (n1 & 0b1010)] += e;
        }
    }
    // `l0·r0`, `l1·r1`, `l1·r0` and `l0·r1` at every key.
    let product: [[Field; 16]; 4] = [
        *lut0,
        *lut1,
        std::array::from_fn(|x| lut1[x & 0b0101] * lut0[x & 0b1010]),
        std::array::from_fn(|x| lut0[x & 0b0101] * lut1[x & 0b1010]),
    ];
    // Product `b` is `l_b·r_b`.
    let b = usize::from(AT_ONE);
    let sum_inf = (0..4).fold(Wide256::zero(), |acc, j| acc + dot(&bucket[j], &product[j]));
    (dot(&bucket[b], &product[b]), sum_inf)
}

/// `Σ bucket · value`, unreduced.
fn dot(bucket: &[Field; 16], value: &[Field; 16]) -> Wide256 {
    bucket
        .iter()
        .zip(value)
        .fold(Wide256::zero(), |acc, (&b, &v)| acc + Wide256::mul(b, v))
}

/// Round 1's sums node by node, as [`round_sums`] forms them, with the
/// operands read from the lookup tables.
fn layer2_sums<const LAYER: usize, const AT_ONE: bool>(
    lut_layers: &LutLayers,
    eq: &[Field],
) -> (Wide256, Wide256) {
    let at = |v0: Field, v1: Field| if AT_ONE { v1 } else { v0 };
    let half = eq.len();
    lut_layers
        .pieces(half)
        .map(|positions| {
            let (luts, pairs) = lut_layers.pairs::<LAYER>(half, positions.clone());
            let eq = eq[positions].chunks_exact(NIBBLES_IN_WORD);
            pairs
                .zip(eq)
                .fold((Wide256::zero(), Wide256::zero()), |sums, (word, eq)| {
                    word.zip(eq).fold(sums, |(sum_b, sum_inf), (n, &e)| {
                        let [l0, l1, r0, r1] = operands::<LAYER>(&luts, n);
                        (
                            sum_b + Wide256::mul(e * at(l0, l1), at(r0, r1)),
                            sum_inf + Wide256::mul(e * (l1 - l0), r1 - r0),
                        )
                    })
                })
        })
        .reduce(
            || (Wide256::zero(), Wide256::zero()),
            |(x_b, x_inf), (y_b, y_inf)| (x_b + y_b, x_inf + y_inf),
        )
}

/// Round 1's bind, `left′[k] = l0 + r·(l1 − l0)` and
/// `right′[k] = r0 + r·(r1 − r0)`, with the operands read from the lookup
/// tables.
///
/// What position `k` combines depends on the layer; see [`LutLayers::pairs`]:
/// - Layer 2: four nibbles, one node each, the left pair at `k` and
///   `k + stride` and the right pair half the layer on.
/// - Layer 1: two nibbles, `k` and `k + stride`, each holding a left and a
///   right node, so one scan binds both without jumping ahead.
/// - Leaves: one nibble, which holds both pairs.
fn lut_bind<const LAYER: usize>(
    r: Field,
    lut_layers: &LutLayers,
    left: &mut [Field],
    right: &mut [Field],
) {
    let (half, piece) = (left.len(), lut_layers.piece());
    left.par_chunks_mut(piece)
        .zip(right.par_chunks_mut(piece))
        .zip(lut_layers.pieces(half))
        .for_each(|((left, right), positions)| {
            let (luts, pairs) = lut_layers.pairs::<LAYER>(half, positions);
            let left = left.chunks_exact_mut(NIBBLES_IN_WORD);
            let right = right.chunks_exact_mut(NIBBLES_IN_WORD);
            for ((left, right), word) in left.zip(right).zip(pairs) {
                for ((l, rt), n) in left.iter_mut().zip(right).zip(word) {
                    let [l0, l1, r0, r1] = operands::<LAYER>(&luts, n);
                    (*l, *rt) = (interpolate(l0, l1, r), interpolate(r0, r1, r));
                }
            }
        });
}

/// The line through `at0` (at `0`) and `at1` (at `1`), at `r`: a multilinear
/// function of one variable.
fn interpolate(at0: Field, at1: Field, r: Field) -> Field {
    // Narrow multiply, as the next round will multiply with this result.
    at0 + r * (at1 - at0)
}

/// The quadratic through `at0` (at `0`) and `at1` (at `1`) with leading
/// coefficient `at_inf`, at `r`: `at0 + r·(at1 - at0) + r·(r - 1)·at_inf`, with
/// `r` factored out.
fn quadratic_at(at0: Field, at1: Field, at_inf: Field, r: Field) -> Field {
    at0 + r * ((at1 - at0) + (r - Field::ONE) * at_inf)
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
///
/// The former `SuffixTable` with the point little-endian instead of reversed:
/// the summed coordinates are the prefix `z[..i]`, not a suffix. Same entries.
///
/// Keeping every table takes twice the memory of the largest one. Keeping
/// only the largest and halving it between rounds (`low[k] += high[k]`)
/// would save that half, but it measured 7.5-12% slower, as it adds a pass
/// over memory. Splitting the table makes the trade moot.
// TODO: Split the table.
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

/// Per round, recovers the sum at the other boolean point from
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
        let [sum_b, sum_inf]: [Field; 2] = vs.prover_message().ok()?;
        // `claim = (1 - z) * sum0 + z * sum1`.
        let (sum0, sum1) = if sends_one(z) {
            (claim, sum_b)
        } else {
            (sum_b, (claim - (Field::ONE - z) * sum_b) / z)
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
        // The bottom three layers are looked up in 16-entry lookup tables, one
        // per four rows, so the top two row bits are split off and moved beside
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
        let lut_layers = LutLayers::new(self.matrix, lower_luts(self.images));
        let layer3 = layer3(&lut_layers);
        let (output, upper_nodes) = fold_layers(layer3, 1 << lut_layers.log_trees);

        (
            output,
            LayerWitnesses {
                lut_layers,
                upper_nodes,
            },
        )
    }
}

const NIBBLE_BITS: usize = 4;
const NIBBLES_IN_WORD: usize = u64::BITS as usize / NIBBLE_BITS;

/// Layers 0 to 2 as lookup tables, one per four images: `lut[s]` is the
/// layer-2 node whose four leaves are the bits of `s`.
///
/// Each lookup table takes one image from each quarter of `images`.
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
    a.par_iter()
        .zip(b)
        .zip(c)
        .zip(d)
        // A lookup table writes 16 entries.
        .with_min_len(PARALLEL_MIN_LANES / 16)
        .map(|(((&a, &b), &c), &d)| create_lut([a, b, c, d]))
        .collect()
}

/// Layer 3, the first layer built in full: node `k` multiplies layer-2 nodes
/// `k` and `k + n/2`, each a lookup-table entry looked up by its nibble.
/// Those are the two nibbles that round 1 of layer 1 reads at position `k`,
/// here read whole.
fn layer3(lut_layers: &LutLayers) -> Vec<Field> {
    let half = lut_layers.nodes() / 2;
    let mut layer3 = Field::zeroed_vec(half);
    layer3
        .par_chunks_mut(lut_layers.piece())
        .zip(lut_layers.pieces(half))
        .for_each(|(nodes, positions)| {
            let ([lut0, lut1, ..], pairs) = lut_layers.pairs::<LAYER1>(half, positions);
            for (nodes, word) in nodes.chunks_exact_mut(NIBBLES_IN_WORD).zip(pairs) {
                for (node, [n0, n1, ..]) in nodes.iter_mut().zip(word) {
                    *node = lut0[n0] * lut1[n1];
                }
            }
        });
    layer3
}

/// Each layer-2 node of a run of row groups as the lookup table of its row
/// group and the nibble of its four leaves, in node order: node
/// `k = (b' << s) | c` reads nibble `k`, and all `2^s` columns of row group
/// `b'` share its lookup table. `nibbles` and `luts` cover the same row
/// groups.
fn lut_nodes<'a>(
    nibbles: &'a [u128],
    luts: &'a [[Field; 16]],
) -> impl ExactSizeIterator<Item = (&'a [Field; 16], usize)> {
    let nibbles = StepIter::<4>::new(nibbles.iter());
    // Every lookup table serves the same run of nibbles, one per column.
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
        // Node `k` multiplies node `k` of the left half and node `k` of the
        // right half below it.
        let (l, r) = layer.split_at(layer.len() / 2);
        let next = l
            .par_iter()
            .zip(r)
            .with_min_len(PARALLEL_MIN_LANES)
            .map(|(&a, &b)| a * b)
            .collect();
        witnesses.push(layer);
        layer = next;
    }

    (layer, witnesses)
}

/// The layers below the output, yielded from the top down. The bottom three
/// are kept as the lookup tables they are looked up in, the rest in full.
// TODO: This can be a single vec that is allocated at the start.
pub struct LayerWitnesses {
    /// Layers 0 to 2, in their lookup tables.
    lut_layers: LutLayers,
    /// Layer 3 and up, layer 3 first.
    upper_nodes: Vec<Vec<Field>>,
}

impl LayerWitnesses {
    /// The leaves and layers 1 and 2, looked up in full. A node is the
    /// product of the nibble bits its subtree covers: leaf quarter `q` is bit
    /// `q` alone, layer-1 half `h` pairs bits `h` and `h + 2`, and layer 2
    /// takes all four.
    fn lower_layers(&self) -> [Vec<Field>; 3] {
        let nodes = lut_nodes(&self.lut_layers.matrix, &self.lut_layers.luts);
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
// are rebuilt from the lookup tables here; layer 3 and up are already in
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
    /// `fold_layers` tests below take any leaves, not only those a bit table
    /// gives.
    fn pad(mut leaves: Vec<Field>) -> Vec<Field> {
        if !leaves.is_empty() {
            leaves.resize(leaves.len().next_power_of_two(), Field::ONE);
        }
        leaves
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
    }

    /// Small enough that `BitZParams` accepts every shape below.
    const Q: u128 = (1 << 100) - 15;

    /// The smallest shape the gates accept, `(t, s)` with `t + s = 22`.
    const SMALLEST: (usize, usize) = (7, 15);

    /// How the bits of a test table are set.
    #[derive(Clone, Copy, Debug)]
    enum Bits {
        /// Every leaf is one.
        Clear,
        /// Every leaf is its row's image.
        Set,
        /// Bits drawn from the seed.
        Drawn,
    }

    fn bits() -> impl Strategy<Value = Bits> {
        prop_oneof![Just(Bits::Clear), Just(Bits::Set), Just(Bits::Drawn)]
    }

    /// A splitmix64 stream: test data that is fixed by its seed.
    fn stream(seed: u64) -> impl Iterator<Item = u64> {
        let mut x = seed;
        std::iter::repeat_with(move || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let z = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        })
    }

    fn element(words: &mut impl Iterator<Item = u64>) -> F128 {
        let mut word = || words.next().unwrap();
        F128::from([word(), word()])
    }

    /// The packed bits of a `(t, s)` table and one image per row, from `seed`.
    fn instance((t, s): (usize, usize), bits: Bits, seed: u64) -> (Shape, Vec<F128>, Vec<Field>) {
        let shape = Shape::new(t, s).unwrap();
        let mut words = stream(seed);
        let packed = (0..(1usize << shape.log_bits()) / 128)
            .map(|_| match bits {
                Bits::Clear => F128::ZERO,
                Bits::Set => F128::from(u128::MAX),
                Bits::Drawn => element(&mut words),
            })
            .collect();
        let row_images = (0..shape.rows()).map(|_| element(&mut words)).collect();
        (shape, packed, row_images)
    }

    fn table(shape: Shape, packed: &[F128]) -> BitTable<'_> {
        BitZParams::<Q>::new(shape, smallest_generator())
            .unwrap()
            .table(packed)
            .unwrap()
    }

    /// Leaf `(b << s) | c` is `row_images[b]` where `bit(c, b)` is set and one
    /// elsewhere, as `GrandProductCircuit::new` describes, built one leaf at a
    /// time: the reference the circuit is checked against.
    fn leaves(table: &BitTable, row_images: &[Field]) -> Vec<Field> {
        let shape = table.shape();
        let mut leaves = vec![Field::ONE; 1 << shape.log_bits()];
        for (b, &image) in row_images.iter().enumerate() {
            for c in 0..shape.columns() {
                if table.bit(c, b) {
                    leaves[(b << shape.log_columns()) | c] = image;
                }
            }
        }
        leaves
    }

    proptest! {
        // Every case proves a whole circuit at the smallest shape.
        #![proptest_config(ProptestConfig::with_cases(4))]

        // Full protocol: the verifier accepts, and the output is the product
        // of each column's leaves.
        #[test]
        fn gpgkr_round_trip(bits in bits(), seed in any::<u64>()) {
            let (shape, packed, row_images) = instance(SMALLEST, bits, seed);
            let table = table(shape, &packed);
            let leaves = leaves(&table, &row_images);

            let (output, proof) = prove(&table, &row_images);
            prop_assert!(verify(&leaves, &output, &proof));

            let columns = shape.columns();
            let expected: Vec<Field> = (0..columns)
                .map(|c| leaves[c..].iter().step_by(columns).product())
                .collect();
            prop_assert_eq!(output, expected);
        }

        // negative/soundness test
        #[test]
        fn gpgkr_verify_rejects_tampered_output(bits in bits(), seed in any::<u64>()) {
            let (shape, packed, row_images) = instance(SMALLEST, bits, seed);
            let table = table(shape, &packed);
            let (output, proof) = prove(&table, &row_images);

            let mut tampered = output;
            tampered[0] += Field::ONE;

            prop_assert!(!verify(&leaves(&table, &row_images), &tampered, &proof));
        }
    }

    // Zero and one coordinates in the output point take the `sends_one`
    // branch in the top layer's rounds.
    #[test]
    fn gpgkr_round_trip_with_zero_coordinates() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 1);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);
        let s = shape.log_columns();
        let layers = shape.log_rows() as u32;

        // `value` at coordinate `i`, and a nonzero value at every other.
        let nonzero = Field::from(5u128);
        let with = |i: usize, value: Field| -> Vec<Field> {
            (0..s)
                .map(|j| if j == i { value } else { nonzero })
                .collect()
        };
        for point in [
            vec![Field::ZERO; s],
            with(s - 1, Field::ZERO),
            with(0, Field::ZERO),
            with(s - 1, Field::ONE),
            with(0, Field::ONE),
        ] {
            let (output, witnesses) = GrandProductCircuit::new(&row_images, &table).batched_eval();
            let claim = mle(output, &point);
            let mut prover = transcript::build_prover("gkr-zero", &point);
            let terminal = gpgkr_prove(&mut prover, shape.log_bits(), &point, witnesses);
            assert_eq!(terminal.1, mle(leaves.clone(), &terminal.0));
            let proof = prover.finish();

            let mut verifier = transcript::build_verifier("gkr-zero", &point, &proof);
            assert_eq!(
                gpgkr_verify(&mut verifier, claim, &point, layers),
                Some(terminal)
            );
            verifier.check_eof().unwrap();

            let mut verifier = transcript::build_verifier("gkr-zero", &point, &proof);
            assert!(gpgkr_verify(&mut verifier, claim + Field::ONE, &point, layers).is_none());
        }
    }

    /// Each LUT-backed layer proved on its own, from a point whose top
    /// coordinate is zero (round 1's `sends_one` branch) or not: the verifier
    /// accepts it, and the final claim is the MLE of the children.
    #[test]
    fn lut_layers_prove_from_any_top_coordinate() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 4);
        let table = table(shape, &packed);
        // The leaves and every layer above, built one leaf at a time.
        let (_, layers) = fold_layers(leaves(&table, &row_images), shape.columns());
        let (_, witnesses) = GrandProductCircuit::new(&row_images, &table).batched_eval();
        let lut_layers = witnesses.lut_layers;
        let mut bound_children = Field::zeroed_vec(2 * lut_layers.nodes());
        let mut eq_memory = EqTables::alloc_storage(shape.log_bits());

        let mut draws = stream(5);
        for layer in [LEAVES, LAYER1, LAYER2] {
            type Prove = fn(
                &mut ProverState,
                &mut [Field],
                &[Field],
                &LutLayers,
                &mut [Field],
            ) -> (Vec<Field>, Field);
            let prove: Prove = match layer {
                LEAVES => prove_lut_layer::<LEAVES>,
                LAYER1 => prove_lut_layer::<LAYER1>,
                _ => prove_lut_layer::<LAYER2>,
            };
            let (children, parent) = (&layers[layer], &layers[layer + 1]);
            for top in [Field::ZERO, Field::from(7u128)] {
                let mut point: Vec<Field> = (0..parent.len().ilog2())
                    .map(|_| element(&mut draws))
                    .collect();
                *point.last_mut().unwrap() = top;

                let mut prover = transcript::build_prover("lut-layer", &point);
                let terminal = prove(
                    &mut prover,
                    &mut eq_memory,
                    &point,
                    &lut_layers,
                    &mut bound_children,
                );
                let proof = prover.finish();

                let mut verifier = transcript::build_verifier("lut-layer", &point, &proof);
                let claim = mle(parent.clone(), &point);
                assert_eq!(
                    verify_layer(&mut verifier, claim, &point),
                    Some(terminal.clone()),
                    "layer {layer}, top coordinate {top}"
                );
                assert_eq!(terminal.1, mle(children.clone(), &terminal.0));
            }
        }
    }

    /// The bottom three layers looked up in the lookup tables, and every layer
    /// above, match the circuit built one leaf at a time from the same table.
    #[test]
    fn lut_layers_match_the_leaf_construction() {
        // At t = 7 and 8 the transposed row groups are shorter than a `u128`.
        for (log_rows, log_columns) in [(7, 15), (8, 14), (9, 13), (15, 7)] {
            let (shape, packed, row_images) = instance((log_rows, log_columns), Bits::Drawn, 3);
            let table = table(shape, &packed);
            let (expected_output, expected_layers) =
                fold_layers(leaves(&table, &row_images), shape.columns());

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
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 2);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);
        let (output, proof) = prove(&table, &row_images);
        assert!(verify(&leaves, &output, &proof));

        // Cover partial field elements and missing messages in every layer.
        for len in 0..proof.narg_string.len() {
            let mut truncated = proof.clone();
            truncated.narg_string.truncate(len);
            assert!(
                !verify(&leaves, &output, &truncated),
                "accepted {len} bytes"
            );
        }
    }

    /// Proves the circuit of `table` at a point the verifier draws, as
    /// `gkr_reduce` does. Returns the output layer, one product per column.
    fn prove(table: &BitTable, row_images: &[Field]) -> (Vec<Field>, transcript::Proof) {
        let (output, witnesses) = GrandProductCircuit::new(row_images, table).batched_eval();
        let log_bits = table.shape().log_bits();

        // TODO instance is a bit loose and should be replaced by PCS
        let instance = (output.clone(), 1u128 << log_bits);
        let mut prover = transcript::build_prover("gkr", &instance);

        // Mirrors `verify`, since both sides must draw the same number of
        // challenges here.
        let point: Vec<Field> = (0..output.len().ilog2())
            .map(|_| prover.verifier_message())
            .collect();

        gpgkr_prove(&mut prover, log_bits, &point, witnesses);
        (output, prover.finish())
    }

    /// Verifies `proof` for `output`, then checks the final claim against the
    /// MLE of `leaves`.
    fn verify(leaves: &[Field], output: &[Field], proof: &transcript::Proof) -> bool {
        let instance = (output.to_vec(), leaves.len() as u128);
        let mut verifier = transcript::build_verifier("gkr", &instance, proof);

        let point: Vec<Field> = (0..output.len().ilog2())
            .map(|_| verifier.verifier_message())
            .collect();
        let claim = mle(output.to_vec(), &point);
        let layers = leaves.len().ilog2() - output.len().ilog2();

        match gpgkr_verify(&mut verifier, claim, &point, layers) {
            Some((point, claim)) => {
                mle(leaves.to_vec(), &point) == claim && verifier.check_eof().is_ok()
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
