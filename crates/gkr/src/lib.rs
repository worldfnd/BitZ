use common::{BitMatrix, BitTable, SecurityLevel};
use field::{F128, Wide256};
use num_traits::{ConstOne, ConstZero};
use rayon::prelude::*;
use std::ops::Range;
use transcript::{ProverState, VerifierState};

pub type Field = F128;

const CUBIC_GRINDING_LABEL: &[u8] = b"gkr/cubic/v1";

// TODO replace this with a proper approach
/// Matches `poly`'s `PARALLEL_FOLD_THRESHOLD`, chosen for the same per-round
/// MLE-fold shape.
const PARALLEL_MIN_LANES: usize = 1 << 12;

/// Proves the layer-by-layer sumcheck reduction from a claim at `point`
/// (an evaluation point on the output layer) down to a claim on the leaves.
///
/// Points are little-endian, as in `poly`'s tables: `point[i]` is the
/// coordinate of index bit `i`. Every layer adds one coordinate, at the top.
///
/// # Panics
///
/// If `point` does not have a coordinate per tree bit: the output layer has
/// one value per tree.
#[must_use]
#[tracing::instrument(name = "Prove GKR", skip_all)]
pub fn gpgkr_prove(
    ps: &mut ProverState,
    point: &[Field],
    // Every layer below the output: layer 3 and up in full, layers 0 to 2 in
    // their lookup tables.
    witnesses: LayerWitnesses,
    security: SecurityLevel,
) -> (Vec<Field>, Field) {
    let mut point = point.to_owned();
    let LayerWitnesses {
        lut_layers,
        upper_nodes,
    } = witnesses;
    // Every layer below adds a coordinate, so this fixes the length of every
    // point after it.
    assert_eq!(
        point.len(),
        lut_layers.log_trees,
        "the output has one value per tree, so the point has a coordinate per tree bit"
    );
    // The leaves' sumcheck has the largest point, a coordinate per index bit
    // of its parent, layer 1.
    let mut eq_memory = EqMemory::new(lut_layers.nodes::<LAYER1>(), lut_layers.eq_split());
    for children in upper_nodes.into_iter().rev() {
        (point, _) = prove_layer(ps, &mut eq_memory, &point, children, security);
    }

    // Layers 2, 1 and 0 stay in their lookup tables. Round 1 of each writes the
    // children it has bound, `[left′ | right′]`, into one allocation sized
    // for the leaves': round 1 halves them to as many as layer 1 has nodes.
    let mut bound_children = Field::zeroed_vec(lut_layers.nodes::<LAYER1>());
    let (point, _) = prove_lut_layer::<LAYER2>(
        ps,
        &mut eq_memory,
        &point,
        &lut_layers,
        &mut bound_children,
        security,
    );
    let (point, _) = prove_lut_layer::<LAYER1>(
        ps,
        &mut eq_memory,
        &point,
        &lut_layers,
        &mut bound_children,
        security,
    );
    prove_lut_layer::<LEAVES>(
        ps,
        &mut eq_memory,
        &point,
        &lut_layers,
        &mut bound_children,
        security,
    )
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
    eq_memory: &mut EqMemory,
    point: &[Field],
    mut children: Vec<Field>,
    security: SecurityLevel,
) -> (Vec<Field>, Field) {
    let eq_tables = eq_memory.tables(point);
    // The top index bit of `children` picks the half: `L`, then `R`.
    let mid = children.len() / 2;
    let (left, right) = children.split_at_mut(mid);
    prove_rounds(ps, &eq_tables, point, None, left, right, security)
}

/// The dense rounds of a layer, then its last message. A LUT-backed layer
/// has already proved round 1, which binds the top coordinate of `point`, and
/// passes that round's challenge as `round_one`; its `left` and `right` are
/// then the bound halves. A dense layer passes `None` and proves every round
/// here.
fn prove_rounds(
    ps: &mut ProverState,
    eq_tables: &SplitEqTables,
    point: &[Field],
    round_one: Option<Field>,
    mut left: &mut [Field],
    mut right: &mut [Field],
    security: SecurityLevel,
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
        let eq = eq_tables.round(i);
        let (sum_b, sum_inf) = if sends_one(z) {
            round_sums::<true>(left0, left1, right0, right1, eq)
        } else {
            round_sums::<false>(left0, left1, right0, right1, eq)
        };
        ps.prover_message(&[eq_bound * sum_b.reduce(), eq_bound * sum_inf.reduce()]);

        ps.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3));
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
///
/// Above the split, the positions are summed in parts of at most
/// `PARALLEL_MIN_LANES`, each within one block: first under the inner
/// weights, then the part's sums are multiplied by its block's outer weight.
#[tracing::instrument(name = "Accumulate round coefficients", level = "debug", skip_all)]
fn round_sums<const AT_ONE: bool>(
    left0: &[Field],
    left1: &[Field],
    right0: &[Field],
    right1: &[Field],
    eq: EqFactors,
) -> (Wide256, Wide256) {
    let at = |v0: Field, v1: Field| if AT_ONE { v1 } else { v0 };
    debug_assert_eq!(eq.inner.len() * eq.outer.len(), left0.len());

    // The last round has a single summand, and `eq` there is the empty
    // product `1`: no multiplication by it, and too few for two lanes.
    if eq.inner.len() == 1 {
        let (l0, l1, r0, r1) = (left0[0], left1[0], right0[0], right1[0]);
        return (
            Wide256::mul(at(l0, l1), at(r0, r1)),
            Wide256::mul(l1 - l0, r1 - r0),
        );
    }

    // Above the split: a block per outer weight.
    // A part is `min(block_len, PARALLEL_MIN_LANES)` and is summed serially.
    if eq.outer.len() > 1 {
        let block_len = eq.inner.len();
        let part_len = block_len.min(PARALLEL_MIN_LANES);
        return left0
            .par_chunks_exact(part_len)
            .zip(left1.par_chunks_exact(part_len))
            .zip(right0.par_chunks_exact(part_len))
            .zip(right1.par_chunks_exact(part_len))
            .enumerate()
            .with_min_len(PARALLEL_MIN_LANES / part_len)
            .map(|(p, (((l0, l1), r0), r1))| {
                let start = p * part_len;
                let inner = &eq.inner[start % block_len..][..part_len];
                let sums = l0
                    .chunks_exact(2)
                    .zip(l1.chunks_exact(2))
                    .zip(r0.chunks_exact(2))
                    .zip(r1.chunks_exact(2))
                    .zip(inner.chunks_exact(2))
                    .fold(no_sums(), |sums, ((((l0, l1), r0), r1), e)| {
                        add_lanes::<AT_ONE>(sums, l0, l1, r0, r1, e)
                    });
                scale_sums(eq.outer[start / block_len], sums)
            })
            .reduce(no_sums, add_sums);
    }

    // Two lanes per step: independent multiplication chains. Chunking also
    // reduces variance in benchmarking.
    left0
        .par_chunks_exact(2)
        .zip(left1.par_chunks_exact(2))
        .zip(right0.par_chunks_exact(2))
        .zip(right1.par_chunks_exact(2))
        .zip(eq.inner.par_chunks_exact(2))
        // Chunked by 2, so the element-count threshold below is halved in
        // terms of chunks.
        .with_min_len(PARALLEL_MIN_LANES / 2)
        .fold(no_sums, |sums, ((((l0, l1), r0), r1), e)| {
            add_lanes::<AT_ONE>(sums, l0, l1, r0, r1, e)
        })
        .reduce(no_sums, add_sums)
}

/// One step of [`round_sums`]: adds the terms of two positions, the lanes,
/// to the sums.
#[inline(always)]
fn add_lanes<const AT_ONE: bool>(
    (mut sum_b, mut sum_inf): (Wide256, Wide256),
    l0: &[Field],
    l1: &[Field],
    r0: &[Field],
    r1: &[Field],
    e: &[Field],
) -> (Wide256, Wide256) {
    let at = |v0: Field, v1: Field| if AT_ONE { v1 } else { v0 };
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
}

/// A round's two sums before any term: `(0, 0)`.
fn no_sums() -> (Wide256, Wide256) {
    (Wide256::zero(), Wide256::zero())
}

/// The sums of two parts of a round.
fn add_sums(
    (x_b, x_inf): (Wide256, Wide256),
    (y_b, y_inf): (Wide256, Wide256),
) -> (Wide256, Wide256) {
    (x_b + y_b, x_inf + y_inf)
}

/// A block's sums under the inner weights, times its outer weight `e`.
fn scale_sums(e: Field, (sum_b, sum_inf): (Wide256, Wide256)) -> (Wide256, Wide256) {
    (
        Wide256::mul(e, sum_b.reduce()),
        Wide256::mul(e, sum_inf.reduce()),
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
    /// The nibbles of the layer-2 nodes; see [`LutLayers::new`].
    matrix: BitMatrix<'static>,
    luts: Vec<[Field; LUT_ENTRIES]>,
    /// `s`: a lookup table's run of nibbles holds one per tree.
    log_trees: usize,
}

impl LutLayers {
    fn new(matrix: BitMatrix<'_>, images: &[Field]) -> Self {
        // Leaf `(b << s) | c` is `images[b]` where `bit(c, b)` is set and one
        // elsewhere: the low `s` index bits pick one of the `2^s` trees and
        // the rest are the leaf's position in it. Each layer folds the top
        // position bit, so tree `c` multiplies the images `bit(c, ·)` selects.
        //
        // Layers 0 to 2 are looked up in 16-entry lookup tables, one per four
        // images, and the four leaves under a layer-2 node are a quarter of
        // the positions apart, so the top two position bits are split off and
        // moved below the tree bits. After the transpose, layer-2 node
        // position `g` is one contiguous run, a nibble per tree in ascending
        // order, and nibble `c` of it holds `bit(c, g + q·n)` at bit `q`, `n`
        // being a quarter of the images: the four leaves under layer-2 node
        // `(g << s) | c`.
        let log_dim2 = matrix.log_dim2() - 2;
        let matrix = matrix.reshape(log_dim2).transpose();

        let log_trees = (matrix.dim2() / NIBBLE_BITS).ilog2() as usize;
        // `pairs` reads whole words of one lookup table's run.
        assert!(
            1 << log_trees >= NIBBLES_IN_WORD,
            "each lookup table needs at least a word of nibbles"
        );
        Self {
            matrix,
            luts: lower_luts(images),
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

    /// The nodes of layer `LAYER`: one per nibble in layer 2, twice as many
    /// in each layer below it.
    fn nodes<const LAYER: usize>(&self) -> usize {
        const { assert!(LAYER <= LAYER2, "layers 0 to 2 are the LUT-backed ones") };
        (self.words().len() * NIBBLES_IN_WORD) << (LAYER2 - LAYER)
    }

    /// The positions of round 1 handled as one piece: a lookup table's run,
    /// or a part of it, so that a piece never spans two lookup tables.
    fn piece(&self) -> usize {
        (1 << self.log_trees).min(PARALLEL_MIN_LANES)
    }

    /// Eq tables split point: a block of positions is a piece of round 1,
    /// so the inner table has at most `PARALLEL_MIN_LANES` weights and every
    /// piece reads it whole.
    fn eq_split(&self) -> usize {
        // Measured alternatives:
        // * splitting at the tree bits, `log_trees` - the same up to `s = 15`, worse above
        // * a block of `PARALLEL_MIN_LANES` positions whatever `s` - no improvement
        self.piece().ilog2() as usize
    }

    /// Round 1's positions `0..half` in pieces; see [`pieces`].
    fn pieces(&self, half: usize) -> impl IndexedParallelIterator<Item = Range<usize>> + use<> {
        pieces(half, self.piece())
    }

    /// The pairs that round 1 binds at `positions`, which lie in one lookup
    /// table's run and start and end on a word: per position, the nibbles of
    /// its operands `l0, l1, r0, r1`, unmasked, and each operand's lookup
    /// table.
    ///
    /// Round 1 binds the top bit of the parent's index, so each node pairs
    /// with its partner `stride` nodes later, where `stride` is the number of
    /// round-1 positions, half the parent's nodes:
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
        positions: Range<usize>,
    ) -> (
        [&[Field; LUT_ENTRIES]; 4],
        impl Iterator<Item = impl Iterator<Item = [usize; 4]>> + '_,
    ) {
        // Half the parent's nodes, which are half the layer's.
        let stride = self.nodes::<LAYER>() / 4;
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

/// The positions `0..half` in pieces of `piece`, as a parallel iterator
/// taking at least `PARALLEL_MIN_LANES` positions per task, whatever the
/// piece.
fn pieces(half: usize, piece: usize) -> impl IndexedParallelIterator<Item = Range<usize>> {
    (0..half / piece)
        .into_par_iter()
        .with_min_len(PARALLEL_MIN_LANES / piece)
        .map(move |p| p * piece..(p + 1) * piece)
}

/// The operands `[l0, l1, r0, r1]` of one position of round 1 of layer
/// `LAYER`, from their lookup tables and nibbles.
fn operands<const LAYER: usize>(
    luts: &[&[Field; LUT_ENTRIES]; 4],
    nibbles: [usize; 4],
) -> [Field; 4] {
    std::array::from_fn(|o| luts[o][nibbles[o] & MASK[LAYER][o]])
}

/// Proves the layer whose children are layer `LAYER` of the lookup tables,
/// like [`prove_layer`]. Round 1 reads its operands from the lookup tables
/// and writes only the children it has bound, `[left′ | right′]`, into
/// `bound_children`; the rounds after it are dense.
fn prove_lut_layer<const LAYER: usize>(
    ps: &mut ProverState,
    eq_memory: &mut EqMemory,
    point: &[Field],
    lut_layers: &LutLayers,
    bound_children: &mut [Field],
    security: SecurityLevel,
) -> (Vec<Field>, Field) {
    let eq_tables = eq_memory.tables(point);
    let top = point.len() - 1;

    // Nothing is bound before round 1, so its sums go out as they are.
    let eq = eq_tables.round(top);
    let (sum_b, sum_inf) = if sends_one(point[top]) {
        round_one_sums::<LAYER, true>(lut_layers, eq)
    } else {
        round_one_sums::<LAYER, false>(lut_layers, eq)
    };
    ps.prover_message(&[sum_b.reduce(), sum_inf.reduce()]);

    ps.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3));
    let r = ps.verifier_message();
    let half = 1 << top;
    let (left, right) = bound_children[..2 * half].split_at_mut(half);
    lut_bind::<LAYER>(r, lut_layers, left, right);
    prove_rounds(ps, &eq_tables, point, Some(r), left, right, security)
}

/// Round 1's sum at the boolean point `b` (`1` if `AT_ONE`) and its leading
/// coefficient, unreduced, as [`round_sums`] gives them. The leaves and
/// layer 1 sum their eq weights in buckets. Layer 2 pairs whole nibbles of
/// two lookup tables, 256 keys per product, which costs more to finish than
/// a piece saves at small `s`, so it works node by node.
#[tracing::instrument(name = "Accumulate round 1 coefficients", level = "debug", skip_all)]
fn round_one_sums<const LAYER: usize, const AT_ONE: bool>(
    lut_layers: &LutLayers,
    eq: EqFactors,
) -> (Wide256, Wide256) {
    match LAYER {
        LEAVES => piece_sums(lut_layers, eq, leaf_buckets::<AT_ONE>),
        LAYER1 => piece_sums(lut_layers, eq, layer1_buckets::<AT_ONE>),
        LAYER2 => piece_sums(lut_layers, eq, layer2_nodes::<AT_ONE>),
        _ => unreachable!("layers 0 to 2 are the LUT-backed ones"),
    }
}

/// Round 1's sums a piece at a time: `sums` gives those of a piece under its
/// inner eq weights, which stay in cache, and the outer weight of the
/// piece's block joins them once.
fn piece_sums(
    lut_layers: &LutLayers,
    eq: EqFactors,
    sums: impl Fn(&LutLayers, Range<usize>, &[Field]) -> (Wide256, Wide256) + Sync,
) -> (Wide256, Wide256) {
    let block = eq.inner.len();
    // A piece lies in one block, as it lies in one lookup table's run.
    let piece = lut_layers.piece().min(block);
    pieces(block * eq.outer.len(), piece)
        .map(|positions| {
            let e = eq.outer[positions.start / block];
            let inner = &eq.inner[positions.start % block..][..positions.len()];
            scale_sums(e, sums(lut_layers, positions, inner))
        })
        .reduce(no_sums, add_sums)
}

/// The piece of round 1 for the leaves at `positions`, its weights in `eq`.
/// Every operand is one bit of the same nibble, so both sums are functions of
/// the nibble: each position adds its weight into its nibble's bucket, and
/// each of the 16 buckets is multiplied once.
fn leaf_buckets<const AT_ONE: bool>(
    lut_layers: &LutLayers,
    positions: Range<usize>,
    eq: &[Field],
) -> (Wide256, Wide256) {
    let ([lut, ..], pairs) = lut_layers.pairs::<LEAVES>(positions);
    let mut bucket = [Field::ZERO; LUT_ENTRIES];
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
    positions: Range<usize>,
    eq: &[Field],
) -> (Wide256, Wide256) {
    // `l0` and `r0` read nibble `n0` in its lookup table, `l1` and `r1`
    // nibble `n1`.
    let ([lut0, lut1, ..], pairs) = lut_layers.pairs::<LAYER1>(positions);
    let mut bucket = [[Field::ZERO; LUT_ENTRIES]; 4];
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
    let product: [[Field; LUT_ENTRIES]; 4] = [
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
fn dot(bucket: &[Field; LUT_ENTRIES], value: &[Field; LUT_ENTRIES]) -> Wide256 {
    bucket
        .iter()
        .zip(value)
        .fold(Wide256::zero(), |acc, (&b, &v)| acc + Wide256::mul(b, v))
}

/// The piece of round 1 for layer 2 at `positions`, its weights in `eq`: node
/// by node, as [`round_sums`] forms them, with the operands read from the
/// lookup tables.
fn layer2_nodes<const AT_ONE: bool>(
    lut_layers: &LutLayers,
    positions: Range<usize>,
    eq: &[Field],
) -> (Wide256, Wide256) {
    let at = |v0: Field, v1: Field| if AT_ONE { v1 } else { v0 };
    let (luts, pairs) = lut_layers.pairs::<LAYER2>(positions);
    let (mut sum_b, mut sum_inf) = no_sums();
    for (word, eq) in pairs.zip(eq.chunks_exact(NIBBLES_IN_WORD)) {
        for (n, &e) in word.zip(eq) {
            let [l0, l1, r0, r1] = operands::<LAYER2>(&luts, n);
            sum_b += Wide256::mul(e * at(l0, l1), at(r0, r1));
            sum_inf += Wide256::mul(e * (l1 - l0), r1 - r0);
        }
    }
    (sum_b, sum_inf)
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
            let (luts, pairs) = lut_layers.pairs::<LAYER>(positions);
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

/// `eq(z_{<i}, ·)` for every round `i` of a layer, as two factors. Index bit
/// `split` divides a position, `x = (g << split) | k`, and
/// `eq(z_{<i}, x) = eq(z_{[split, i)}, g) · eq(z_{<split}, k)`.
///
/// The split in defined in ePrint 2025/1117, 5.2. "Algorithm 5: Additional Speedup
/// via Splitting Eq-Poly" in "Speeding Up Sum-Check Proving".
///
/// Rounds bind the top coordinate first, so the low factor is the one that stays,
/// and its positions are contiguous: a round above the split sums each block of
/// `2^split` positions under the low weights, then multiplies the block's
/// sums by its high weight. A round at or below the split has one block, and
/// its weights are a low table.
#[derive(Debug)]
struct SplitEqTables<'a> {
    /// `eq(z_{<j}, ·)` for `j in 0..=split`.
    low: EqTables<'a>,
    /// `eq(z_{[split, split + j)}, ·)` for `j in 0..=(top - split)`, where `top`
    /// is the index of the top coordinate.
    high: EqTables<'a>,
    /// Split point between `low` and `high`.
    /// Minimum of requested split, and index of the top coordinate.
    split: usize,
}

impl SplitEqTables<'_> {
    /// The weights of round `i`, which sums over the index bits below `i`.
    fn round(&self, i: usize) -> EqFactors<'_> {
        let above_split = i.saturating_sub(self.split);
        EqFactors {
            inner: self.low.below(i - above_split),
            outer: self.high.below(above_split),
        }
    }
}

/// The eq weights of a round:
/// Position `(g << m) | k` weighs `outer[g] · inner[k]`, where `inner` has `2^m` weights.
/// At or below the split `outer` is `[1]`.
#[derive(Debug)]
struct EqFactors<'a> {
    inner: &'a [Field],
    outer: &'a [Field],
}

/// The lowest split point: round 1 of the LUT-backed layers reads whole words
/// of nibbles, so a block holds at least one.
const MIN_EQ_SPLIT: usize = NIBBLES_IN_WORD.ilog2() as usize;

/// The buffers of [`SplitEqTables`], allocated once for all layers.
#[derive(Debug)]
struct EqMemory {
    low: Vec<Field>,
    high: Vec<Field>,
    split: usize,
    /// The longest point the buffers have room for.
    coordinates: usize,
}

impl EqMemory {
    /// Room for the tables of every sumcheck whose parent layer has up to
    /// `nodes` nodes. Its point has a coordinate per index bit of the parent,
    /// and its tables split at `split`, or at its top coordinate if that is
    /// lower: `2^(split + 1) - 1` low entries, and as many high ones as the
    /// positions above the split count.
    ///
    /// # Panics
    ///
    /// If `split` is below [`MIN_EQ_SPLIT`].
    fn new(nodes: usize, split: usize) -> Self {
        assert!(
            split >= MIN_EQ_SPLIT,
            "a block of the eq split holds at least a word of nibbles"
        );
        let coordinates = nodes.ilog2() as usize;
        let split_bit = split.min(coordinates.saturating_sub(1));
        Self {
            low: Field::zeroed_vec((2 << split_bit) - 1),
            high: Field::zeroed_vec((1 << (coordinates - split_bit)) - 1),
            split,
            coordinates,
        }
    }

    /// Builds the tables of `point`; nothing is allocated here.
    ///
    /// # Panics
    ///
    /// If `point` is empty or longer than the buffers have room for.
    #[tracing::instrument(name = "Build eq tables", level = "debug", skip_all)]
    fn tables(&mut self, point: &[Field]) -> SplitEqTables<'_> {
        assert!(
            point.len() <= self.coordinates,
            "the point has more coordinates than the eq tables have room for"
        );
        let top = point
            .len()
            .checked_sub(1)
            .expect("a sumcheck has a round per coordinate, so at least one");
        let split = self.split.min(top);
        SplitEqTables {
            low: EqTables::new(&mut self.low, &point[..=split]),
            high: EqTables::new(&mut self.high, &point[split..]),
            split,
        }
    }
}

/// `eq(z_{<i}, ·)` for every `i` below the length of a point `z`, in one
/// buffer: a factor of [`SplitEqTables`].
///
/// The former `SuffixTable` with the point little-endian instead of reversed:
/// the summed coordinates are the prefix `z[..i]`, not a suffix. Same entries.
#[derive(Debug)]
struct EqTables<'a> {
    storage: &'a [Field],
}

impl<'a> EqTables<'a> {
    /// `eq(z_{<i}, x)` for `x ∈ {0,1}^i`, in the order of
    /// `poly::eq_table(&z[..i])`. Table `i` sits at
    /// `2^i - 1 .. 2^(i+1) - 1`.
    fn below(&self, i: usize) -> &[Field] {
        let start = (1 << i) - 1;
        let end = (1 << (i + 1)) - 1;
        &self.storage[start..end]
    }

    /// Builds every table of `point` in `storage`, `2^point.len() - 1`
    /// entries; nothing is allocated here.
    // Inlining or not doesn't improve performance, but it does hide it in
    // profiling therefore it has been annotated with inline never.
    #[inline(never)]
    fn new(storage: &'a mut [Field], point: &[Field]) -> EqTables<'a> {
        // Table `i` reads only the coordinates below `i`, so the top one
        // never enters a table.
        let (_, below_top) = point
            .split_last()
            .expect("a sumcheck has a round per coordinate, so at least one");
        // Establishes the whole table's space budget up front so LLVM can
        // prove the `low`/`hi` writes in the loop below are in bounds
        let needed = (1usize << point.len()) - 1;
        assert!(storage.len() >= needed);

        storage[0] = Field::ONE;
        let (mut prev, mut remaining) = storage.split_at_mut(1);

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
}

/// Verifies what [`gpgkr_prove`] proves: `layers` layer sumchecks, from
/// `claim`, the output's MLE at `point`, down to a claim on the leaves.
///
/// Returns the leaves' point, `point.len() + layers` coordinates, and the
/// claim on it; `None` if a check fails or the proof runs short.
#[must_use]
#[tracing::instrument(name = "Verify GKR", skip_all)]
pub fn gpgkr_verify(
    vs: &mut VerifierState,
    mut claim: Field,
    point: &[Field],
    layers: usize,
    security: SecurityLevel,
) -> Option<(Vec<Field>, Field)> {
    let mut point = point.to_owned();
    for _ in 0..layers {
        (point, claim) = verify_layer(vs, claim, &point, security)?
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
    security: SecurityLevel,
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

        vs.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3))
            .ok()?;
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

/// Returns the output layer, the product of each column of the table, and
/// the witnesses of the layers below it.
///
/// Layers 0 to 2 stay as lookup tables, layer 3 is the first one built in
/// full, and the layers above it fold as usual.
///
/// # Panics
///
/// If `row_images` does not have one image per row of `bittable`, or the
/// table has fewer than 16 trees (`s < 4`).
#[tracing::instrument(name = "Evaluate grand-product circuit", level = "debug", skip_all)]
pub fn grand_product(row_images: &[Field], bittable: &BitTable) -> (Vec<Field>, LayerWitnesses) {
    assert_eq!(
        row_images.len(),
        bittable.shape().rows(),
        "one image per row of the table"
    );
    let lut_layers = LutLayers::new(bittable.as_matrix(), row_images);
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

const NIBBLE_BITS: usize = 4;
const NIBBLES_IN_WORD: usize = u64::BITS as usize / NIBBLE_BITS;
/// A lookup table has an entry per nibble value.
const LUT_ENTRIES: usize = 1 << NIBBLE_BITS;

/// Layers 0 to 2 as lookup tables, one per four images: `lut[nibble]` is the
/// layer-2 node whose four leaves are the bits of `nibble`.
///
/// Each lookup table takes one image from each quarter of `images`.
fn lower_luts(images: &[Field]) -> Vec<[Field; LUT_ENTRIES]> {
    /// `lut[nibble]` is the product of the inputs whose bit is set in
    /// `nibble`, `a` on bit 0 through `d` on bit 3 (`0b0000 -> 1`,
    /// `0b0001 -> a`, `0b0011 -> a*b`, ...), so a nibble indexes its
    /// multilinear value directly.
    ///
    /// Built by doubling: entries `2^i..2^(i+1)` are entries `0..2^i`
    /// times the `i`-th input. The first entry of each new half is the
    /// input itself, so it is copied rather than multiplied by one,
    /// leaving 11 multiplications -- one per entry with two or more
    /// factors, which is the minimum.
    fn create_lut(c: [Field; NIBBLE_BITS]) -> [Field; LUT_ENTRIES] {
        let mut lut = [Field::ONE; LUT_ENTRIES];
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
        // A lookup table writes `LUT_ENTRIES` entries.
        .with_min_len(PARALLEL_MIN_LANES / LUT_ENTRIES)
        .map(|(((&a, &b), &c), &d)| create_lut([a, b, c, d]))
        .collect()
}

/// Layer 3, the first layer built in full: node `k` multiplies layer-2 nodes
/// `k` and `k + n/2`, each a lookup-table entry looked up by its nibble.
/// Those are the two nibbles that round 1 of layer 1 reads at position `k`,
/// here read whole.
fn layer3(lut_layers: &LutLayers) -> Vec<Field> {
    let half = lut_layers.nodes::<LAYER2>() / 2;
    let mut layer3 = Field::zeroed_vec(half);
    layer3
        .par_chunks_mut(lut_layers.piece())
        .zip(lut_layers.pieces(half))
        .for_each(|(nodes, positions)| {
            let ([lut0, lut1, ..], pairs) = lut_layers.pairs::<LAYER1>(positions);
            for (nodes, word) in nodes.chunks_exact_mut(NIBBLES_IN_WORD).zip(pairs) {
                for (node, [n0, n1, ..]) in nodes.iter_mut().zip(word) {
                    *node = lut0[n0] * lut1[n1];
                }
            }
        });
    layer3
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

/// The intermediate layers. Layers 0 to 2 stay in their lookup tables,
/// which is how round 1 of their sumchecks reads them; layer 3 and up are
/// kept in full.
pub struct LayerWitnesses {
    /// Layers 0 to 2, in their lookup tables.
    lut_layers: LutLayers,
    /// Layer 3 and up, layer 3 first.
    upper_nodes: Vec<Vec<Field>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::SecurityLevel::{Bits100, Bits128};
    use common::shape::{MIN_LOG_BITS, PACK_BITS};
    use common::{BitZParams, Shape};
    use field::gf128::smallest_generator;
    use proptest::prelude::*;

    fn field() -> impl Strategy<Value = Field> {
        any::<u128>().prop_map(Field::from)
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
            let expected: Field = leaves.iter().product();

            let (top, witnesses) = fold_layers(pad(leaves), 1);

            prop_assert_eq!(top.iter().product::<Field>(), expected);

            for layer in witnesses {
                prop_assert_eq!(layer.iter().product::<Field>(), expected);
            }
        }

        #[test]
        fn fold_layers_top_layer_is_interleaved_group_products(
            leaves in prop::collection::vec(field(), 1..32),
            groups_log2 in 0usize..4,
        ) {
            let padded = pad(leaves);
            let groups = 1usize << groups_log2.min(padded.len().ilog2() as usize);

            let (top, _witnesses) = fold_layers(padded.clone(), groups);
            prop_assert_eq!(top.len(), groups);

            for (j, &value) in top.iter().enumerate() {
                let expected: Field = padded[j..].iter().step_by(groups).product();
                prop_assert_eq!(value, expected);
            }
        }
    }

    /// The smallest shape the gates accept, `(t, s)`: the narrowest `t` at
    /// the smallest total.
    const SMALLEST: (usize, usize) = (PACK_BITS as usize, MIN_LOG_BITS - PACK_BITS as usize);

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
        let packed = (0..1usize << shape.log_packed_len())
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
        BitZParams::<field::FqDefault>::new(shape, smallest_generator())
            .unwrap()
            .table(packed)
            .unwrap()
    }

    /// Leaf `(b << s) | c` is `row_images[b]` where `bit(c, b)` is set and one
    /// elsewhere, as `LutLayers::new` describes, built one leaf at a
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
        // of each tree's leaves.
        #[test]
        fn gpgkr_round_trip(bits in bits(), seed in any::<u64>()) {
            let (shape, packed, row_images) = instance(SMALLEST, bits, seed);
            let table = table(shape, &packed);
            let leaves = leaves(&table, &row_images);

            let (output, proof) = prove(&table, &row_images, Bits100);
            prop_assert!(verify(&leaves, &output, &proof, Bits100));

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
            let (output, proof) = prove(&table, &row_images, Bits100);

            let mut tampered = output;
            tampered[0] += Field::ONE;

            prop_assert!(!verify(&leaves(&table, &row_images), &tampered, &proof, Bits100));
        }
    }

    // At 128 bits every cubic challenge is ground: the proof carries a nonce
    // per sumcheck round, and the verifier rejects a changed one.
    #[test]
    fn gpgkr_security_targets_replay_and_reject_changed_nonces() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 6);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);

        let (_, unground) = prove(&table, &row_images, Bits100);
        let (output, mut proof) = prove(&table, &row_images, Bits128);
        assert!(verify(&leaves, &output, &proof, Bits128));

        // One cubic challenge per coordinate of each layer's point, from the
        // output's `s` up to the leaves' `log_bits - 1`, each with an
        // eight-byte nonce.
        let cubic_challenges: usize = (shape.log_columns()..shape.log_bits()).sum();
        assert_eq!(
            proof.narg_string.len(),
            unground.narg_string.len() + 8 * cubic_challenges
        );
        // The first cubic message occupies two canonical field elements.
        proof.narg_string[32] ^= 1;
        assert!(!verify(&leaves, &output, &proof, Bits128));
    }

    // Zero and one coordinates in the output point take the `sends_one`
    // branch in the top layer's rounds.
    #[test]
    fn gpgkr_round_trip_with_zero_coordinates() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 1);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);
        let s = shape.log_columns();
        let layers = shape.log_rows();

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
            let (output, witnesses) = grand_product(&row_images, &table);
            let claim = mle(&output, &point);
            let mut prover = transcript::build_prover("gkr-zero", &point);
            let terminal = gpgkr_prove(&mut prover, &point, witnesses, Bits100);
            assert_eq!(terminal.1, mle(&leaves, &terminal.0));
            let proof = prover.finish();

            let mut verifier = transcript::build_verifier("gkr-zero", &point, &proof);
            assert_eq!(
                gpgkr_verify(&mut verifier, claim, &point, layers, Bits100),
                Some(terminal)
            );
            verifier.check_eof().unwrap();

            let mut verifier = transcript::build_verifier("gkr-zero", &point, &proof);
            assert!(
                gpgkr_verify(&mut verifier, claim + Field::ONE, &point, layers, Bits100).is_none()
            );
        }
    }

    /// Proves the smallest shape from a point of `len(s)` coordinates, `s`
    /// being the tree bits. The bits are left clear: the point's length is
    /// checked before any of them is read.
    fn prove_from_point(len: fn(usize) -> usize) {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Clear, 1);
        let table = table(shape, &packed);
        let (_, witnesses) = grand_product(&row_images, &table);
        let point = vec![Field::from(5u128); len(shape.log_columns())];
        let mut prover = transcript::build_prover("gkr-point", &point);
        let _ = gpgkr_prove(&mut prover, &point, witnesses, Bits100);
    }

    #[test]
    #[should_panic(expected = "a coordinate per tree bit")]
    fn gpgkr_prove_rejects_a_short_point() {
        prove_from_point(|s| s - 1);
    }

    #[test]
    #[should_panic(expected = "a coordinate per tree bit")]
    fn gpgkr_prove_rejects_a_long_point() {
        prove_from_point(|s| s + 1);
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
        let (_, witnesses) = grand_product(&row_images, &table);
        let lut_layers = witnesses.lut_layers;
        let mut bound_children = Field::zeroed_vec(lut_layers.nodes::<LAYER1>());
        let mut eq_memory = EqMemory::new(lut_layers.nodes::<LAYER1>(), lut_layers.eq_split());

        let mut draws = stream(5);
        for layer in [LEAVES, LAYER1, LAYER2] {
            let prove = lut_prover(layer);
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
                    Bits100,
                );
                let proof = prover.finish();

                let mut verifier = transcript::build_verifier("lut-layer", &point, &proof);
                let claim = mle(parent, &point);
                assert_eq!(
                    verify_layer(&mut verifier, claim, &point, Bits100),
                    Some(terminal.clone()),
                    "layer {layer}, top coordinate {top}"
                );
                assert_eq!(terminal.1, mle(children, &terminal.0));
            }
        }
    }

    type ProveLut = fn(
        &mut ProverState,
        &mut EqMemory,
        &[Field],
        &LutLayers,
        &mut [Field],
        SecurityLevel,
    ) -> (Vec<Field>, Field);

    /// [`prove_lut_layer`] of the LUT-backed layer `layer`.
    fn lut_prover(layer: usize) -> ProveLut {
        match layer {
            LEAVES => prove_lut_layer::<LEAVES>,
            LAYER1 => prove_lut_layer::<LAYER1>,
            _ => prove_lut_layer::<LAYER2>,
        }
    }

    /// A point of `len` coordinates that cycles through zero, one and a drawn
    /// element, `shift` into the cycle: whatever the split point, every kind
    /// lies on both sides of it.
    fn cycling_point(
        len: usize,
        shift: usize,
        draws: &mut impl Iterator<Item = u64>,
    ) -> Vec<Field> {
        (0..len)
            .map(|i| match (i + shift) % 3 {
                0 => Field::ZERO,
                1 => Field::ONE,
                _ => element(draws),
            })
            .collect()
    }

    /// The dense reference of [`prove_layer`]: a term at a time, every weight
    /// read from `poly::eq_table`.
    fn prove_layer_dense(
        ps: &mut ProverState,
        point: &[Field],
        children: &[Field],
        security: SecurityLevel,
    ) -> (Vec<Field>, Field) {
        let n = point.len();
        let (left, right) = children.split_at(children.len() / 2);
        let (mut left, mut right) = (left.to_vec(), right.to_vec());
        let mut next_point = vec![Field::ZERO; n + 1];
        let mut eq_bound = Field::ONE;

        for (i, &z) in point.iter().enumerate().rev() {
            let half = 1 << i;
            let (mut sum_b, mut sum_inf) = (Field::ZERO, Field::ZERO);
            for (x, e) in poly::eq_table(&point[..i]).into_iter().enumerate() {
                let (l0, l1, r0, r1) = (left[x], left[x + half], right[x], right[x + half]);
                sum_b += e * if sends_one(z) { l1 * r1 } else { l0 * r0 };
                sum_inf += e * (l1 - l0) * (r1 - r0);
            }
            ps.prover_message(&[eq_bound * sum_b, eq_bound * sum_inf]);

            ps.grind(CUBIC_GRINDING_LABEL, security.grinding_bits(3));
            let r = ps.verifier_message();
            for x in 0..half {
                left[x] = interpolate(left[x], left[x + half], r);
                right[x] = interpolate(right[x], right[x + half], r);
            }
            next_point[i] = r;
            eq_bound *= eq_factor(r, z);
        }

        ps.prover_message(&[left[0], right[0]]);
        let t = ps.verifier_message();
        next_point[n] = t;
        (next_point, interpolate(left[0], right[0], t))
    }

    /// Every round's factors multiply to the dense table, at every split
    /// point: below the top coordinate, at it, and past it, where the tables
    /// split at the top instead.
    #[test]
    fn split_eq_factors_multiply_to_the_dense_table() {
        const COORDINATES: usize = 9;
        let mut draws = stream(7);
        for shift in 0..3 {
            let point = cycling_point(COORDINATES, shift, &mut draws);
            for split in MIN_EQ_SPLIT..=COORDINATES {
                // One buffer for points of every length, as the layers of a
                // proof share one; the longest first, so that every shorter
                // one finds the buffers written.
                let mut eq_memory = EqMemory::new(1 << COORDINATES, split);
                for len in (1..=COORDINATES).rev() {
                    let tables = eq_memory.tables(&point[..len]);
                    for i in 0..len {
                        let eq_factors = tables.round(i);
                        let outer_len = eq_factors.outer.len();
                        let inner_len = eq_factors.inner.len();
                        // A block of `2^split` positions per outer weight.
                        assert_eq!(outer_len, 1 << i.saturating_sub(split));
                        let dense = poly::eq_table(&point[..i]);
                        assert_eq!(inner_len * outer_len, dense.len());
                        for (x, &weight) in dense.iter().enumerate() {
                            assert_eq!(
                                eq_factors.outer[x / inner_len] * eq_factors.inner[x % inner_len],
                                weight,
                                "split {split}, {len} coordinates, round {i}, position {x}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Checks that a dense layer of `n` coordinates sends the dense
    /// reference's messages at each of `splits`, from points with zero and
    /// one coordinates on both sides of them.
    fn dense_layer_proves_the_reference(
        n: usize,
        splits: std::ops::RangeInclusive<usize>,
        draws: &mut impl Iterator<Item = u64>,
    ) {
        let children: Vec<Field> = (0..2 << n).map(|_| element(draws)).collect();
        for shift in 0..3 {
            let point = cycling_point(n, shift, draws);
            let mut reference = transcript::build_prover("dense-layer", &point);
            let terminal = prove_layer_dense(&mut reference, &point, &children, Bits100);
            let expected = (terminal, reference.finish());

            for split in splits.clone() {
                let mut eq_memory = EqMemory::new(1 << n, split);
                let mut prover = transcript::build_prover("dense-layer", &point);
                let terminal = prove_layer(
                    &mut prover,
                    &mut eq_memory,
                    &point,
                    children.clone(),
                    Bits100,
                );
                assert!(
                    (terminal, prover.finish()) == expected,
                    "{n} coordinates, shift {shift}, split {split}"
                );
            }
        }
    }

    /// Dense layers of up to `MIN_EQ_SPLIT + 5` coordinates, at every split
    /// point: a block is never longer than a part.
    #[test]
    fn dense_layers_prove_the_reference_at_every_split() {
        let mut draws = stream(8);
        // `n` coordinates: every round is at or below the lowest split point
        // up to `MIN_EQ_SPLIT + 1` of them, and rounds above it follow.
        for n in 1..=MIN_EQ_SPLIT + 5 {
            let splits = MIN_EQ_SPLIT..=n.max(MIN_EQ_SPLIT);
            dense_layer_proves_the_reference(n, splits, &mut draws);
        }
    }

    /// A block longer than `PARALLEL_MIN_LANES` is summed in parts of that
    /// many positions: blocks of one, two and four parts, each with rounds
    /// above its split point.
    #[test]
    fn dense_layers_prove_the_reference_with_blocks_in_parts() {
        let log_part = PARALLEL_MIN_LANES.ilog2() as usize;
        let splits = log_part..=log_part + 2;
        dense_layer_proves_the_reference(log_part + 4, splits, &mut stream(15));
    }

    /// Each LUT-backed layer sends the dense reference's messages, from
    /// points whose top coordinate, round 1's, is zero, one or neither, with
    /// the split point at:
    /// - `MIN_EQ_SPLIT`: a block is a word of nibbles, shorter than a piece;
    /// - `eq_split()`, the prover's: a block is a piece;
    /// - `s`: a block is a lookup table's run, two pieces at this shape;
    /// - the coordinate below the top: round 1 has two blocks;
    /// - the top coordinate: one block in every round, the dense tables.
    #[test]
    fn lut_layers_prove_the_reference_at_every_split() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 9);
        let table = table(shape, &packed);
        // The leaves and every layer above, built one leaf at a time.
        let (_, layers) = fold_layers(leaves(&table, &row_images), shape.columns());
        let (_, witnesses) = grand_product(&row_images, &table);
        let lut_layers = witnesses.lut_layers;
        let mut bound_children = Field::zeroed_vec(lut_layers.nodes::<LAYER1>());

        let mut draws = stream(10);
        for layer in [LEAVES, LAYER1, LAYER2] {
            let prove = lut_prover(layer);
            let (children, parent) = (&layers[layer], &layers[layer + 1]);
            let n = parent.len().ilog2() as usize;
            for shift in 0..3 {
                let point = cycling_point(n, shift, &mut draws);
                let mut reference = transcript::build_prover("lut-layer-split", &point);
                let terminal = prove_layer_dense(&mut reference, &point, children, Bits100);
                let expected = (terminal, reference.finish());

                for split in [
                    MIN_EQ_SPLIT,
                    lut_layers.eq_split(),
                    lut_layers.log_trees,
                    n - 2,
                    n - 1,
                ] {
                    let mut eq_memory = EqMemory::new(parent.len(), split);
                    let mut prover = transcript::build_prover("lut-layer-split", &point);
                    let terminal = prove(
                        &mut prover,
                        &mut eq_memory,
                        &point,
                        &lut_layers,
                        &mut bound_children,
                        Bits100,
                    );
                    assert!(
                        (terminal, prover.finish()) == expected,
                        "layer {layer}, shift {shift}, split {split}"
                    );
                }
            }
        }
    }

    /// The whole proof is the dense reference's, byte for byte: every layer
    /// built one leaf at a time and proved by `prove_layer_dense`.
    #[test]
    fn gpgkr_proves_the_dense_reference() {
        // A piece of round 1 is part of a lookup table's run at `s = 13`, and
        // a whole run, shorter than `PARALLEL_MIN_LANES`, at `s = 7`.
        for (shape, shift, security) in [
            (SMALLEST, 0, Bits100),
            (SMALLEST, 1, Bits128),
            ((13, 7), 2, Bits100),
        ] {
            let (shape, packed, row_images) = instance(shape, Bits::Drawn, 11);
            let table = table(shape, &packed);
            let (_, layers) = fold_layers(leaves(&table, &row_images), shape.columns());
            let s = shape.log_columns();

            let point = cycling_point(s, shift, &mut stream(12));
            let mut reference = transcript::build_prover("gkr-dense", &point);
            let mut expected = (point.clone(), Field::ZERO);
            for children in layers.iter().rev() {
                expected = prove_layer_dense(&mut reference, &expected.0, children, security);
            }

            let (_, witnesses) = grand_product(&row_images, &table);
            let mut prover = transcript::build_prover("gkr-dense", &point);
            let terminal = gpgkr_prove(&mut prover, &point, witnesses, security);
            assert_eq!(terminal, expected, "s = {s}, shift {shift}");
            assert!(
                prover.finish() == reference.finish(),
                "s = {s}, shift {shift}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "more coordinates than the eq tables have room for")]
    fn eq_memory_rejects_a_point_longer_than_its_tables() {
        let mut eq_memory = EqMemory::new(1 << 6, MIN_EQ_SPLIT);
        let _ = eq_memory.tables(&[Field::ONE; 7]);
    }

    #[test]
    #[should_panic(expected = "a round per coordinate, so at least one")]
    fn eq_memory_rejects_an_empty_point() {
        let mut eq_memory = EqMemory::new(1 << 6, MIN_EQ_SPLIT);
        let _ = eq_memory.tables(&[]);
    }

    #[test]
    #[should_panic(expected = "at least a word of nibbles")]
    fn eq_memory_rejects_a_split_below_a_word_of_nibbles() {
        let _ = EqMemory::new(1 << 6, MIN_EQ_SPLIT - 1);
    }

    /// The output and every layer from layer 3 up match the circuit built one
    /// leaf at a time from the same table. Layers 0 to 2 stay in their lookup
    /// tables; `lut_layers_prove_from_any_top_coordinate` checks them through
    /// the sumchecks that read them.
    #[test]
    fn lut_layers_match_the_leaf_construction() {
        // At t = 7 and 8 there are 32 and 64 lookup tables, fewer than the bits
        // of a `u128`, so the transpose in `LutLayers::new` moves
        // `u32` and `u64` blocks; from t = 9 on it moves `u128` ones. At s = 7
        // a lookup table's run is 128 nibbles, 8 words, below
        // `PARALLEL_MIN_LANES`, so a piece of `layer3` is a whole run rather
        // than part of one.
        for (log_rows, log_columns) in [(7, 15), (8, 14), (9, 13), (15, 7)] {
            let (shape, packed, row_images) = instance((log_rows, log_columns), Bits::Drawn, 3);
            let table = table(shape, &packed);
            let (expected_output, expected_layers) =
                fold_layers(leaves(&table, &row_images), shape.columns());

            let (output, witnesses) = grand_product(&row_images, &table);
            // Both start at layer 3: the reference after its first three.
            let (layers, expected_layers) = (&witnesses.upper_nodes, &expected_layers[3..]);

            assert!(output == expected_output, "output, t = {log_rows}");
            assert_eq!(layers.len(), expected_layers.len(), "t = {log_rows}");
            for (i, (layer, expected)) in layers.iter().zip(expected_layers).enumerate() {
                assert!(layer == expected, "layer {}, t = {log_rows}", i + 3);
            }
        }
    }

    #[test]
    fn gpgkr_verify_rejects_every_truncated_prefix() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 2);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);
        let (output, proof) = prove(&table, &row_images, Bits100);
        assert!(verify(&leaves, &output, &proof, Bits100));

        // Cover partial field elements and missing messages in every layer.
        for len in 0..proof.narg_string.len() {
            let mut truncated = proof.clone();
            truncated.narg_string.truncate(len);
            assert!(
                !verify(&leaves, &output, &truncated, Bits100),
                "accepted {len} bytes"
            );
        }
    }

    #[test]
    fn gpgkr_verify_rejects_every_changed_message() {
        let (shape, packed, row_images) = instance(SMALLEST, Bits::Drawn, 13);
        let table = table(shape, &packed);
        let leaves = leaves(&table, &row_images);
        let (output, proof) = prove(&table, &row_images, Bits100);
        assert!(verify(&leaves, &output, &proof, Bits100));

        // Without grinding the proof is field elements only: two per round,
        // and two that close each layer.
        const ELEMENT_BYTES: usize = size_of::<Field>();
        assert_eq!(proof.narg_string.len() % ELEMENT_BYTES, 0);
        for element in 0..proof.narg_string.len() / ELEMENT_BYTES {
            let mut changed = proof.clone();
            changed.narg_string[element * ELEMENT_BYTES] ^= 1;
            assert!(
                !verify(&leaves, &output, &changed, Bits100),
                "accepted a change to element {element}"
            );
        }
    }

    /// Proves the circuit of `table` at a point the verifier draws, as
    /// `gkr_reduce` does. Returns the output layer, one product per tree.
    fn prove(
        table: &BitTable,
        row_images: &[Field],
        security: SecurityLevel,
    ) -> (Vec<Field>, transcript::Proof) {
        let (output, witnesses) = grand_product(row_images, table);

        // TODO instance is a bit loose and should be replaced by PCS
        let instance = (output.clone(), 1u128 << table.shape().log_bits());
        let mut prover = transcript::build_prover("gkr", &instance);

        // Mirrors `verify`, since both sides must draw the same number of
        // challenges here.
        let point: Vec<Field> = (0..output.len().ilog2())
            .map(|_| prover.verifier_message())
            .collect();

        // `verify` checks the terminal claim against the leaves itself.
        let _terminal = gpgkr_prove(&mut prover, &point, witnesses, security);
        (output, prover.finish())
    }

    /// Verifies `proof` for `output`, then checks the final claim against the
    /// MLE of `leaves`.
    fn verify(
        leaves: &[Field],
        output: &[Field],
        proof: &transcript::Proof,
        security: SecurityLevel,
    ) -> bool {
        let instance = (output.to_vec(), leaves.len() as u128);
        let mut verifier = transcript::build_verifier("gkr", &instance, proof);

        let point: Vec<Field> = (0..output.len().ilog2())
            .map(|_| verifier.verifier_message())
            .collect();
        let claim = mle(output, &point);
        let layers = (leaves.len().ilog2() - output.len().ilog2()) as usize;

        match gpgkr_verify(&mut verifier, claim, &point, layers, security) {
            Some((point, claim)) => mle(leaves, &point) == claim && verifier.check_eof().is_ok(),
            None => false,
        }
    }

    /// The multilinear extension of `evaluations` at `point`.
    fn mle(evaluations: &[Field], point: &[Field]) -> Field {
        poly::DenseMultilinearExtension::evaluate_exact(evaluations, point)
    }
}
