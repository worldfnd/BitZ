//! Modulus-independent reverse-mode tape for `r * (A + x B + x^2 C)`.
//!
//! The circuit is replayed once without witness values or a field modulus.
//! Z-side linear arithmetic is recorded as a Wengert graph, then dead nodes are
//! removed and the remaining graph is transposed and ordered by reverse depth.
//! Applying the finished tape is reverse-mode automatic differentiation of
//! `r * (A + x B + x^2 C) * w`. Nodes at one depth write disjoint adjoints, so
//! sufficiently wide depths are evaluated in parallel without atomics.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::Sum;
use std::mem::{size_of, size_of_val};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use std::sync::{Arc, Mutex};

use crypto_bigint::modular::{FixedMontyForm, FixedMontyParams};
use crypto_bigint::{Odd, U128};
use num_traits::{One, Zero};
use rayon::prelude::*;

use crate::matrix_products::{RuntimeModulus, StoredInteger};
use crate::witgen::Z;
use crate::{BoolWitness, Circuit, HintResult, PackedBits, ScalarBits, WitnessContext};

const PARALLEL_LEVEL_THRESHOLD: usize = 1 << 16;
const PARALLEL_VECTOR_THRESHOLD: usize = 1 << 14;
const OUTPUT_CHUNK_SIZE: usize = 1 << 12;
const NO_NODE: u32 = u32::MAX;

/// A value-free Boolean handle used while recording the arithmetic tape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WengertBit;

impl From<bool> for WengertBit {
    fn from(_: bool) -> Self {
        Self
    }
}

impl BoolWitness for WengertBit {
    type Repr<const N: usize, const M: usize> = ScalarBits<Self, N>;
}

#[derive(Debug)]
struct RawTerm {
    node: u32,
    coefficient: StoredInteger,
}

impl RawTerm {
    fn new<const LIMBS: usize>(node: u32, coefficient: Z<LIMBS>) -> Self {
        Self {
            node,
            coefficient: StoredInteger::from_fixed(coefficient),
        }
    }
}

#[derive(Debug)]
enum RawNode {
    Input,
    Sum(Box<[RawTerm]>),
}

#[derive(Clone, Copy, Debug)]
enum RootKind {
    A,
    B,
    C,
}

#[derive(Debug)]
struct RawRoot {
    row: u32,
    kind: RootKind,
    term: RawTerm,
}

#[derive(Debug)]
struct Recorder {
    nodes: Vec<RawNode>,
    input_nodes: Vec<u32>,
    roots: Vec<RawRoot>,
    power_groups: Vec<RawPowerGroup>,
    constraints: usize,
}

#[derive(Debug)]
struct RawPowerGroup {
    first_column: u32,
    len: u32,
    low_len: u32,
    full_node: u32,
    low_node: u32,
}

impl Recorder {
    fn new() -> Self {
        Self {
            // Integer-witness column zero is the implicit constant one.
            nodes: vec![RawNode::Input],
            input_nodes: vec![0],
            roots: Vec::new(),
            power_groups: Vec::new(),
            constraints: 0,
        }
    }

    fn push_input(&mut self) -> u32 {
        let node = u32::try_from(self.nodes.len()).expect("too many Wengert nodes");
        self.nodes.push(RawNode::Input);
        self.input_nodes.push(node);
        node
    }

    fn push_sum(&mut self, terms: Box<[RawTerm]>) -> u32 {
        debug_assert!(terms.len() >= 2);
        let node = u32::try_from(self.nodes.len()).expect("too many Wengert nodes");
        debug_assert!(terms.iter().all(|term| term.node < node));
        self.nodes.push(RawNode::Sum(terms));
        node
    }
}

#[derive(Debug)]
struct SharedRecorder(Mutex<Option<Recorder>>);

impl SharedRecorder {
    fn with_mut<R>(&self, apply: impl FnOnce(&mut Recorder) -> R) -> R {
        let mut guard = self.0.lock().expect("Wengert recorder lock poisoned");
        apply(
            guard
                .as_mut()
                .expect("the Wengert generator has already been finished"),
        )
    }
}

/// A scaled Wengert handle with the gadget-local coefficient width.
#[derive(Clone, Debug)]
pub struct WengertValue<const LIMBS: usize> {
    location: ValueLocation,
    coefficient: Z<LIMBS>,
}

#[derive(Clone, Debug)]
enum ValueLocation {
    /// A coefficient times the implicit constant-one input.
    Constant,
    Node {
        recorder: Arc<SharedRecorder>,
        node: u32,
    },
}

impl<const LIMBS: usize> WengertValue<LIMBS> {
    fn attached(recorder: Arc<SharedRecorder>, node: u32, coefficient: Z<LIMBS>) -> Self {
        Self {
            location: ValueLocation::Node { recorder, node },
            coefficient,
        }
    }

    fn recorder(&self) -> Option<&Arc<SharedRecorder>> {
        match &self.location {
            ValueLocation::Constant => None,
            ValueLocation::Node { recorder, .. } => Some(recorder),
        }
    }

    fn raw_term(&self) -> RawTerm {
        let node = match self.location {
            ValueLocation::Constant => 0,
            ValueLocation::Node { node, .. } => node,
        };
        RawTerm::new(node, self.coefficient)
    }

    fn same_recorder(left: &Arc<SharedRecorder>, right: &Arc<SharedRecorder>) {
        assert!(
            Arc::ptr_eq(left, right),
            "cannot combine values from different Wengert generators"
        );
    }
}

impl<const LIMBS: usize> From<Z<LIMBS>> for WengertValue<LIMBS> {
    fn from(coefficient: Z<LIMBS>) -> Self {
        Self {
            location: ValueLocation::Constant,
            coefficient,
        }
    }
}

impl<const LIMBS: usize> Zero for WengertValue<LIMBS> {
    fn zero() -> Self {
        Self::from(Z::zero())
    }

    fn is_zero(&self) -> bool {
        self.coefficient.is_zero()
    }
}

impl<const LIMBS: usize> Add for WengertValue<LIMBS> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }

        match (&self.location, &rhs.location) {
            (ValueLocation::Constant, ValueLocation::Constant) => {
                Self::from(self.coefficient + rhs.coefficient)
            }
            (
                ValueLocation::Node {
                    recorder: left,
                    node: left_node,
                },
                ValueLocation::Node {
                    recorder: right,
                    node: right_node,
                },
            ) if left_node == right_node => {
                Self::same_recorder(left, right);
                Self::attached(left.clone(), *left_node, self.coefficient + rhs.coefficient)
            }
            _ => {
                let recorder = self
                    .recorder()
                    .or_else(|| rhs.recorder())
                    .expect("nonconstant Wengert sum needs a recorder")
                    .clone();
                if let Some(other) = self.recorder() {
                    Self::same_recorder(&recorder, other);
                }
                if let Some(other) = rhs.recorder() {
                    Self::same_recorder(&recorder, other);
                }
                let terms = [self.raw_term(), rhs.raw_term()].into();
                let node = recorder.with_mut(|tape| tape.push_sum(terms));
                Self::attached(recorder, node, Z::one())
            }
        }
    }
}

impl<const LIMBS: usize> AddAssign for WengertValue<LIMBS> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<const LIMBS: usize> Neg for WengertValue<LIMBS> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.coefficient = -self.coefficient;
        self
    }
}

impl<const LIMBS: usize> Sub for WengertValue<LIMBS> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<const LIMBS: usize> SubAssign for WengertValue<LIMBS> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<const LIMBS: usize> Mul<Z<LIMBS>> for WengertValue<LIMBS> {
    type Output = Self;

    fn mul(mut self, rhs: Z<LIMBS>) -> Self::Output {
        self.coefficient = self.coefficient * rhs;
        self
    }
}

impl<const LIMBS: usize> Sum for WengertValue<LIMBS> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    dependent: u32,
    coefficient: u32,
}

#[derive(Clone, Copy, Debug)]
struct Root {
    row: u32,
    coefficient: u32,
    kind: RootKind,
}

#[derive(Clone, Copy, Debug)]
struct PowerGroup {
    first_column: u32,
    len: u32,
    low_len: u32,
    full_node: u32,
    low_node: u32,
}

/// A compact, modulus-independent reverse-mode program.
#[derive(Debug)]
pub struct WengertTape {
    constraints: usize,
    input_nodes: Box<[u32]>,
    level_offsets: Box<[u32]>,
    edge_offsets: Box<[u32]>,
    edges: Box<[Edge]>,
    root_offsets: Box<[u32]>,
    roots: Box<[Root]>,
    power_groups: Box<[PowerGroup]>,
    coefficients: Box<[StoredInteger]>,
}

/// A modulus-prepared evaluator with reusable reverse-pass storage.
///
/// Construct this after the random prime is known with [`WengertTape::prepare`].
/// Repeated calls reuse all large allocations and the modulus-dependent
/// coefficient conversion.
#[derive(Debug)]
pub struct PreparedWengertEvaluator<'a> {
    tape: &'a WengertTape,
    modulus_uint: U128,
    modulus_words: [u64; 2],
    mod_neg_inv: u64,
    params: FixedMontyParams<2>,
    coefficients: Vec<[u64; 2]>,
    weighted_challenges: Vec<[[u64; 2]; 3]>,
    adjoints: Vec<[u64; 2]>,
    output: Vec<[u64; 2]>,
    powers_of_two: Vec<[u64; 2]>,
}

impl WengertTape {
    fn from_recorder(recorder: Recorder) -> Self {
        let Recorder {
            nodes,
            input_nodes,
            roots: raw_roots,
            power_groups: raw_power_groups,
            constraints,
        } = recorder;
        let node_count = nodes.len();

        // Reverse reachability removes arithmetic that cannot affect A/B/C.
        let mut live = vec![false; node_count];
        for root in &raw_roots {
            if !root.term.coefficient.is_zero() {
                live[root.term.node as usize] = true;
            }
        }
        for node in (0..node_count).rev() {
            if !live[node] {
                continue;
            }
            if let RawNode::Sum(terms) = &nodes[node] {
                for term in terms {
                    if !term.coefficient.is_zero() {
                        live[term.node as usize] = true;
                    }
                }
            }
        }

        // Packed lifts are differentiated as power groups, so their private
        // scalar inputs and per-bit edges are redundant in the reverse graph.
        for group in &raw_power_groups {
            let start = group.first_column as usize;
            let end = start + group.len as usize;
            for &input in &input_nodes[start..end] {
                live[input as usize] = false;
            }
        }

        // A source is one level after all dependents in the reverse pass.
        let mut depth = vec![0_u32; node_count];
        for target in (0..node_count).rev() {
            if !live[target] {
                continue;
            }
            if let RawNode::Sum(terms) = &nodes[target] {
                for term in terms {
                    if live[term.node as usize] && !term.coefficient.is_zero() {
                        depth[term.node as usize] = depth[term.node as usize].max(
                            depth[target]
                                .checked_add(1)
                                .expect("Wengert depth overflow"),
                        );
                    }
                }
            }
        }

        let live_count = live.iter().filter(|value| **value).count();
        let live_internal_count = nodes
            .iter()
            .zip(&live)
            .filter(|(node, live)| **live && matches!(node, RawNode::Sum(_)))
            .count();
        let level_count = depth
            .iter()
            .zip(nodes.iter().zip(&live))
            .filter_map(|(depth, (node, live))| {
                (*live && matches!(node, RawNode::Sum(_))).then_some(*depth as usize + 1)
            })
            .max()
            .unwrap_or(0);
        let mut level_offsets = vec![0_u32; level_count + 1];
        for ((&node_depth, node), &is_live) in depth.iter().zip(&nodes).zip(&live) {
            if is_live && matches!(node, RawNode::Sum(_)) {
                level_offsets[node_depth as usize + 1] += 1;
            }
        }
        for level in 0..level_count {
            level_offsets[level + 1] += level_offsets[level];
        }
        debug_assert_eq!(
            level_offsets.last().copied().unwrap_or(0) as usize,
            live_internal_count
        );

        let mut ordering = vec![0_u32; live_count];
        let mut level_cursors = level_offsets[..level_count].to_vec();
        for (old, ((&node_depth, node), &is_live)) in
            depth.iter().zip(&nodes).zip(&live).enumerate()
        {
            if is_live && matches!(node, RawNode::Sum(_)) {
                let cursor = &mut level_cursors[node_depth as usize];
                ordering[*cursor as usize] = old as u32;
                *cursor += 1;
            }
        }
        let mut input_cursor = live_internal_count;
        for &old in &input_nodes {
            if live[old as usize] {
                ordering[input_cursor] = old;
                input_cursor += 1;
            }
        }
        debug_assert_eq!(input_cursor, live_count);
        let mut old_to_new = vec![NO_NODE; node_count];
        for (new, &old) in ordering.iter().enumerate() {
            old_to_new[old as usize] = new as u32;
        }

        let mut coefficient_map = HashMap::new();
        let one = StoredInteger::from_fixed(Z::<1>::one());
        let negative_one = StoredInteger::from_fixed(-Z::<1>::one());
        coefficient_map.insert(one.clone(), 0_u32);
        coefficient_map.insert(negative_one.clone(), 1_u32);
        let mut coefficients = vec![one, negative_one];
        let mut intern = |coefficient: &StoredInteger| {
            if let Some(index) = coefficient_map.get(coefficient) {
                return *index;
            }
            let index = u32::try_from(coefficients.len()).expect("too many tape coefficients");
            coefficients.push(coefficient.clone());
            coefficient_map.insert(coefficient.clone(), index);
            index
        };

        let mut edge_counts = vec![0_u32; live_count];
        for (target, node) in nodes.iter().enumerate() {
            if !live[target] {
                continue;
            }
            if let RawNode::Sum(terms) = node {
                for term in terms {
                    if live[term.node as usize] && !term.coefficient.is_zero() {
                        edge_counts[old_to_new[term.node as usize] as usize] += 1;
                    }
                }
            }
        }
        let edge_offsets = prefix_offsets(&edge_counts, "too many Wengert edges");
        let mut edge_cursors = edge_offsets[..live_count].to_vec();
        let mut edges = vec![
            Edge {
                dependent: 0,
                coefficient: 0,
            };
            edge_offsets[live_count] as usize
        ];
        for (target, node) in nodes.iter().enumerate() {
            if !live[target] {
                continue;
            }
            if let RawNode::Sum(terms) = node {
                for term in terms {
                    if !live[term.node as usize] || term.coefficient.is_zero() {
                        continue;
                    }
                    let source = old_to_new[term.node as usize] as usize;
                    let cursor = &mut edge_cursors[source];
                    edges[*cursor as usize] = Edge {
                        dependent: old_to_new[target],
                        coefficient: intern(&term.coefficient),
                    };
                    *cursor += 1;
                }
            }
        }

        let mut root_counts = vec![0_u32; live_count];
        for root in &raw_roots {
            if !root.term.coefficient.is_zero() {
                root_counts[old_to_new[root.term.node as usize] as usize] += 1;
            }
        }
        let root_offsets = prefix_offsets(&root_counts, "too many Wengert roots");
        let mut root_cursors = root_offsets[..live_count].to_vec();
        let mut roots = vec![
            Root {
                row: 0,
                coefficient: 0,
                kind: RootKind::A,
            };
            root_offsets[live_count] as usize
        ];
        for root in raw_roots {
            if root.term.coefficient.is_zero() {
                continue;
            }
            let node = old_to_new[root.term.node as usize] as usize;
            let cursor = &mut root_cursors[node];
            roots[*cursor as usize] = Root {
                row: root.row,
                coefficient: intern(&root.term.coefficient),
                kind: root.kind,
            };
            *cursor += 1;
        }

        Self {
            constraints,
            input_nodes: input_nodes
                .into_iter()
                .map(|node| old_to_new[node as usize])
                .collect(),
            level_offsets: level_offsets.into_boxed_slice(),
            edge_offsets: edge_offsets.into_boxed_slice(),
            edges: edges.into_boxed_slice(),
            root_offsets: root_offsets.into_boxed_slice(),
            roots: roots.into_boxed_slice(),
            power_groups: raw_power_groups
                .into_iter()
                .map(|group| PowerGroup {
                    first_column: group.first_column,
                    len: group.len,
                    low_len: group.low_len,
                    full_node: old_to_new[group.full_node as usize],
                    low_node: if group.low_node == NO_NODE {
                        NO_NODE
                    } else {
                        old_to_new[group.low_node as usize]
                    },
                })
                .collect(),
            coefficients: coefficients.into_boxed_slice(),
        }
    }

    /// Number of R1CS rows, and therefore required `r` elements.
    pub const fn row_count(&self) -> usize {
        self.constraints
    }

    /// Number of integer-witness columns, including constant column zero.
    pub const fn column_count(&self) -> usize {
        self.input_nodes.len()
    }

    /// Number of live arithmetic and input nodes after pruning.
    pub const fn node_count(&self) -> usize {
        self.edge_offsets.len() - 1
    }

    /// Number of differentiated tape edges.
    pub const fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of reverse-depth batches.
    pub const fn level_count(&self) -> usize {
        self.level_offsets.len() - 1
    }

    /// Number of distinct modulus-independent integer coefficients.
    pub const fn coefficient_count(&self) -> usize {
        self.coefficients.len()
    }

    /// Bytes occupied by the tape's indexed payload and coefficient words.
    pub fn payload_bytes(&self) -> usize {
        self.input_nodes.len() * size_of::<u32>()
            + self.level_offsets.len() * size_of::<u32>()
            + self.edge_offsets.len() * size_of::<u32>()
            + self.edges.len() * size_of::<Edge>()
            + self.root_offsets.len() * size_of::<u32>()
            + self.roots.len() * size_of::<Root>()
            + self.power_groups.len() * size_of::<PowerGroup>()
            + self.coefficients.len() * size_of::<StoredInteger>()
            + self
                .coefficients
                .iter()
                .map(|coefficient| size_of_val(coefficient.words()))
                .sum::<usize>()
    }

    /// Prepares this tape for repeated evaluation modulo `modulus`.
    pub fn prepare(
        &self,
        modulus: &RuntimeModulus<2>,
    ) -> Result<PreparedWengertEvaluator<'_>, WengertApplyError> {
        self.prepare_inner(modulus, None)
    }

    fn prepare_inner(
        &self,
        modulus: &RuntimeModulus<2>,
        force_parallel: Option<bool>,
    ) -> Result<PreparedWengertEvaluator<'_>, WengertApplyError> {
        let modulus_words = *modulus.modulus_words();
        let modulus_uint = U128::from_words(modulus_words);
        let odd = Option::<Odd<U128>>::from(Odd::new(modulus_uint))
            .ok_or(WengertApplyError::EvenModulus)?;
        let params = FixedMontyParams::new_vartime(odd);
        let coefficients = parallel_map(&self.coefficients, force_parallel, |coefficient| {
            FixedMontyForm::new(&U128::from_words(modulus.reduce(coefficient)), &params)
                .to_montgomery()
                .to_words()
        });
        debug_assert_eq!(
            coefficients[0],
            FixedMontyForm::one(&params).to_montgomery().to_words()
        );
        let max_power_group_len = self
            .power_groups
            .iter()
            .map(|group| group.len as usize)
            .max()
            .unwrap_or(0);
        let mut powers_of_two = Vec::with_capacity(max_power_group_len);
        if max_power_group_len != 0 {
            let mut power = coefficients[0];
            for _ in 0..max_power_group_len {
                powers_of_two.push(power);
                power = add_mod_words(power, power, modulus_words);
            }
        }
        Ok(PreparedWengertEvaluator {
            tape: self,
            modulus_uint,
            modulus_words,
            mod_neg_inv: params.mod_neg_inv().0,
            params,
            coefficients,
            weighted_challenges: vec![[[0; 2]; 3]; self.row_count()],
            adjoints: vec![[0; 2]; self.level_offsets.last().copied().unwrap_or(0) as usize],
            output: vec![[0; 2]; self.column_count()],
            powers_of_two,
        })
    }

    /// Computes `r * (A + x B + x^2 C)` modulo a runtime two-limb prime.
    pub fn apply(
        &self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &RuntimeModulus<2>,
    ) -> Result<Vec<[u64; 2]>, WengertApplyError> {
        let mut output = Vec::new();
        self.apply_into(challenges, x, modulus, &mut output)?;
        Ok(output)
    }

    /// Computes the product into a reusable output allocation.
    pub fn apply_into(
        &self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &RuntimeModulus<2>,
        output: &mut Vec<[u64; 2]>,
    ) -> Result<(), WengertApplyError> {
        self.apply_inner(challenges, x, modulus, output, None)
    }

    fn apply_inner(
        &self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &RuntimeModulus<2>,
        output: &mut Vec<[u64; 2]>,
        force_parallel: Option<bool>,
    ) -> Result<(), WengertApplyError> {
        let mut evaluator = self.prepare_inner(modulus, force_parallel)?;
        let montgomery_challenges = parallel_map(challenges, force_parallel, |challenge| {
            evaluator.to_montgomery(*challenge)
        });
        let x = evaluator.to_montgomery(x);
        evaluator.apply_inner(&montgomery_challenges, x, force_parallel)?;
        output.resize(self.column_count(), [0; 2]);
        let parallel = force_parallel.unwrap_or_else(|| {
            rayon::current_num_threads() > 1 && output.len() >= PARALLEL_VECTOR_THRESHOLD
        });
        if parallel {
            output
                .par_iter_mut()
                .zip(evaluator.output.par_iter())
                .for_each(|(canonical, montgomery)| {
                    *canonical = evaluator.from_montgomery(*montgomery)
                });
        } else {
            output
                .iter_mut()
                .zip(evaluator.output.iter())
                .for_each(|(canonical, montgomery)| {
                    *canonical = evaluator.from_montgomery(*montgomery)
                });
        }
        Ok(())
    }
}

struct ReverseContext<'a> {
    tape: &'a WengertTape,
    modulus_words: [u64; 2],
    mod_neg_inv: u64,
    coefficients: &'a [[u64; 2]],
    weighted_challenges: &'a [[[u64; 2]; 3]],
}

impl ReverseContext<'_> {
    #[inline(always)]
    fn scale(&self, source: [u64; 2], coefficient: u32) -> [u64; 2] {
        if coefficient == 0 {
            source
        } else if coefficient == 1 {
            neg_mod_words(source, self.modulus_words)
        } else {
            montgomery_mul_2(
                source,
                self.coefficients[coefficient as usize],
                self.modulus_words,
                self.mod_neg_inv,
            )
        }
    }

    #[inline(always)]
    fn evaluate(&self, node: usize, adjoints: &[[u64; 2]]) -> [u64; 2] {
        let root_start = self.tape.root_offsets[node] as usize;
        let root_end = self.tape.root_offsets[node + 1] as usize;
        let roots = &self.tape.roots[root_start..root_end];
        let edge_start = self.tape.edge_offsets[node] as usize;
        let edge_end = self.tape.edge_offsets[node + 1] as usize;
        let edges = &self.tape.edges[edge_start..edge_end];
        if roots.is_empty() && edges.len() == 1 {
            let edge = edges[0];
            debug_assert!((edge.dependent as usize) < adjoints.len());
            return self.scale(adjoints[edge.dependent as usize], edge.coefficient);
        }

        let mut value = [0_u64; 2];
        let mut initialized = false;
        for root in roots {
            let seed = self.weighted_challenges[root.row as usize][match root.kind {
                RootKind::A => 0,
                RootKind::B => 1,
                RootKind::C => 2,
            }];
            let contribution = self.scale(seed, root.coefficient);
            if initialized {
                value = add_mod_words(value, contribution, self.modulus_words);
            } else {
                value = contribution;
                initialized = true;
            }
        }
        for edge in edges {
            debug_assert!((edge.dependent as usize) < adjoints.len());
            let contribution = self.scale(adjoints[edge.dependent as usize], edge.coefficient);
            if initialized {
                value = add_mod_words(value, contribution, self.modulus_words);
            } else {
                value = contribution;
                initialized = true;
            }
        }
        value
    }
}

impl PreparedWengertEvaluator<'_> {
    /// Converts a canonical element into Montgomery form for this modulus.
    pub fn to_montgomery(&self, canonical: [u64; 2]) -> [u64; 2] {
        FixedMontyForm::new(&reduce_words(canonical, self.modulus_uint), &self.params)
            .to_montgomery()
            .to_words()
    }

    /// Converts a Montgomery element into canonical form for this modulus.
    pub fn from_montgomery(&self, montgomery: [u64; 2]) -> [u64; 2] {
        montgomery_retrieve_2(montgomery, self.modulus_words, self.mod_neg_inv)
    }

    /// Bytes occupied by reduced coefficients and reusable apply vectors.
    pub fn workspace_bytes(&self) -> usize {
        self.coefficients.len() * size_of::<[u64; 2]>()
            + self.weighted_challenges.len() * size_of::<[[u64; 2]; 3]>()
            + self.adjoints.len() * size_of::<[u64; 2]>()
            + self.output.len() * size_of::<[u64; 2]>()
            + self.powers_of_two.len() * size_of::<[u64; 2]>()
    }

    /// Computes `r * (A + x B + x^2 C)` using Montgomery inputs and output.
    ///
    /// Every challenge and `x` must already be in Montgomery form for the
    /// prepared modulus. The output remains in Montgomery form. The returned
    /// slice remains valid until the next mutable use of this evaluator.
    pub fn apply(
        &mut self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
    ) -> Result<&[[u64; 2]], WengertApplyError> {
        self.apply_inner(challenges, x, None)?;
        Ok(&self.output)
    }

    fn apply_inner(
        &mut self,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        force_parallel: Option<bool>,
    ) -> Result<(), WengertApplyError> {
        let Self {
            tape,
            modulus_uint: _,
            modulus_words,
            mod_neg_inv,
            params: _,
            coefficients,
            weighted_challenges,
            adjoints,
            output,
            powers_of_two,
        } = self;
        if challenges.len() != tape.constraints {
            return Err(WengertApplyError::ChallengeLength {
                expected: tape.constraints,
                actual: challenges.len(),
            });
        }
        let x_squared = montgomery_mul_2(x, x, *modulus_words, *mod_neg_inv);
        let prepare_challenge = |challenge: &[u64; 2]| {
            let r = *challenge;
            let rx = montgomery_mul_2(r, x, *modulus_words, *mod_neg_inv);
            [
                r,
                rx,
                montgomery_mul_2(r, x_squared, *modulus_words, *mod_neg_inv),
            ]
        };
        let parallel = force_parallel.unwrap_or_else(|| {
            rayon::current_num_threads() > 1
                && weighted_challenges.len() >= PARALLEL_VECTOR_THRESHOLD
        });
        if parallel {
            weighted_challenges
                .par_iter_mut()
                .zip(challenges.par_iter())
                .for_each(|(weighted, challenge)| *weighted = prepare_challenge(challenge));
        } else {
            weighted_challenges
                .iter_mut()
                .zip(challenges.iter())
                .for_each(|(weighted, challenge)| *weighted = prepare_challenge(challenge));
        }

        let context = ReverseContext {
            tape,
            modulus_words: *modulus_words,
            mod_neg_inv: *mod_neg_inv,
            coefficients,
            weighted_challenges,
        };

        for level in 0..tape.level_count() {
            let start = tape.level_offsets[level] as usize;
            let end = tape.level_offsets[level + 1] as usize;
            let (prior, current_and_later) = adjoints.split_at_mut(start);
            let current = &mut current_and_later[..end - start];
            let evaluate = |new_node: usize| context.evaluate(start + new_node, prior);
            let parallel = force_parallel.unwrap_or_else(|| {
                rayon::current_num_threads() > 1 && current.len() >= PARALLEL_LEVEL_THRESHOLD
            });
            if parallel {
                current
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(node, value)| *value = evaluate(node));
            } else {
                current
                    .iter_mut()
                    .enumerate()
                    .for_each(|(node, value)| *value = evaluate(node));
            }
        }

        let evaluate_output_chunk = |chunk_index: usize, chunk: &mut [[u64; 2]]| {
            let chunk_start = chunk_index * OUTPUT_CHUNK_SIZE;
            let chunk_end = chunk_start + chunk.len();
            let groups = &tape.power_groups;
            let mut group_index = groups.partition_point(|group| {
                group.first_column as usize + group.len as usize <= chunk_start
            });
            let fill_scalars = |start: usize, values: &mut [[u64; 2]]| {
                for (offset, value) in values.iter_mut().enumerate() {
                    let node = tape.input_nodes[start + offset];
                    if node == NO_NODE {
                        *value = [0; 2];
                    } else {
                        *value = context.evaluate(node as usize, adjoints);
                    }
                }
            };
            let mut column = chunk_start;
            while column < chunk_end {
                let Some(group) = groups.get(group_index) else {
                    fill_scalars(column, &mut chunk[column - chunk_start..]);
                    break;
                };
                let group_start = group.first_column as usize;
                let group_end = group_start + group.len as usize;
                if column < group_start {
                    let scalar_end = group_start.min(chunk_end);
                    fill_scalars(
                        column,
                        &mut chunk[column - chunk_start..scalar_end - chunk_start],
                    );
                    column = scalar_end;
                    continue;
                }

                let low_end = group_start + group.low_len as usize;
                let segment_end = if column < low_end {
                    low_end.min(group_end).min(chunk_end)
                } else {
                    group_end.min(chunk_end)
                };
                let read_adjoint = |node: u32| {
                    if node == NO_NODE {
                        [0; 2]
                    } else {
                        adjoints[node as usize]
                    }
                };
                let full = read_adjoint(group.full_node);
                let mut base = full;
                if column < low_end {
                    let low = read_adjoint(group.low_node);
                    base = add_mod_words(base, low, *modulus_words);
                }
                let power = column - group_start;
                let mut value = if power == 0 || base == [0; 2] {
                    base
                } else {
                    montgomery_mul_2(base, powers_of_two[power], *modulus_words, *mod_neg_inv)
                };
                for output in &mut chunk[column - chunk_start..segment_end - chunk_start] {
                    *output = value;
                    value = add_mod_words(value, value, *modulus_words);
                }
                column = segment_end;
                if column == group_end {
                    group_index += 1;
                }
            }
        };
        let parallel = force_parallel.unwrap_or_else(|| {
            rayon::current_num_threads() > 1 && output.len() >= PARALLEL_VECTOR_THRESHOLD
        });
        if parallel {
            output
                .par_chunks_mut(OUTPUT_CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk, output)| evaluate_output_chunk(chunk, output));
        } else {
            output
                .chunks_mut(OUTPUT_CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk, output)| evaluate_output_chunk(chunk, output));
        }
        Ok(())
    }
}

fn prefix_offsets(counts: &[u32], message: &'static str) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0_u32);
    for count in counts {
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(*count)
                .expect(message),
        );
    }
    offsets
}

fn parallel_map<T: Sync, U: Send>(
    values: &[T],
    force_parallel: Option<bool>,
    apply: impl Fn(&T) -> U + Sync + Send,
) -> Vec<U> {
    let parallel = force_parallel.unwrap_or_else(|| {
        rayon::current_num_threads() > 1 && values.len() >= PARALLEL_VECTOR_THRESHOLD
    });
    if parallel {
        values.par_iter().map(apply).collect()
    } else {
        values.iter().map(apply).collect()
    }
}

fn reduce_words(words: [u64; 2], modulus: U128) -> U128 {
    let value = U128::from_words(words);
    let modulus_words = modulus.to_words();
    if words[1] < modulus_words[1] || (words[1] == modulus_words[1] && words[0] < modulus_words[0])
    {
        return value;
    }
    let nonzero = crypto_bigint::NonZero::new(modulus).expect("validated nonzero modulus");
    value.rem_vartime(&nonzero)
}

#[inline(always)]
fn carrying_mul_add(left: u64, right: u64, addend: u64, carry: u64) -> (u64, u64) {
    let value = u128::from(left) * u128::from(right) + u128::from(addend) + u128::from(carry);
    (value as u64, (value >> 64) as u64)
}

/// Two-limb FIOS Montgomery multiplication with a final canonical reduction.
#[inline(always)]
pub(crate) fn montgomery_mul_2(
    left: [u64; 2],
    right: [u64; 2],
    modulus: [u64; 2],
    mod_neg_inv: u64,
) -> [u64; 2] {
    let mut output = [0_u64; 2];
    let mut meta_carry = 0_u128;
    for left_limb in left {
        let low_product = u128::from(left_limb) * u128::from(right[0]) + u128::from(output[0]);
        let multiplier = (low_product as u64).wrapping_mul(mod_neg_inv);
        let (sum, overflow) =
            (u128::from(multiplier) * u128::from(modulus[0])).overflowing_add(low_product);
        let mut carry = (u128::from(overflow) << 64) | (sum >> 64);

        let high_product = u128::from(left_limb) * u128::from(right[1]) + u128::from(output[1]);
        let modulus_product = u128::from(multiplier) * u128::from(modulus[1]) + carry;
        let (sum, overflow) = high_product.overflowing_add(modulus_product);
        output[0] = sum as u64;
        carry = (u128::from(overflow) << 64) | (sum >> 64);

        carry += meta_carry;
        output[1] = carry as u64;
        meta_carry = carry >> 64;
    }

    let (low, low_borrow) = output[0].overflowing_sub(modulus[0]);
    let (high, first_borrow) = output[1].overflowing_sub(modulus[1]);
    let (high, second_borrow) = high.overflowing_sub(u64::from(low_borrow));
    if meta_carry == 0 && (first_borrow || second_borrow) {
        output
    } else {
        [low, high]
    }
}

/// Two-limb specialization of Montgomery retrieval. The input is reduced and
/// in Montgomery form, so HAC 14.32 needs no final conditional subtraction.
#[inline(always)]
fn montgomery_retrieve_2(value: [u64; 2], modulus: [u64; 2], mod_neg_inv: u64) -> [u64; 2] {
    let mut output = [0_u64; 2];
    for input in value {
        let multiplier = output[0].wrapping_add(input).wrapping_mul(mod_neg_inv);
        let (_, carry) = carrying_mul_add(multiplier, modulus[0], input, output[0]);
        (output[0], output[1]) = carrying_mul_add(multiplier, modulus[1], output[1], carry);
    }
    output
}

pub(crate) fn add_mod_words(left: [u64; 2], right: [u64; 2], modulus: [u64; 2]) -> [u64; 2] {
    let left = u128::from(left[0]) | (u128::from(left[1]) << 64);
    let right = u128::from(right[0]) | (u128::from(right[1]) << 64);
    let modulus = u128::from(modulus[0]) | (u128::from(modulus[1]) << 64);
    let (sum, overflow) = left.overflowing_add(right);
    let reduced = if overflow {
        sum.wrapping_sub(modulus)
    } else if sum >= modulus {
        sum - modulus
    } else {
        sum
    };
    [reduced as u64, (reduced >> 64) as u64]
}

#[inline(always)]
pub(crate) fn neg_mod_words(value: [u64; 2], modulus: [u64; 2]) -> [u64; 2] {
    if value == [0, 0] {
        return value;
    }
    let (low, borrow) = modulus[0].overflowing_sub(value[0]);
    let high = modulus[1]
        .wrapping_sub(value[1])
        .wrapping_sub(u64::from(borrow));
    [low, high]
}

/// Failure to apply a Wengert tape with the supplied runtime data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WengertApplyError {
    ChallengeLength { expected: usize, actual: usize },
    EvenModulus,
}

impl Display for WengertApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChallengeLength { expected, actual } => write!(
                formatter,
                "challenge vector has length {actual}, expected {expected}"
            ),
            Self::EvenModulus => formatter.write_str("Wengert evaluation requires an odd modulus"),
        }
    }
}

impl Error for WengertApplyError {}

/// Records and preprocesses the circuit's Z-side linear arithmetic.
#[derive(Debug)]
pub struct WengertGenerator {
    recorder: Arc<SharedRecorder>,
    inputs: Box<[WengertBit]>,
}

impl WengertGenerator {
    /// Creates a generator with the circuit's Boolean input shape.
    pub fn new(input_count: usize) -> Self {
        Self {
            recorder: Arc::new(SharedRecorder(Mutex::new(Some(Recorder::new())))),
            inputs: vec![WengertBit; input_count].into_boxed_slice(),
        }
    }

    /// Moves every Boolean input handle into a fixed-size boxed array.
    pub fn take_boxed_inputs<const N: usize>(&mut self) -> Box<[WengertBit; N]> {
        assert_eq!(N, self.inputs.len(), "input witness count mismatch");
        std::mem::take(&mut self.inputs)
            .try_into()
            .unwrap_or_else(|_| unreachable!("input length was checked"))
    }

    /// Moves dynamically sized Boolean input handles out.
    pub fn take_inputs(&mut self) -> Box<[WengertBit]> {
        std::mem::take(&mut self.inputs)
    }

    /// Prunes, transposes, and reverse-depth-orders the recorded graph.
    pub fn finish(self) -> WengertTape {
        let recorder = self
            .recorder
            .0
            .lock()
            .expect("Wengert recorder lock poisoned")
            .take()
            .expect("the Wengert generator has already been finished");
        WengertTape::from_recorder(recorder)
    }

    fn sum_terms<const LIMBS: usize>(
        &self,
        terms: impl IntoIterator<Item = WengertValue<LIMBS>>,
    ) -> WengertValue<LIMBS> {
        let mut terms: Vec<_> = terms.into_iter().filter(|term| !term.is_zero()).collect();
        match terms.len() {
            0 => WengertValue::zero(),
            1 => terms.pop().unwrap(),
            _ => {
                let terms = terms.iter().map(WengertValue::raw_term).collect::<Vec<_>>();
                let node = self
                    .recorder
                    .with_mut(|recorder| recorder.push_sum(terms.into_boxed_slice()));
                WengertValue::attached(self.recorder.clone(), node, Z::one())
            }
        }
    }
}

impl Circuit for WengertGenerator {
    type Bool = WengertBit;
    type Coefficient<const LIMBS: usize> = Z<LIMBS>;
    type Z<const LIMBS: usize> = WengertValue<LIMBS>;

    fn coefficient_from_le_words<const LIMBS: usize>(words: &[u64]) -> Z<LIMBS> {
        Z::from_le_words(words)
    }

    fn xor(&mut self, _: WengertBit, _: WengertBit) -> WengertBit {
        WengertBit
    }

    fn hint<const LIMBS: usize, const N: usize, const M: usize, H>(
        &mut self,
        _: H,
    ) -> ScalarBits<WengertBit, N>
    where
        H: Fn(
                &dyn WitnessContext<WengertValue<LIMBS>, WengertBit, Z<LIMBS>>,
            ) -> HintResult<PackedBits<N, M>>
            + Send
            + Sync
            + 'static,
    {
        assert_eq!(M, N.div_ceil(64), "incorrect packed limb count");
        ScalarBits([WengertBit; N])
    }

    fn BitZ<const LIMBS: usize>(&mut self, _: WengertBit) -> WengertValue<LIMBS> {
        let node = self.recorder.with_mut(Recorder::push_input);
        WengertValue::attached(self.recorder.clone(), node, Z::one())
    }

    fn BitZ_unsigned<const LIMBS: usize, const N: usize, const M: usize, const LOW: usize>(
        &mut self,
        _: &<WengertBit as BoolWitness>::Repr<N, M>,
    ) -> (WengertValue<LIMBS>, WengertValue<LIMBS>) {
        assert!(LOW <= N, "low part cannot be wider than the input");
        let first_column = self
            .recorder
            .with_mut(|recorder| recorder.input_nodes.len());
        let mut power = Z::<LIMBS>::one();
        let lifted: Vec<_> = (0..N)
            .map(|_| {
                let value = self.BitZ::<LIMBS>(WengertBit) * power;
                power += power;
                value
            })
            .collect();
        let full = self.sum_terms(lifted.iter().cloned());
        let low = if LOW == N {
            full.clone()
        } else {
            self.sum_terms(lifted.into_iter().take(LOW))
        };
        if N >= 2 && LOW != 1 {
            let full_node = match &full.location {
                ValueLocation::Node { node, .. } => *node,
                ValueLocation::Constant => unreachable!("a nonempty lift is not constant"),
            };
            let low_node = if LOW == 0 || LOW == N {
                NO_NODE
            } else {
                match &low.location {
                    ValueLocation::Node { node, .. } => *node,
                    ValueLocation::Constant => unreachable!("a nonempty low lift is not constant"),
                }
            };
            self.recorder.with_mut(|recorder| {
                debug_assert_eq!(recorder.input_nodes.len(), first_column + N);
                recorder.power_groups.push(RawPowerGroup {
                    first_column: u32::try_from(first_column)
                        .expect("too many integer-witness columns"),
                    len: u32::try_from(N).expect("power group is too wide"),
                    low_len: if LOW == 0 || LOW == N {
                        0
                    } else {
                        u32::try_from(LOW).expect("low power group is too wide")
                    },
                    full_node,
                    low_node,
                });
            });
        }
        (full, low)
    }

    fn assert_r1c<const LIMBS: usize>(
        &mut self,
        a: WengertValue<LIMBS>,
        b: WengertValue<LIMBS>,
        c: WengertValue<LIMBS>,
    ) {
        for value in [&a, &b, &c] {
            if let Some(recorder) = value.recorder() {
                WengertValue::<LIMBS>::same_recorder(&self.recorder, recorder);
            }
        }
        self.recorder.with_mut(|recorder| {
            let row = u32::try_from(recorder.constraints).expect("too many R1CS rows");
            recorder.constraints += 1;
            recorder.roots.extend([
                RawRoot {
                    row,
                    kind: RootKind::A,
                    term: a.raw_term(),
                },
                RawRoot {
                    row,
                    kind: RootKind::B,
                    term: b.raw_term(),
                },
                RawRoot {
                    row,
                    kind: RootKind::C,
                    term: c.raw_term(),
                },
            ]);
        });
    }

    fn sign_extend_z<const FROM_LIMBS: usize, const TO_LIMBS: usize>(
        &mut self,
        value: WengertValue<FROM_LIMBS>,
    ) -> WengertValue<TO_LIMBS> {
        assert!(
            TO_LIMBS >= FROM_LIMBS,
            "cannot sign-extend into fewer limbs"
        );
        WengertValue {
            location: value.location,
            coefficient: value.coefficient.sign_extend(),
        }
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::{BigInt, BigUint};
    use num_traits::Signed;

    use super::*;
    use crate::constraints::{ConstraintGenerator, ConstraintMatrices};
    use crate::sha256::{COMPRESSION_INPUT_BITS, compression_circuit};

    fn example_circuit<CS: Circuit>(circuit: &mut CS, inputs: &[CS::Bool; 3]) {
        let a = circuit.BitZ::<2>(inputs[0].clone());
        let b = circuit.BitZ::<2>(inputs[1].clone());
        let c = circuit.BitZ::<2>(inputs[2].clone());
        let seven = CS::Coefficient::<2>::from(7);
        let eleven = CS::Coefficient::<2>::from(11);
        let thirteen = CS::Coefficient::<2>::from(13);
        circuit.assert_r1c(
            a.clone() * seven.clone() - b.clone(),
            b.clone() * eleven + CS::Z::<2>::from(CS::Coefficient::<2>::from(5)),
            c.clone() * thirteen,
        );
        circuit.assert_r1c(
            a + c.clone(),
            -c,
            b + CS::Z::<2>::from(CS::Coefficient::<2>::from(19)),
        );
    }

    fn build_tape() -> WengertTape {
        let mut generator = WengertGenerator::new(3);
        let inputs = generator.take_boxed_inputs();
        example_circuit(&mut generator, &inputs);
        generator.finish()
    }

    fn direct_product(
        matrices: &ConstraintMatrices,
        challenges: &[[u64; 2]],
        x: [u64; 2],
        modulus: &BigUint,
    ) -> Vec<[u64; 2]> {
        let modulus_int = BigInt::from(modulus.clone());
        let as_bigint = |words: [u64; 2]| {
            BigInt::from(BigUint::from(words[0]) + (BigUint::from(words[1]) << 64_usize))
        };
        let x = as_bigint(x);
        let x_squared = &x * &x;
        let mut output = vec![BigInt::zero(); matrices.a.column_count()];
        for (row, challenge) in challenges.iter().enumerate() {
            let challenge = as_bigint(*challenge);
            for (column, coefficient) in matrices.a.rows()[row].entries() {
                output[*column] += &challenge * coefficient;
            }
            for (column, coefficient) in matrices.b.rows()[row].entries() {
                output[*column] += &challenge * &x * coefficient;
            }
            for (column, coefficient) in matrices.c.rows()[row].entries() {
                output[*column] += &challenge * &x_squared * coefficient;
            }
        }
        output
            .into_iter()
            .map(|mut value| {
                value %= &modulus_int;
                if value.is_negative() {
                    value += &modulus_int;
                }
                let words = value.to_biguint().unwrap().to_u64_digits();
                [
                    words.first().copied().unwrap_or(0),
                    words.get(1).copied().unwrap_or(0),
                ]
            })
            .collect()
    }

    #[test]
    fn tape_matches_materialized_matrices_for_primes_known_afterward() {
        let tape = build_tape();
        let mut generator = ConstraintGenerator::new(3);
        let inputs = generator.inputs();
        example_circuit(&mut generator, &inputs);
        let matrices = generator.into_matrices();
        let challenges = [[23, 0], [29, 0]];
        let x = [17, 0];

        for modulus in [
            (BigUint::one() << 127_usize) - BigUint::one(),
            (BigUint::one() << 128_usize) - BigUint::from(159_u64),
        ] {
            let runtime = RuntimeModulus::<2>::new(modulus.clone()).unwrap();
            assert_eq!(
                tape.apply(&challenges, x, &runtime).unwrap(),
                direct_product(&matrices, &challenges, x, &modulus)
            );
        }
    }

    #[test]
    fn parallel_and_sequential_reverse_batches_agree() {
        let tape = build_tape();
        let modulus =
            RuntimeModulus::<2>::new((BigUint::one() << 128_usize) - BigUint::from(159_u64))
                .unwrap();
        let challenges = [[0x1234_5678_9abc_def0, 7], [0x0fed_cba9_8765_4321, 11]];
        let mut sequential = Vec::new();
        let mut parallel = Vec::new();
        tape.apply_inner(&challenges, [31, 3], &modulus, &mut sequential, Some(false))
            .unwrap();
        tape.apply_inner(&challenges, [31, 3], &modulus, &mut parallel, Some(true))
            .unwrap();
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn prepared_evaluator_reuses_storage_and_matches_one_shot_apply() {
        let tape = build_tape();
        let modulus =
            RuntimeModulus::<2>::new((BigUint::one() << 128_usize) - BigUint::from(159_u64))
                .unwrap();
        let challenges = [[0x1234_5678_9abc_def0, 7], [0x0fed_cba9_8765_4321, 11]];
        let x = [31, 3];
        let expected = tape.apply(&challenges, x, &modulus).unwrap();
        let mut evaluator = tape.prepare(&modulus).unwrap();
        let montgomery_challenges: Vec<_> = challenges
            .into_iter()
            .map(|challenge| evaluator.to_montgomery(challenge))
            .collect();
        let montgomery_x = evaluator.to_montgomery(x);

        for _ in 0..2 {
            let output = evaluator
                .apply(&montgomery_challenges, montgomery_x)
                .unwrap()
                .to_vec();
            let output: Vec<_> = output
                .into_iter()
                .map(|value| evaluator.from_montgomery(value))
                .collect();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn two_limb_montgomery_kernel_matches_crypto_bigint() {
        let modulus_words = [u64::MAX - 158, u64::MAX];
        let modulus = U128::from_words(modulus_words);
        let params = FixedMontyParams::new_vartime(Odd::new(modulus).unwrap());
        let mod_neg_inv = params.mod_neg_inv().0;
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..1_000 {
            let canonical_left = reduce_words([random(), random()], modulus);
            let canonical_right = reduce_words([random(), random()], modulus);
            let left = FixedMontyForm::new(&canonical_left, &params);
            let right = FixedMontyForm::new(&canonical_right, &params);
            let expected = (left * right).to_montgomery().to_words();
            let actual = montgomery_mul_2(
                left.to_montgomery().to_words(),
                right.to_montgomery().to_words(),
                modulus_words,
                mod_neg_inv,
            );

            assert_eq!(actual, expected);
            assert_eq!(
                montgomery_retrieve_2(actual, modulus_words, mod_neg_inv),
                (left * right).retrieve().to_words()
            );
        }
    }

    #[test]
    fn sha256_compression_tape_matches_materialized_sparse_matrices() {
        let mut tape_generator = WengertGenerator::new(COMPRESSION_INPUT_BITS);
        let tape_inputs = tape_generator.take_boxed_inputs();
        let _ = compression_circuit(&mut tape_generator, &tape_inputs);
        let tape = tape_generator.finish();

        let mut matrix_generator = ConstraintGenerator::new(COMPRESSION_INPUT_BITS);
        let matrix_inputs = matrix_generator.boxed_inputs();
        let _ = compression_circuit(&mut matrix_generator, &matrix_inputs);
        let matrices = matrix_generator.into_matrices();

        let challenges: Vec<_> = (0..tape.row_count())
            .map(|row| [(17 * row + 3) as u64, (5 * row + 1) as u64])
            .collect();
        let x = [0x1234_5678_9abc_def0, 0x0123_4567_89ab_cdef];
        let prime = (BigUint::one() << 128_usize) - BigUint::from(159_u64);
        let modulus = RuntimeModulus::<2>::new(prime.clone()).unwrap();

        assert_eq!(tape.row_count(), matrices.a.row_count());
        assert_eq!(tape.column_count(), matrices.a.column_count());
        assert_eq!(
            tape.apply(&challenges, x, &modulus).unwrap(),
            direct_product(&matrices, &challenges, x, &prime)
        );
        assert!(tape.node_count() <= tape.edge_count());
        assert!(tape.payload_bytes() < 4 * 1024 * 1024);
    }

    #[test]
    fn rejects_wrong_challenge_length_and_even_modulus() {
        let tape = build_tape();
        let odd = RuntimeModulus::<2>::new(BigUint::from(101_u64)).unwrap();
        assert_eq!(
            tape.apply(&[[1, 0]], [2, 0], &odd),
            Err(WengertApplyError::ChallengeLength {
                expected: 2,
                actual: 1,
            })
        );
        let even = RuntimeModulus::<2>::new(BigUint::from(100_u64)).unwrap();
        assert_eq!(
            tape.apply(&[[1, 0], [2, 0]], [3, 0], &even),
            Err(WengertApplyError::EvenModulus)
        );
    }

    #[test]
    fn dead_arithmetic_is_pruned_and_unused_inputs_return_zero() {
        let mut generator = WengertGenerator::new(2);
        let inputs = generator.take_boxed_inputs::<2>();
        let used = generator.BitZ::<1>(inputs[0]);
        let unused = generator.BitZ::<1>(inputs[1]);
        let _dead = unused.clone() + unused;
        generator.assert_r1c(used, WengertValue::zero(), WengertValue::zero());
        let tape = generator.finish();
        let modulus = RuntimeModulus::<2>::new(BigUint::from(101_u64)).unwrap();
        assert_eq!(
            tape.apply(&[[7, 0]], [3, 0], &modulus).unwrap(),
            [[0, 0], [7, 0], [0, 0]]
        );
        assert_eq!(tape.node_count(), 1);
    }
}
