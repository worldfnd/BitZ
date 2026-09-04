//! The batched product-tree prover and verifier.
//!
//! Both roles stream: the prover writes each layer's round polynomials and
//! child evaluations onto the narg channel as it goes, and the verifier reads
//! and checks them in the same order. Nothing crosses the boundary as a proof
//! struct, so a caller cannot forget to transmit one, and the whole argument
//! lands in the same byte stream as the fold round that precedes it.

use crypto_primitives::ConstField;
use poly::eq_eval;
use transcript::{Encoding, NargDeserialize, ProverState, TranscriptChallenge, VerifierState};

/// A forest the verifier rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForestError {
    /// No trees, a tree of depth zero, or depths that do not descend.
    InvalidDepths,
    /// A leaf count is not a power of two.
    MalformedLeaves,
    /// The caller's root is not the product of the leaves it handed over.
    RootMismatch,
    /// A record is missing or does not decode.
    MalformedProof,
    /// Layer zero's children do not multiply to the root.
    RootNotChildProduct,
    /// A round polynomial does not reduce the claim it was given.
    RoundClaimMismatch,
    /// A layer sumcheck's terminal value is not `eq * left * right`.
    TerminalMismatch,
}

/// Where one tree's leaf layer was queried, and what it evaluated to.
///
/// The caller is what makes this mean anything: until it ties `evaluation` to
/// the actual leaves at `point`, the forest has proved a product of values
/// nobody has pinned down.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeClaim<F> {
    pub point: Vec<F>,
    pub evaluation: F,
}

/// Builds every layer of one tree, leaves last.
fn build_layers<F: ConstField + Copy>(leaves: Vec<F>) -> Vec<Vec<F>> {
    let depth = leaves.len().trailing_zeros() as usize;
    let mut layers = vec![Vec::new(); depth + 1];
    layers[depth] = leaves;
    for k in (0..depth).rev() {
        let child = &layers[k + 1];
        let half = 1usize << k;
        layers[k] = (0..half).map(|i| child[i] * child[i + half]).collect();
    }
    layers
}

/// `[c0, c1, c2, c3]` of `(eq0 + X d_eq)(l0 + X d_l)(r0 + X d_r)`.
#[inline]
fn cubic_coefficients<F: ConstField + Copy>(eq: [F; 2], left: [F; 2], right: [F; 2]) -> [F; 4] {
    let (eq_zero, eq_delta) = (eq[0], eq[1] - eq[0]);
    let (left_zero, left_delta) = (left[0], left[1] - left[0]);
    let (right_zero, right_delta) = (right[0], right[1] - right[0]);

    [
        eq_zero * left_zero * right_zero,
        eq_zero * left_zero * right_delta
            + eq_zero * left_delta * right_zero
            + eq_delta * left_zero * right_zero,
        eq_zero * left_delta * right_delta
            + eq_delta * left_zero * right_delta
            + eq_delta * left_delta * right_zero,
        eq_delta * left_delta * right_delta,
    ]
}

/// Binds a table's lowest-index variable to `challenge`, halving it.
fn fold_in_place<F: ConstField + Copy>(table: &mut Vec<F>, challenge: F) {
    let half = table.len() / 2;
    for i in 0..half {
        let (zero, one) = (table[2 * i], table[2 * i + 1]);
        table[i] = zero + challenge * (one - zero);
    }
    table.truncate(half);
}

fn horner<F: ConstField + Copy>(coefficients: &[F; 4], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(F::ZERO, |value, coefficient| value * point + coefficient)
}

/// The per-tree scales `1, rho, rho^2, ...` that merge the active claims.
///
/// A single active tree needs no batching, so no challenge is drawn for it.
fn batching_scales<F: ConstField + Copy>(active: usize, rho: Option<F>) -> Vec<F> {
    let mut scales = Vec::with_capacity(active);
    let mut scale = F::ONE;
    for _ in 0..active {
        scales.push(scale);
        if let Some(rho) = rho {
            scale *= rho;
        }
    }
    scales
}

fn check_depths(depths: &[usize], root_count: usize) -> Result<(), ForestError> {
    if depths.is_empty()
        || depths.len() != root_count
        || depths.contains(&0)
        || depths.windows(2).any(|pair| pair[0] < pair[1])
    {
        return Err(ForestError::InvalidDepths);
    }
    Ok(())
}

/// How many trees are still being reduced at layer `k`.
///
/// Depths descend, so this is always a prefix, which is what lets one sumcheck
/// cover every active tree.
fn active_at(depths: &[usize], k: usize) -> usize {
    depths.iter().filter(|&&d| d > k).count()
}

/// Proves `prod leaves_t = root_t` for every tree, writing the argument to the
/// transcript.
///
/// `leaves_per_tree` must be in non-increasing length order, every length a
/// power of two above one. `roots` is public on both sides: F2Z derives it
/// from the folds the prover already sent, so it never rides the proof.
pub fn prove_product_forest<F>(
    transcript: &mut ProverState,
    leaves_per_tree: Vec<Vec<F>>,
    roots: &[F],
) -> Result<Vec<TreeClaim<F>>, ForestError>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
    let depths: Vec<usize> = leaves_per_tree
        .iter()
        .map(|leaves| leaves.len().trailing_zeros() as usize)
        .collect();
    check_depths(&depths, roots.len())?;
    if leaves_per_tree
        .iter()
        .any(|leaves| !leaves.len().is_power_of_two())
    {
        return Err(ForestError::MalformedLeaves);
    }

    let trees: Vec<Vec<Vec<F>>> = {
        let _guard = prof::scope("gkr/build-trees");
        leaves_per_tree.into_iter().map(build_layers).collect()
    };
    for (tree, &root) in trees.iter().zip(roots) {
        if tree[0][0] != root {
            return Err(ForestError::RootMismatch);
        }
    }

    transcript.public_message(&roots.to_vec());

    let mut values: Vec<F> = roots.to_vec();
    let mut points: Vec<Vec<F>> = vec![Vec::new(); depths.len()];

    let _guard = prof::scope("gkr/layers");
    for k in 0..depths[0] {
        let active = active_at(&depths, k);
        let half = 1usize << k;

        // The halves of layer `k + 1`, which this layer reduces its claim to.
        let mut lefts: Vec<Vec<F>> = (0..active)
            .map(|t| trees[t][k + 1][..half].to_vec())
            .collect();
        let mut rights: Vec<Vec<F>> = (0..active)
            .map(|t| trees[t][k + 1][half..].to_vec())
            .collect();

        let mut shared_point = Vec::with_capacity(k);
        if k > 0 {
            let rho = (active > 1).then(|| transcript.squeeze::<F>());
            let scales = batching_scales(active, rho);
            let mut equalities: Vec<Vec<F>> =
                (0..active).map(|t| poly::eq_table(&points[t])).collect();

            for _ in 0..k {
                let mut coefficients = [F::ZERO; 4];
                for t in 0..active {
                    let (eq, left, right) = (&equalities[t], &lefts[t], &rights[t]);
                    let mut tree_sum = [F::ZERO; 4];
                    for pair in 0..eq.len() / 2 {
                        let (lo, hi) = (2 * pair, 2 * pair + 1);
                        let term = cubic_coefficients(
                            [eq[lo], eq[hi]],
                            [left[lo], left[hi]],
                            [right[lo], right[hi]],
                        );
                        for (accumulated, added) in tree_sum.iter_mut().zip(term) {
                            *accumulated += added;
                        }
                    }
                    for (accumulated, added) in coefficients.iter_mut().zip(tree_sum) {
                        *accumulated += scales[t] * added;
                    }
                }

                transcript.prover_message(&coefficients);
                let challenge = transcript.squeeze::<F>();
                shared_point.push(challenge);

                for t in 0..active {
                    fold_in_place(&mut equalities[t], challenge);
                    fold_in_place(&mut lefts[t], challenge);
                    fold_in_place(&mut rights[t], challenge);
                }
            }
        }

        for t in 0..active {
            transcript.prover_message(&[lefts[t][0], rights[t][0]]);
        }

        let lambda = transcript.squeeze::<F>();
        for t in 0..active {
            let (left, right) = (lefts[t][0], rights[t][0]);
            values[t] = left + lambda * (right - left);
            points[t] = shared_point.clone();
            points[t].push(lambda);
        }
    }

    Ok(points
        .into_iter()
        .zip(values)
        .map(|(point, evaluation)| TreeClaim { point, evaluation })
        .collect())
}

/// Replays the forest and returns, per tree, where its leaves were queried and
/// what they must evaluate to.
pub fn verify_product_forest<F>(
    transcript: &mut VerifierState<'_>,
    depths: &[usize],
    roots: &[F],
) -> Result<Vec<TreeClaim<F>>, ForestError>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge + NargDeserialize,
{
    check_depths(depths, roots.len())?;

    transcript.public_message(&roots.to_vec());

    let mut values: Vec<F> = roots.to_vec();
    let mut points: Vec<Vec<F>> = vec![Vec::new(); depths.len()];

    for k in 0..depths[0] {
        let active = active_at(depths, k);

        let mut shared_point = Vec::with_capacity(k);
        let mut claim = F::ZERO;
        let mut scales = Vec::new();
        if k > 0 {
            let rho = (active > 1).then(|| transcript.squeeze::<F>());
            scales = batching_scales(active, rho);
            claim = (0..active).fold(F::ZERO, |sum, t| sum + scales[t] * values[t]);

            for _ in 0..k {
                let coefficients = transcript
                    .prover_message::<[F; 4]>()
                    .map_err(|_| ForestError::MalformedProof)?;
                let at_one = coefficients
                    .iter()
                    .copied()
                    .fold(F::ZERO, |sum, coefficient| sum + coefficient);
                if coefficients[0] + at_one != claim {
                    return Err(ForestError::RoundClaimMismatch);
                }
                let challenge = transcript.squeeze::<F>();
                claim = horner(&coefficients, challenge);
                shared_point.push(challenge);
            }
        }

        let children = (0..active)
            .map(|_| {
                transcript
                    .prover_message::<[F; 2]>()
                    .map_err(|_| ForestError::MalformedProof)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if k == 0 {
            for (t, child) in children.iter().enumerate() {
                if values[t] != child[0] * child[1] {
                    return Err(ForestError::RootNotChildProduct);
                }
            }
        } else {
            let expected = children
                .iter()
                .enumerate()
                .fold(F::ZERO, |sum, (t, child)| {
                    sum + scales[t] * eq_eval(&shared_point, &points[t]) * child[0] * child[1]
                });
            if expected != claim {
                return Err(ForestError::TerminalMismatch);
            }
        }

        let lambda = transcript.squeeze::<F>();
        for (t, child) in children.iter().enumerate() {
            let (left, right) = (child[0], child[1]);
            values[t] = left + lambda * (right - left);
            points[t] = shared_point.clone();
            points[t].push(lambda);
        }
    }

    Ok(points
        .into_iter()
        .zip(values)
        .map(|(point, evaluation)| TreeClaim { point, evaluation })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::{F128, FqDefault};
    use num_traits::ConstOne;
    use poly::DenseMultilinearExtension;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use transcript::{build_prover, build_verifier};

    const SESSION: &[u8] = b"gkr/product-forest/test";

    fn leaves<F: ConstField + Copy>(rng: &mut Pcg64, depth: usize) -> Vec<F> {
        (0..1usize << depth)
            .map(|_| F::from(rng.random::<u128>()))
            .collect()
    }

    fn product<F: ConstField + Copy>(values: &[F]) -> F {
        values.iter().fold(F::ONE, |acc, &v| acc * v)
    }

    /// Proves a forest, replays it, and checks that each returned claim really
    /// is that tree's leaf MLE at the returned point.
    fn round_trip<F>(depths: &[usize], instance: &[u8])
    where
        F: ConstField
            + Copy
            + Encoding<[u8]>
            + TranscriptChallenge
            + NargDeserialize
            + std::fmt::Debug,
    {
        let mut rng = Pcg64::seed_from_u64(0xf0_1e_57);
        let trees: Vec<Vec<F>> = depths.iter().map(|&d| leaves(&mut rng, d)).collect();
        let roots: Vec<F> = trees.iter().map(|t| product(t)).collect();

        let mut prover = build_prover(SESSION, instance);
        let prover_claims = prove_product_forest(&mut prover, trees.clone(), &roots).unwrap();
        let transcript_proof = prover.finish();

        let mut verifier = build_verifier(SESSION, instance, &transcript_proof);
        let claims = verify_product_forest(&mut verifier, depths, &roots)
            .unwrap_or_else(|e| panic!("{depths:?}: {e:?}"));
        verifier.check_eof().unwrap();

        assert_eq!(claims, prover_claims, "roles disagree on the leaf claims");
        for (t, claim) in claims.iter().enumerate() {
            assert_eq!(claim.point.len(), depths[t]);
            let mle =
                DenseMultilinearExtension::from_evaluations(depths[t], trees[t].clone()).unwrap();
            assert_eq!(
                claim.evaluation,
                mle.evaluate(&claim.point).unwrap(),
                "tree {t} claim is not its leaf MLE at the returned point"
            );
        }
    }

    #[test]
    fn single_tree_round_trips_at_every_depth() {
        for depth in 1..=8 {
            round_trip::<FqDefault>(&[depth], b"single-fq");
            round_trip::<F128>(&[depth], b"single-f128");
        }
    }

    #[test]
    fn equal_depth_forests_round_trip() {
        for trees in [2usize, 3, 5, 8] {
            round_trip::<FqDefault>(&vec![5; trees], b"equal-fq");
            round_trip::<F128>(&vec![5; trees], b"equal-f128");
        }
    }

    #[test]
    fn ragged_forests_round_trip() {
        round_trip::<FqDefault>(&[6, 6, 4, 3, 1], b"ragged-fq");
        round_trip::<F128>(&[6, 6, 4, 3, 1], b"ragged-f128");
        round_trip::<FqDefault>(&[4, 1, 1, 1], b"ragged2-fq");
    }

    #[test]
    fn a_tampered_root_is_rejected() {
        let mut rng = Pcg64::seed_from_u64(7);
        let trees: Vec<Vec<FqDefault>> = vec![leaves(&mut rng, 4)];
        let roots = vec![product(&trees[0])];

        let mut prover = build_prover(SESSION, b"wrong-root");
        prove_product_forest(&mut prover, trees, &roots).unwrap();
        let transcript_proof = prover.finish();

        let tampered = vec![roots[0] + FqDefault::ONE];
        let mut verifier = build_verifier(SESSION, b"wrong-root", &transcript_proof);
        assert_eq!(
            verify_product_forest(&mut verifier, &[4], &tampered),
            Err(ForestError::RootNotChildProduct)
        );
    }

    #[test]
    fn the_prover_rejects_a_root_that_is_not_the_product() {
        let mut rng = Pcg64::seed_from_u64(11);
        let trees: Vec<Vec<FqDefault>> = vec![leaves(&mut rng, 3)];
        let mut prover = build_prover(SESSION, b"bad-root");
        assert_eq!(
            prove_product_forest(&mut prover, trees, &[FqDefault::ONE]),
            Err(ForestError::RootMismatch)
        );
    }

    #[test]
    fn ascending_depths_are_rejected() {
        let mut rng = Pcg64::seed_from_u64(13);
        let trees: Vec<Vec<FqDefault>> = vec![leaves(&mut rng, 2), leaves(&mut rng, 4)];
        let roots: Vec<FqDefault> = trees.iter().map(|t| product(t)).collect();
        let mut prover = build_prover(SESSION, b"ascending");
        assert_eq!(
            prove_product_forest(&mut prover, trees, &roots),
            Err(ForestError::InvalidDepths)
        );
    }
}
