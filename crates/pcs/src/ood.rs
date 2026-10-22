//! Initial out-of-domain claim on the packed commitment polynomial.
//!
//! The caller binds the root and PCS parameters before this round.
//! We sample one independent field coordinate per packed variable, as in Flock's recursive OOD checks.
//! The prover sends `value = p(point)`, where `p` is the packed witness MLE.
//! The verifier derives the same point and reads the claimed value.
//!
//! This adapts BitZ Lemma B.2 with independent coordinates and the multilinear Schwartz-Zippel bound.
//! Two distinct candidates differ by a nonzero multilinear polynomial of total degree at most `point.len()`.
//! For at most `L` candidates, the collision bound is `L^2 * point.len() / (2 * 2^128)`.
//! This meets the 100-bit target for every supported witness size without initial OOD grinding.
//!
//! After ring switching, a fresh `coefficient` batches this claim into Ligerito:
//! `basis += coefficient * eq(point, ·)` and `target += coefficient * value`.
//! The claim remains borrowed from commitment state so it can be used by multiple
//! openings. Unique-decoding profiles omit this initial round.

use field::F128;
use flock_core::field::F128 as FlockF128;
use num_traits::ConstOne;
use poly::{DenseMultilinearExtension, eq_table};
use rayon::{current_num_threads, prelude::*};
use transcript::{ProverState, PublicTranscript, VerifierState};

use crate::bridge::{as_flock_f128, from_flock_f128};
use crate::{Pcs, SecurityLevel, VerifyError};

const OOD_ROUND_TAG: &[u8] = b"bitz/pcs/ood/v1";
const OOD_BATCHING_TAG: &[u8] = b"bitz/pcs/ood-batching/v1";
const BLOCK_LOG: usize = 12;
const PARALLEL_MIN_LEN: usize = 1 << 18;

/// An evaluation of the packed witness MLE, authenticated by the batched opening.
#[derive(Debug)]
pub(crate) struct OodClaim {
    /// Independent field coordinates in low-bit-first variable order.
    pub(crate) point: Vec<F128>,
    /// Claimed MLE evaluation at `point`.
    pub(crate) value: F128,
}

/// Sends the initial evaluation after binding the commitment and configuration.
/// Returns `None` without transcript events when the profile omits OOD sampling.
pub(crate) fn prove(pcs: &Pcs, packed: &[F128], transcript: &mut ProverState) -> Option<OodClaim> {
    if pcs.security_level() != SecurityLevel::Bits100 {
        return None;
    }
    let point = sample_point(pcs.packed_len(), transcript);
    let value = DenseMultilinearExtension::evaluate_exact(packed, &point);
    transcript.prover_message(&value);
    Some(OodClaim { point, value })
}

/// Reads the initial evaluation after sampling its point.
/// Reading the value does not authenticate it; the caller must verify its opening.
pub(crate) fn verify(
    pcs: &Pcs,
    transcript: &mut VerifierState<'_>,
) -> Result<Option<OodClaim>, VerifyError> {
    if pcs.security_level() != SecurityLevel::Bits100 {
        return Ok(None);
    }
    let point = sample_point(pcs.packed_len(), transcript);
    let value = transcript
        .prover_message::<F128>()
        .map_err(|_| VerifyError::MalformedProof)?;
    Ok(Some(OodClaim { point, value }))
}

/// Samples the OOD batching coefficient after the ring-switch claims are bound.
pub(crate) fn batching_challenge(transcript: &mut impl PublicTranscript) -> F128 {
    transcript.public_message(OOD_BATCHING_TAG);
    transcript.verifier_message_f128()
}

/// Adds `coefficient * eq(claim.point, ·)` to the prover's Boolean evaluation table.
pub(crate) fn add_dense_basis(basis: &mut [FlockF128], claim: &OodClaim, coefficient: F128) {
    let low = claim.point.len().min(BLOCK_LOG);
    let block = 1usize << low;
    let tail = eq_table(&claim.point[..low]);
    let head = eq_table(&claim.point[low..]);
    let update = |(chunk, &scale): (&mut [FlockF128], &F128)| {
        for (basis, &weight) in chunk.iter_mut().zip(&tail) {
            *basis += as_flock_f128(coefficient * scale * weight);
        }
    };
    if basis.len() >= PARALLEL_MIN_LEN && current_num_threads() >= 4 {
        basis
            .par_chunks_exact_mut(block)
            .zip(head.par_iter())
            .for_each(update);
    } else {
        basis.chunks_exact_mut(block).zip(&head).for_each(update);
    }
}

/// Adds the same equality polynomial after its low coordinates are fixed to `ris`.
pub(crate) fn add_succinct_basis(
    basis: &mut [FlockF128],
    claim: &OodClaim,
    coefficient: F128,
    ris: &[FlockF128],
) {
    let suffix_vars = basis.len().ilog2() as usize;
    if claim.point.len() != ris.len() + suffix_vars {
        basis.fill(FlockF128::ZERO);
        return;
    }
    let prefix = claim.point[..ris.len()]
        .iter()
        .zip(ris)
        .fold(F128::ONE, |weight, (&point, &query)| {
            weight * (F128::ONE + point + from_flock_f128(query))
        });
    let scale = coefficient * prefix;
    for (basis, weight) in basis.iter_mut().zip(eq_table(&claim.point[ris.len()..])) {
        *basis += as_flock_f128(scale * weight);
    }
}

fn sample_point(packed_len: usize, transcript: &mut impl PublicTranscript) -> Vec<F128> {
    transcript.public_message(OOD_ROUND_TAG);
    (0..packed_len.ilog2())
        .map(|_| transcript.verifier_message_f128())
        .collect()
}

#[cfg(test)]
mod tests {
    use num_traits::ConstZero;
    use rayon::ThreadPoolBuilder;
    use transcript::build_prover;

    use super::*;

    #[test]
    fn each_ood_coordinate_consumes_a_fresh_challenge() {
        for variables in 13..=28 {
            let mut prover = build_prover(b"ood-test", b"independent-coordinates");
            let mut expected = build_prover(b"ood-test", b"independent-coordinates");
            expected.public_message(OOD_ROUND_TAG);
            let point = sample_point(1 << variables, &mut prover);
            assert_eq!(point.len(), variables);
            for coordinate in point {
                assert_eq!(coordinate, expected.verifier_message::<F128>());
            }
            assert_eq!(
                prover.verifier_message::<F128>(),
                expected.verifier_message::<F128>()
            );
            assert!(prover.finish().narg_string.is_empty());
        }
    }

    #[test]
    fn dense_basis_updates_match_the_full_equality_table_across_thread_counts() {
        let pools: Vec<_> = [1, 2, 4, 8]
            .into_iter()
            .map(|threads| {
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .collect();
        for log_len in [0, 8, 12, 17, 18, 19] {
            let len = 1usize << log_len;
            let claim = OodClaim {
                point: sample_point(len, &mut build_prover(b"ood-test", b"dense-basis")),
                value: F128::ZERO,
            };
            let weights = eq_table(&claim.point);
            let initial: Vec<_> = (0..len)
                .map(|i| FlockF128::new(i as u64 + 1, (i as u64).rotate_left(17) + 3))
                .collect();
            for coefficient in [F128::ZERO, F128::ONE, F128::new(13, 17)] {
                let expected: Vec<_> = initial
                    .iter()
                    .zip(&weights)
                    .map(|(&value, &weight)| value + as_flock_f128(coefficient * weight))
                    .collect();
                for pool in &pools {
                    let mut actual = initial.clone();
                    pool.install(|| add_dense_basis(&mut actual, &claim, coefficient));
                    assert_eq!(
                        actual,
                        expected,
                        "log_len={log_len}, threads={}",
                        pool.current_num_threads()
                    );
                }
            }
        }
    }

    #[test]
    fn dense_and_succinct_ood_bases_agree_after_folding() {
        let point = sample_point(1 << 14, &mut build_prover(b"ood-test", b"folded-basis"));
        let coefficient = F128::new(13, 17);
        let claim = OodClaim {
            point,
            value: F128::ZERO,
        };
        let mut dense = vec![FlockF128::ZERO; 1 << 14];
        add_dense_basis(&mut dense, &claim, coefficient);
        let queries: Vec<_> = (0..9)
            .map(|i| as_flock_f128(F128::new(i + 2, i + 19)))
            .collect();
        for &query in &queries {
            for i in 0..dense.len() / 2 {
                dense[i] = dense[2 * i] + query * (dense[2 * i] + dense[2 * i + 1]);
            }
            dense.truncate(dense.len() / 2);
        }
        let mut succinct = vec![FlockF128::ZERO; dense.len()];
        add_succinct_basis(&mut succinct, &claim, coefficient, &queries);
        assert_eq!(dense, succinct);
    }
}
