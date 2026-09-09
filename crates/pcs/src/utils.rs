//! Shared transcript rules for PCS openings.

use crate::bridge::{as_flock_f128, from_flock_f128};
use crate::{OpeningQuery, Pcs, VerifyError};
use flock_core::field::F128 as FlockF128;
use flock_core::pcs::pack::PACKING_WIDTH as CLAIM_COUNT;
pub(crate) use transcript::PublicTranscript;
use transcript::{ProverState, VerifierState};

const MLE_STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v1";
const INNER_PRODUCT_STATEMENT_LABEL: &[u8] = b"f2z/pcs/bit-inner-product/v1";
const CHALLENGES_LABEL: &[u8] = b"f2z/pcs/ring-switch-challenges/v1";

/// Writes one fixed ring-switch claim array.
pub(crate) fn write_claims(
    transcript: &mut ProverState,
    query: &OpeningQuery,
    claims: &[FlockF128; CLAIM_COUNT],
) {
    transcript.public_message(query.label());
    let claims: [field::F128; CLAIM_COUNT] =
        core::array::from_fn(|index| from_flock_f128(claims[index]));
    transcript.prover_message(&claims);
}

/// Reads one fixed ring-switch claim array.
pub(crate) fn read_claims(
    transcript: &mut VerifierState<'_>,
    query: &OpeningQuery,
) -> Result<[FlockF128; CLAIM_COUNT], VerifyError> {
    transcript.public_message(query.label());
    transcript
        .prover_message::<[field::F128; CLAIM_COUNT]>()
        .map(|claims| claims.map(as_flock_f128))
        .map_err(|_| VerifyError::MalformedProof)
}

/// Samples one fixed group of ring-switch challenges.
pub(crate) fn sample_challenges<const N: usize>(
    transcript: &mut impl PublicTranscript,
) -> [FlockF128; N] {
    transcript.public_message(CHALLENGES_LABEL);
    core::array::from_fn(|_| as_flock_f128(transcript.verifier_message_f128()))
}

/// Absorbs an MLE statement in either transcript.
pub(crate) fn bind_mle_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    point: &[field::F128],
    target: field::F128,
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(MLE_STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(point.len() as u64));
    for coordinate in point {
        transcript.public_message(coordinate);
    }
    transcript.public_message(&target);
}

/// Absorbs an arbitrary original-bit inner-product statement in either transcript.
pub(crate) fn bind_inner_product_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    weights: &[field::F128],
    target: field::F128,
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(INNER_PRODUCT_STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(weights.len() as u64));
    transcript.public_message(&weight_digest(weights));
    transcript.public_message(&target);
}

fn weight_digest(weights: &[field::F128]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"f2z/pcs/bit-inner-product-weights/v1");
    hasher.update(&(weights.len() as u64).to_le_bytes());
    for weight in weights {
        hasher.update(&weight.to_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use ::transcript::{build_prover, build_verifier};
    use common::Shape;
    use field::F128;
    use proptest::prelude::*;

    use super::*;
    use crate::{HashKind, LigeritoProfile};

    #[test]
    fn fixed_claim_arrays_round_trip_and_separate_domains() {
        let claims = core::array::from_fn(|index| FlockF128::new(index as u64, 0));
        let mut challenges = [F128::default(); 2];

        for (index, query) in [
            OpeningQuery::Mle {
                point: Vec::new(),
                target: F128::default(),
            },
            OpeningQuery::InnerProduct {
                weights: Vec::new(),
                target: F128::default(),
            },
        ]
        .iter()
        .enumerate()
        {
            let mut prover = build_prover(b"pcs-protocol-test", b"fixed-claims");
            write_claims(&mut prover, query, &claims);
            challenges[index] = prover.verifier_message();
            let proof = prover.finish();
            assert_eq!(proof.narg_string.len(), CLAIM_COUNT * 16);

            let mut verifier = build_verifier(b"pcs-protocol-test", b"fixed-claims", &proof);
            assert_eq!(read_claims(&mut verifier, query).unwrap(), claims);
            assert_eq!(verifier.verifier_message::<F128>(), challenges[index]);
            verifier.check_eof().unwrap();
        }

        assert_ne!(challenges[0], challenges[1]);
    }

    proptest! {
        #[test]
        fn mle_statement_binding_matches_between_roles(
            root in any::<[u8; 32]>(),
            point_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..32),
            target_words in (any::<u64>(), any::<u64>()),
        ) {
            let shape = Shape::new(7, 15).unwrap();
            let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
            let point = point_words
                .iter()
                .map(|&(lo, hi)| F128::new(lo, hi))
                .collect::<Vec<_>>();
            let target = F128::new(target_words.0, target_words.1);

            let mut prover = build_prover(b"pcs-protocol-test", b"statement-binding");
            bind_mle_statement(&pcs, &root, &point, target, &mut prover);
            let expected = prover.verifier_message::<F128>();
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"statement-binding",
                &proof,
            );
            bind_mle_statement(&pcs, &root, &point, target, &mut verifier);
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }

        #[test]
        fn inner_product_statement_binding_matches_between_roles(
            root in any::<[u8; 32]>(),
            weight_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..32),
            target_words in (any::<u64>(), any::<u64>()),
        ) {
            let shape = Shape::new(7, 15).unwrap();
            let pcs = Pcs::new(&shape, LigeritoProfile::Secure, HashKind::Blake3).unwrap();
            let weights = weight_words
                .iter()
                .map(|&(lo, hi)| F128::new(lo, hi))
                .collect::<Vec<_>>();
            let target = F128::new(target_words.0, target_words.1);

            let mut prover = build_prover(b"pcs-protocol-test", b"inner-product-statement");
            bind_inner_product_statement(&pcs, &root, &weights, target, &mut prover);
            let expected = prover.verifier_message::<F128>();
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"inner-product-statement",
                &proof,
            );
            bind_inner_product_statement(&pcs, &root, &weights, target, &mut verifier);
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }
    }
}
