//! Shared proof transport, transcript, and wire rules for PCS openings.

mod proof;
mod transcript;
mod wire;

pub(crate) use proof::{read_opening_proof, write_opening_proof};
pub(crate) use transcript::PublicTranscript;
pub(crate) use wire::{
    bind_ring_switch_message, observe_opening_target, read_inner_product_coordinates,
    sample_inner_product_batching_challenges, sample_ring_switch_point,
    write_inner_product_coordinates,
};

use crate::Pcs;

const STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v1";
const INNER_PRODUCT_STATEMENT_LABEL: &[u8] = b"f2z/pcs/bit-inner-product/v1";

/// Absorbs the public statement in either transcript.
pub(crate) fn bind_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    point: &[field::F128],
    target: field::F128,
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(STATEMENT_LABEL);
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

    proptest! {
        #[test]
        fn statement_binding_matches_between_roles(
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
            bind_statement(&pcs, &root, &point, target, &mut prover);
            let expected = prover.verifier_message::<F128>();
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"statement-binding",
                &proof,
            );
            bind_statement(&pcs, &root, &point, target, &mut verifier);
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
