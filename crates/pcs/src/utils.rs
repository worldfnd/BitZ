//! Shared proof transport, transcript, and wire rules for multilinear openings.

mod proof;
mod transcript;
mod wire;

pub(crate) use proof::{read_opening_proof, write_opening_proof};
pub(crate) use transcript::PublicTranscript;
pub(crate) use wire::{bind_ring_switch_message, observe_opening_target, sample_ring_switch_point};

use crate::{OpeningQuery, Pcs};

const STATEMENT_LABEL: &[u8] = b"f2z/pcs/mle-opening/v1";

/// Absorbs the public statement in either transcript.
pub(crate) fn bind_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    query: &OpeningQuery,
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(query.point.len() as u64));
    for coordinate in &query.point {
        transcript.public_message(coordinate);
    }
    transcript.public_message(&query.target);
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
            let query = OpeningQuery {
                point: point_words
                    .iter()
                    .map(|&(lo, hi)| F128::new(lo, hi))
                    .collect(),
                target: F128::new(target_words.0, target_words.1),
            };

            let mut prover = build_prover(b"pcs-protocol-test", b"statement-binding");
            bind_statement(&pcs, &root, &query, &mut prover);
            let expected = prover.verifier_message::<F128>();
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"statement-binding",
                &proof,
            );
            bind_statement(&pcs, &root, &query, &mut verifier);
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }
    }
}
