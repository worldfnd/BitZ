use crate::{CommitError, Pcs, ScopedOpeningQuery};

use super::{NO_SCOPE, PublicTranscript, STATEMENT_LABEL};

// One more 2,056-byte ring-switch proof exceeds the 64 MiB hint limit.
const MAX_BATCH_QUERIES: usize = 32_640;

pub(crate) fn validate_batch(
    queries: &[ScopedOpeningQuery<'_>],
    expected_m: usize,
) -> Result<(), CommitError> {
    if queries.is_empty() {
        return Err(CommitError::EmptyBatch);
    }
    if queries.len() > MAX_BATCH_QUERIES {
        return Err(CommitError::ProofTooLarge);
    }
    if queries
        .iter()
        .any(|scoped_query| scoped_query.scope == NO_SCOPE)
    {
        return Err(CommitError::InvalidClaimScope);
    }
    if queries
        .windows(2)
        .any(|pair| pair[0].scope >= pair[1].scope)
    {
        return Err(CommitError::InvalidClaimScopeOrder);
    }
    if queries
        .iter()
        .any(|scoped_query| scoped_query.query.point.len() != expected_m)
    {
        return Err(CommitError::PointLengthMismatch);
    }
    Ok(())
}

/// Absorbs the public statement (batch query) in either transcript.
pub(crate) fn bind_statement(
    pcs: &Pcs,
    root: &[u8; 32],
    queries: &[ScopedOpeningQuery<'_>],
    transcript: &mut impl PublicTranscript,
) {
    transcript.public_message(STATEMENT_LABEL);
    transcript.public_message(root);
    transcript.public_message(pcs);
    transcript.public_message(&(queries.len() as u64));
    for scoped_query in queries {
        transcript.public_message(&scoped_query.scope);
        let query = scoped_query.query;
        transcript.public_message(&(query.point.len() as u64));
        for coordinate in &query.point {
            transcript.public_message(coordinate);
        }
        transcript.public_message(&query.target);
    }
}

#[cfg(test)]
mod tests {
    use field::F128;
    use proptest::prelude::*;
    use transcript::{build_prover, build_verifier};

    use super::*;
    use crate::{HashKind, LigeritoProfile, OpeningQuery, ScopedOpeningQuery};

    proptest! {
        #[test]
        fn statement_binding_matches_between_roles(
            root in any::<[u8; 32]>(),
            point_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..32),
            target_words in (any::<u64>(), any::<u64>()),
        ) {
            let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
            let query = OpeningQuery {
                point: point_words
                    .iter()
                    .map(|&(lo, hi)| F128::new(lo, hi))
                    .collect(),
                target: F128::new(target_words.0, target_words.1),
            };
            let scoped_query = ScopedOpeningQuery::new(7, &query);

            let mut prover = build_prover(b"pcs-protocol-test", b"statement-binding");
            bind_statement(
                &pcs,
                &root,
                core::slice::from_ref(&scoped_query),
                &mut prover,
            );
            let expected = prover.verifier_message::<F128>();
            let proof = prover.finish();

            let mut verifier = build_verifier(
                b"pcs-protocol-test",
                b"statement-binding",
                &proof,
            );
            bind_statement(
                &pcs,
                &root,
                core::slice::from_ref(&scoped_query),
                &mut verifier,
            );
            prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
            prop_assert!(verifier.check_eof().is_ok());
        }
    }

    fn statement_challenge(queries: &[ScopedOpeningQuery<'_>]) -> F128 {
        let pcs = Pcs::new(22, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let mut prover = build_prover(b"pcs-protocol-test", b"batch-binding");
        bind_statement(&pcs, &[7; 32], queries, &mut prover);
        prover.verifier_message::<F128>()
    }

    #[test]
    fn statement_binding_commits_to_batch_order_and_count() {
        let first = OpeningQuery {
            point: vec![F128::from(1u64), F128::from(2u64)],
            target: F128::from(3u64),
        };
        let second = OpeningQuery {
            point: vec![F128::from(4u64), F128::from(5u64)],
            target: F128::from(6u64),
        };

        let ordered = statement_challenge(&[
            ScopedOpeningQuery::new(0, &first),
            ScopedOpeningQuery::new(2, &second),
        ]);
        let reversed = statement_challenge(&[
            ScopedOpeningQuery::new(0, &second),
            ScopedOpeningQuery::new(2, &first),
        ]);
        let changed_scope = statement_challenge(&[
            ScopedOpeningQuery::new(0, &first),
            ScopedOpeningQuery::new(3, &second),
        ]);
        let prefix = statement_challenge(&[ScopedOpeningQuery::new(0, &first)]);

        assert_ne!(ordered, reversed);
        assert_ne!(ordered, changed_scope);
        assert_ne!(ordered, prefix);
    }

    #[test]
    fn batch_validation_requires_strictly_increasing_scopes() {
        let query = OpeningQuery {
            point: vec![F128::default(); 22],
            target: F128::default(),
        };

        assert_eq!(
            validate_batch(
                &[
                    ScopedOpeningQuery::new(2, &query),
                    ScopedOpeningQuery::new(2, &query),
                ],
                22,
            ),
            Err(CommitError::InvalidClaimScopeOrder)
        );
        assert_eq!(
            validate_batch(
                &[
                    ScopedOpeningQuery::new(3, &query),
                    ScopedOpeningQuery::new(2, &query),
                ],
                22,
            ),
            Err(CommitError::InvalidClaimScopeOrder)
        );
        assert_eq!(
            validate_batch(&[ScopedOpeningQuery::new(u32::MAX, &query)], 22),
            Err(CommitError::InvalidClaimScope)
        );
    }

    #[test]
    fn batch_validation_rejects_batches_that_cannot_fit_the_hint() {
        let query = OpeningQuery {
            point: vec![F128::default(); 22],
            target: F128::default(),
        };
        let queries = vec![ScopedOpeningQuery::new(0, &query); MAX_BATCH_QUERIES + 1];

        assert_eq!(
            validate_batch(&queries, 22),
            Err(CommitError::ProofTooLarge)
        );
    }
}
