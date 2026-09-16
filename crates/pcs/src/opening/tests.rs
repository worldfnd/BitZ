use std::sync::OnceLock;

use common::{LinearClaim, Shape};
use num_traits::ConstZero;
use proptest::prelude::*;
use transcript::{build_prover, build_verifier};

use super::*;
use crate::{HashKind, LigeritoProfile};

const M: usize = 22;
const SINGLETON: usize = (1 << 21) | (1 << 7) | 0b101_0101;
const SESSION: &[u8] = b"pcs-reduction-wiring-test";
const INSTANCE: &[u8] = b"singleton";

struct Fixture {
    pcs: Pcs,
    root: Root,
    data: ProverData,
    witness: Vec<F128>,
    claim: LinearClaim<F128>,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let shape = Shape::new(7, M - 7).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let mut witness = vec![F128::ZERO; pcs.packed_len()];
        witness[SINGLETON / 128].hi = 1 << (SINGLETON % 128 - 64);
        let (root, data) = pcs.commit(&witness).unwrap();
        let rows = (0..shape.rows())
            .map(|row| F128::from(row as u64 + 2))
            .collect::<Vec<_>>();
        let columns = (0..shape.columns())
            .map(|column| F128::from(column as u64 + 3))
            .collect::<Vec<_>>();
        let target = rows[SINGLETON % shape.rows()] * columns[SINGLETON / shape.rows()];
        let claim = LinearClaim::from_shape(&shape, rows, columns, target).unwrap();
        Fixture {
            pcs,
            root,
            data,
            witness,
            claim,
        }
    })
}

#[test]
fn inner_product_proof_composes_sumcheck_with_a_bound_mle_opening() {
    let fixture = fixture();
    let mut prover = build_prover(SESSION, INSTANCE);
    bind_inner_product_statement(&fixture.pcs, &fixture.root.0, &fixture.claim, &mut prover);
    prove(
        &fixture.pcs,
        &fixture.data,
        fixture.witness.clone(),
        &OpeningQuery::InnerProduct {
            claim: fixture.claim.clone(),
        },
        StatementBinding::AlreadyBound,
        &mut prover,
    )
    .unwrap();
    let proof = prover.finish();

    // Independent composition checks stage order and binding of the derived MLE claim.
    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    bind_inner_product_statement(&fixture.pcs, &fixture.root.0, &fixture.claim, &mut verifier);
    verifier.public_message(SUMCHECK_LABEL);
    let reduced = post_gkr::verify(&fixture.claim, &mut verifier).unwrap();
    verify(
        &fixture.pcs,
        &fixture.root,
        &OpeningQuery::Mle {
            point: reduced.point,
            target: reduced.target,
        },
        StatementBinding::Bind,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();
}

#[test]
fn fixed_mle_claim_arrays_keep_the_existing_domain_and_encoding() {
    let claims = core::array::from_fn(|index| FlockF128::new(index as u64, 0));
    let mut prover = build_prover(SESSION, INSTANCE);
    write_claims(&mut prover, &claims);
    let challenge = prover.verifier_message::<F128>();
    let proof = prover.finish();
    assert_eq!(proof.narg_string.len(), CLAIM_COUNT * 16);

    let mut expected = build_prover(SESSION, INSTANCE);
    expected.public_message(b"bitz/pcs/mle-claims/v1" as &[u8]);
    expected.prover_message(&claims.map(from_flock_f128));
    assert_eq!(expected.verifier_message::<F128>(), challenge);

    let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
    assert_eq!(read_claims(&mut verifier).unwrap(), claims);
    assert_eq!(verifier.verifier_message::<F128>(), challenge);
    verifier.check_eof().unwrap();
}

proptest! {
    #[test]
    fn mle_statement_binding_matches_between_roles(
        root in any::<[u8; 32]>(),
        point_words in prop::collection::vec((any::<u64>(), any::<u64>()), 0..32),
        target_words in (any::<u64>(), any::<u64>()),
    ) {
        let fixture = fixture();
        let point = point_words.iter().map(|&(lo, hi)| F128::new(lo, hi)).collect::<Vec<_>>();
        let target = F128::new(target_words.0, target_words.1);
        let mut prover = build_prover(SESSION, INSTANCE);
        bind_mle_statement(&fixture.pcs, &root, &point, target, &mut prover);
        let expected = prover.verifier_message::<F128>();
        let proof = prover.finish();
        let mut verifier = build_verifier(SESSION, INSTANCE, &proof);
        bind_mle_statement(&fixture.pcs, &root, &point, target, &mut verifier);
        prop_assert_eq!(verifier.verifier_message::<F128>(), expected);
        prop_assert!(verifier.check_eof().is_ok());
    }
}
