//! The virtual path: a claim about `h`, an oracle holding `f`.

use common::{F2ZParams, LinearClaim, Shape, VirtualParams};
use field::{F128, Fq, gf128::smallest_generator};
use pcs::{HashKind, LigeritoProfile, Pcs};
use poly::{ScaledMleEvaluationClaim, eq_table};
use prover::VirtualProver;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use reduction::GrandProduct;
use tests::{Q, SelectionMap, WINDOW, pack_bits, prover_transcript, verifier_transcript};
use verifier::VirtualVerifier;

/// The inner-product opening is only defined for the secure profile, and the
/// virtual path always reaches it: `M^T v` has no point form.
const PROFILE: LigeritoProfile = LigeritoProfile::Secure;

/// Both shapes at the admissibility floor. `h` and `f` need not agree on
/// shape; here they do, which keeps the fixture readable.
fn shape() -> Shape {
    Shape::new(13, 9).unwrap()
}

#[test]
fn a_claim_about_the_integer_witness_opens_against_the_committed_bits() {
    let shape = shape();
    let bits = 1usize << shape.log_bits();
    let mut rng = ChaCha8Rng::seed_from_u64(0x21b);

    // The committed witness, and the integer witness the map derives from it.
    let committed_packed: Vec<F128> = (0..bits / 128)
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let params = F2ZParams::<Q>::new(shape, smallest_generator()).unwrap();
    let committed_table = params.table(&committed_packed).unwrap();

    let map = SelectionMap::strided(bits, bits);
    let integer_bits = map.integer_witness(&committed_table);
    let integer_packed = pack_bits(&integer_bits);

    // A claim about `h`, in the shape a PIOP hands over.
    let point: Vec<Fq<Q>> = (0..shape.log_bits())
        .map(|i| Fq::from(u128::from(rng.next_u64()) + i as u128))
        .collect();
    let (row_point, column_point) = point.split_at(shape.log_rows());
    let (eq_rows, eq_columns) = (eq_table(row_point), eq_table(column_point));
    let mut evaluation = Fq::from(0u128);
    for column in 0..shape.columns() {
        for row in 0..shape.rows() {
            if integer_bits[(column << shape.log_rows()) | row] {
                evaluation += eq_rows[row] * eq_columns[column];
            }
        }
    }
    let scale = Fq::from(0x9e37_79b9_7f4a_7c15u128);
    let claim = LinearClaim::from_evaluation(
        &params,
        &ScaledMleEvaluationClaim::new(point.into_boxed_slice(), scale, scale * evaluation),
    )
    .unwrap();

    // The oracle holds `f`, never `h`.
    let pcs = Pcs::new(&shape, PROFILE, HashKind::Blake3).unwrap();
    let (com, data) = pcs.commit(&committed_packed).unwrap();

    let virtual_params = VirtualParams::new(params, shape, &map).unwrap();
    let mut transcript = prover_transcript();
    VirtualProver::new(virtual_params, WINDOW)
        .prove(
            &map,
            &claim,
            &pcs,
            &data,
            &integer_packed,
            committed_packed,
            &GrandProduct,
            &mut transcript,
        )
        .expect("honest virtual instance");
    let proof = transcript.finish();

    VirtualVerifier::new(VirtualParams::new(params, shape, &map).unwrap(), WINDOW)
        .verify(
            &map,
            &claim,
            &pcs,
            com,
            &GrandProduct,
            verifier_transcript(&proof),
        )
        .expect("honest virtual proof");
}

/// `M` is public input, bound through its digest. Replaying the proof under a
/// different map is a different statement, so the challenge stream diverges.
#[test]
fn a_proof_does_not_verify_under_a_different_map() {
    let shape = shape();
    let bits = 1usize << shape.log_bits();
    let mut rng = ChaCha8Rng::seed_from_u64(0x22c);

    let committed_packed: Vec<F128> = (0..bits / 128)
        .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
        .collect();
    let params = F2ZParams::<Q>::new(shape, smallest_generator()).unwrap();
    let committed_table = params.table(&committed_packed).unwrap();

    let map = SelectionMap::strided(bits, bits);
    let integer_bits = map.integer_witness(&committed_table);
    let integer_packed = pack_bits(&integer_bits);

    let point: Vec<Fq<Q>> = (0..shape.log_bits())
        .map(|i| Fq::from(u128::from(rng.next_u64()) + i as u128))
        .collect();
    let (row_point, column_point) = point.split_at(shape.log_rows());
    let (eq_rows, eq_columns) = (eq_table(row_point), eq_table(column_point));
    let mut evaluation = Fq::from(0u128);
    for column in 0..shape.columns() {
        for row in 0..shape.rows() {
            if integer_bits[(column << shape.log_rows()) | row] {
                evaluation += eq_rows[row] * eq_columns[column];
            }
        }
    }
    let scale = Fq::from(3u128);
    let claim = LinearClaim::from_evaluation(
        &params,
        &ScaledMleEvaluationClaim::new(point.into_boxed_slice(), scale, scale * evaluation),
    )
    .unwrap();

    let pcs = Pcs::new(&shape, PROFILE, HashKind::Blake3).unwrap();
    let (com, data) = pcs.commit(&committed_packed).unwrap();

    let mut transcript = prover_transcript();
    VirtualProver::new(VirtualParams::new(params, shape, &map).unwrap(), WINDOW)
        .prove(
            &map,
            &claim,
            &pcs,
            &data,
            &integer_packed,
            committed_packed,
            &GrandProduct,
            &mut transcript,
        )
        .unwrap();
    let proof = transcript.finish();

    let other = SelectionMap::strided(bits, bits - 1);
    assert!(
        VirtualVerifier::new(VirtualParams::new(params, shape, &other).unwrap(), WINDOW)
            .verify(
                &other,
                &claim,
                &pcs,
                com,
                &GrandProduct,
                verifier_transcript(&proof),
            )
            .is_err()
    );
}
