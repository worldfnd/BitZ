//! Virtual prove/verify through GKR, matrix transposition, and the real PCS.

use common::{
    BitZParams, LinearClaim, Shape, TransposedWeights, VirtualMap, VirtualMapError,
    VirtualStatement,
};
use field::{F128, Fq, gf128::smallest_generator};
use pcs::{HashKind, LigeritoProfile, Pcs};
use prover::{BitZProver, VirtualWitness};
use tests::{Q, WINDOW, prover_transcript, verifier_transcript};
use verifier::BitZVerifier;

/// `h[0] = 1`, `h[1] = f[0]`, `h[128] = f[1]`, `h[129] = f[0] XOR f[1]`.
/// All other virtual bits are zero.
struct Map;

impl VirtualMap for Map {
    fn transpose(&self, weights: &[F128]) -> Result<TransposedWeights, VirtualMapError> {
        if weights.len() < self.h_len() {
            return Err(VirtualMapError::WeightCountMismatch);
        }
        Ok(TransposedWeights::new(
            vec![weights[1] + weights[129], weights[128] + weights[129]],
            weights[0],
        ))
    }

    fn digest(&self) -> [u8; 32] {
        // Both roles use this fixed map, so a constant identifier suffices.
        [7; 32]
    }

    fn h_len(&self) -> usize {
        130
    }
    fn f_len(&self) -> usize {
        3
    }
}

#[test]
fn virtual_inner_product_opens_the_committed_bits() {
    let claim_shape = Shape::new(7, 15).unwrap();
    let committed_shape = Shape::new(8, 14).unwrap();
    let params = BitZParams::<Q>::new(claim_shape, smallest_generator()).unwrap();
    let claim = LinearClaim::new(
        &params,
        vec![Fq::from(1u128); claim_shape.rows()],
        vec![Fq::from(1u128); claim_shape.columns()],
        Fq::from(3u128),
    )
    .unwrap();
    let statement = VirtualStatement::new(params, committed_shape, &Map, &claim).unwrap();
    let mut packed = vec![F128::default(); 1 << committed_shape.log_packed_len()];
    packed[0] = F128::from(1u64);
    let mut virtual_packed = vec![F128::default(); 1 << claim_shape.log_packed_len()];
    virtual_packed[0] = F128::from(3u64);
    virtual_packed[1] = F128::from(2u64);

    let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let (root, data) = pcs.commit(&packed).unwrap();
    let mut transcript = prover_transcript();
    BitZProver::new(params, WINDOW)
        .prove_virtual(
            &statement,
            &pcs,
            &data,
            VirtualWitness {
                committed_bits: packed,
                virtual_bits: &virtual_packed,
            },
            &mut transcript,
        )
        .unwrap();
    let proof = transcript.finish();
    BitZVerifier::new(params, WINDOW)
        .verify_virtual(&statement, &pcs, root, verifier_transcript(&proof))
        .unwrap();
}
