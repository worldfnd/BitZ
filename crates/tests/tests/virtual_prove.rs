//! Virtual prove/verify through GKR, matrix transposition, and the real PCS.

use circuit::{
    matrix_transpose::MTransposeGenerator,
    sha256::{ABC_BLOCK, ABC_DIGEST, COMPRESSION_INPUT_BITS, INITIAL_STATE, compression_circuit},
    witgen::{PackedWitness, Witgen},
};
use common::{
    BitZParams, LinearClaim, Root, Shape, TableError, TransposedWeights, VirtualMap,
    VirtualMapError, VirtualStatement,
};
use field::{F128, Fq, gf128::smallest_generator};
use num_traits::{ConstOne, ConstZero};
use pcs::{HashKind, LigeritoProfile, Pcs, ProverData};
use prover::{BitZProver, ProveError, VirtualWitness};
use tests::{Q, WINDOW, prover_transcript, verifier_transcript};
use transcript::Proof;
use verifier::{BitZVerifier, VerifyError};

/// `h[0] = 1`, `h[1] = f[0]`, `h[128] = f[1]`, `h[129] = f[0] XOR f[1]`.
/// All other virtual bits are zero.
struct Map(u8);

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
        // The tag lets replay tests change only the public map identifier.
        [self.0; 32]
    }

    fn h_len(&self) -> usize {
        130
    }
    fn f_len(&self) -> usize {
        3
    }
}

struct Instance {
    params: BitZParams<Q>,
    committed_shape: Shape,
    claim: LinearClaim<Fq<Q>>,
    committed_bits: Vec<F128>,
    virtual_bits: Vec<F128>,
    pcs: Pcs,
    root: Root,
    data: ProverData,
}

impl Instance {
    fn new() -> Self {
        let claim_shape = Shape::new(7, 15).unwrap();
        let committed_shape = Shape::new(8, 14).unwrap();
        let params = BitZParams::<Q>::new(claim_shape, smallest_generator()).unwrap();
        let claim = LinearClaim::new(
            &params,
            vec![Fq::ONE; claim_shape.rows()],
            vec![Fq::ONE; claim_shape.columns()],
            Fq::from(3u128),
        )
        .unwrap();
        let mut committed_bits = vec![F128::ZERO; 1 << committed_shape.log_packed_len()];
        committed_bits[0] = F128::from(1u64);
        let mut virtual_bits = vec![F128::ZERO; 1 << claim_shape.log_packed_len()];
        virtual_bits[0] = F128::from(3u64);
        virtual_bits[1] = F128::from(2u64);

        let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
        let (root, data) = pcs.commit(&committed_bits).unwrap();
        Self {
            params,
            committed_shape,
            claim,
            committed_bits,
            virtual_bits,
            pcs,
            root,
            data,
        }
    }

    fn statement(&self) -> VirtualStatement<'_, Q, Map> {
        VirtualStatement::new(self.params, self.committed_shape, &Map(7), &self.claim).unwrap()
    }

    fn prove(&self) -> Proof {
        let mut transcript = prover_transcript();
        let (_, data) = self
            .pcs
            .commit_with_ood(&self.committed_bits, &mut transcript)
            .unwrap();
        BitZProver::new(self.params, WINDOW)
            .prove_virtual(
                &self.statement(),
                &self.pcs,
                &data,
                VirtualWitness {
                    committed_bits: self.committed_bits.clone(),
                    virtual_bits: &self.virtual_bits,
                },
                &mut transcript,
            )
            .unwrap();
        transcript.finish()
    }

    fn verify(
        &self,
        statement: &VirtualStatement<'_, Q, Map>,
        root: Root,
        proof: &Proof,
    ) -> Result<(), VerifyError> {
        BitZVerifier::new(self.params, WINDOW).verify_virtual(
            statement,
            &self.pcs,
            root,
            verifier_transcript(proof),
        )
    }
}

#[test]
fn virtual_inner_product_opens_the_committed_bits() {
    let instance = Instance::new();
    let proof = instance.prove();
    instance
        .verify(&instance.statement(), instance.root, &proof)
        .unwrap();
}

#[test]
fn changed_virtual_statements_are_rejected() {
    let instance = Instance::new();
    let statement = instance.statement();
    let proof = instance.prove();
    instance.verify(&statement, instance.root, &proof).unwrap();

    let mut changed_root = instance.root;
    changed_root.0[0] ^= 1;
    assert!(instance.verify(&statement, changed_root, &proof).is_err());

    let changed_map = VirtualStatement::new(
        instance.params,
        instance.committed_shape,
        &Map(8),
        &instance.claim,
    )
    .unwrap();
    assert!(
        instance
            .verify(&changed_map, instance.root, &proof)
            .is_err()
    );

    // Changing a weight on a zero virtual row preserves the integer target.
    // Rejection must therefore depend on the statement, not a false claim.
    let mut rows = instance.claim.row_weights().to_vec();
    rows[2] += Fq::ONE;
    let changed_claim = LinearClaim::new(
        &instance.params,
        rows,
        instance.claim.column_weights().to_vec(),
        instance.claim.target(),
    )
    .unwrap();
    let changed_statement = VirtualStatement::new(
        instance.params,
        instance.committed_shape,
        &Map(7),
        &changed_claim,
    )
    .unwrap();
    assert!(
        instance
            .verify(&changed_statement, instance.root, &proof)
            .is_err()
    );
}

#[test]
fn malformed_virtual_proofs_are_rejected() {
    let instance = Instance::new();
    let proof = instance.prove();
    let verify = |proof: &Proof| instance.verify(&instance.statement(), instance.root, proof);
    verify(&proof).unwrap();

    for hints in [false, true] {
        let mut changed = proof.clone();
        let bytes = if hints {
            &mut changed.hints
        } else {
            &mut changed.narg_string
        };
        bytes.push(0);
        assert_eq!(verify(&changed), Err(VerifyError::TrailingData));

        let mut changed = proof.clone();
        let bytes = if hints {
            &mut changed.hints
        } else {
            &mut changed.narg_string
        };
        assert!(bytes.pop().is_some());
        assert!(verify(&changed).is_err());

        let mut changed = proof.clone();
        let bytes = if hints {
            &mut changed.hints
        } else {
            &mut changed.narg_string
        };
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xff;
        assert!(verify(&changed).is_err());
    }
}

#[test]
fn virtual_witness_lengths_and_setup_must_match_the_statement() {
    let instance = Instance::new();
    let statement = instance.statement();
    let prover = BitZProver::new(instance.params, WINDOW);
    for committed in [false, true] {
        let mut committed_bits = instance.committed_bits.clone();
        let mut virtual_bits = instance.virtual_bits.clone();
        if committed {
            committed_bits.pop();
        } else {
            virtual_bits.pop();
        }
        let mut transcript = prover_transcript();
        assert_eq!(
            prover.prove_virtual(
                &statement,
                &instance.pcs,
                &instance.data,
                VirtualWitness {
                    committed_bits,
                    virtual_bits: &virtual_bits
                },
                &mut transcript,
            ),
            Err(ProveError::Witness(TableError::BitCountMismatch))
        );
        assert_eq!(transcript.finish(), Proof::default());
    }

    let other_params = BitZParams::new(instance.committed_shape, smallest_generator()).unwrap();
    let larger_commitment = VirtualStatement::new(
        instance.params,
        Shape::new(13, 15).unwrap(),
        &Map(7),
        &instance.claim,
    )
    .unwrap();
    for (params, statement) in [
        (other_params, statement),
        (instance.params, larger_commitment),
    ] {
        let mut transcript = prover_transcript();
        assert_eq!(
            BitZProver::new(params, WINDOW).prove_virtual(
                &statement,
                &instance.pcs,
                &instance.data,
                VirtualWitness {
                    committed_bits: instance.committed_bits.clone(),
                    virtual_bits: &instance.virtual_bits,
                },
                &mut transcript,
            ),
            Err(ProveError::ParameterMismatch)
        );
        assert_eq!(transcript.finish(), Proof::default());
        assert_eq!(
            BitZVerifier::new(params, WINDOW).verify_virtual(
                &statement,
                &instance.pcs,
                instance.root,
                verifier_transcript(&Proof::default()),
            ),
            Err(VerifyError::ParameterMismatch)
        );
    }
}

#[test]
fn virtual_bits_inconsistent_with_the_map_cannot_be_opened() {
    let instance = Instance::new();
    let mut virtual_bits = instance.virtual_bits.clone();
    // Move the constant-one bit to row two. The integer sum stays three,
    // but the virtual witness no longer equals M (1 || f).
    virtual_bits[0] = F128::from(6u64);
    let mut transcript = prover_transcript();
    let (_, data) = instance
        .pcs
        .commit_with_ood(&instance.committed_bits, &mut transcript)
        .unwrap();
    assert_eq!(
        BitZProver::new(instance.params, WINDOW).prove_virtual(
            &instance.statement(),
            &instance.pcs,
            &data,
            VirtualWitness {
                committed_bits: instance.committed_bits.clone(),
                virtual_bits: &virtual_bits,
            },
            &mut transcript,
        ),
        Err(ProveError::Opening(pcs::ProveError::InvalidClaim))
    );
}

#[test]
fn sha256_virtual_inner_product_opens_the_committed_bits() {
    let inputs = std::array::from_fn::<_, COMPRESSION_INPUT_BITS, _>(|index| {
        let word = if index < 512 {
            ABC_BLOCK[index / 32]
        } else {
            INITIAL_STATE[(index - 512) / 32]
        };
        (word >> (index % 32)) & 1 != 0
    });
    let mut materializer = MTransposeGenerator::new(inputs.len());
    let matrix_inputs = materializer.take_boxed_inputs();
    compression_circuit(&mut materializer, &matrix_inputs);
    let map = materializer.finish();
    let mut witgen = Witgen::with_inputs(&inputs);
    let output = compression_circuit(&mut witgen, &inputs);
    for (index, bit) in output.into_iter().enumerate() {
        assert_eq!(bit, (ABC_DIGEST[index / 32] >> (index % 32)) & 1 != 0);
    }
    let (f, h) = witgen.into_witnesses();
    assert_eq!(map.f_len(), f.bit_len() + 1);
    assert_eq!(map.h_len(), h.bit_len());

    let claim_shape = Shape::new(7, 15).unwrap();
    let committed_shape = Shape::new(8, 14).unwrap();
    let params = BitZParams::<Q>::new(claim_shape, smallest_generator()).unwrap();
    let pack = |witness: &PackedWitness, shape: Shape| {
        assert!(witness.bit_len() <= 1 << shape.log_bits());
        let mut packed: Vec<_> = witness
            .words()
            .chunks(2)
            .map(|words| F128::new(words[0], words.get(1).copied().unwrap_or(0)))
            .collect();
        packed.resize(1 << shape.log_packed_len(), F128::ZERO);
        packed
    };
    let committed_bits = pack(&f, committed_shape);
    let virtual_bits = pack(&h, claim_shape);
    let rows: Vec<_> = (0..claim_shape.rows())
        .map(|row| Fq::<Q>::from((row + 1) as u128))
        .collect();
    let columns: Vec<_> = (0..claim_shape.columns())
        .map(|column| Fq::<Q>::from((column + 1) as u128))
        .collect();
    let target = (0..h.bit_len())
        .filter(|&index| h.bit(index))
        .map(|index| rows[index % claim_shape.rows()] * columns[index / claim_shape.rows()])
        .sum();
    let claim = LinearClaim::new(&params, rows, columns, target).unwrap();
    let statement = VirtualStatement::new(params, committed_shape, &map, &claim).unwrap();
    let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3).unwrap();
    let mut transcript = prover_transcript();
    let (root, data) = pcs
        .commit_with_ood(&committed_bits, &mut transcript)
        .unwrap();
    BitZProver::new(params, WINDOW)
        .prove_virtual(
            &statement,
            &pcs,
            &data,
            VirtualWitness {
                committed_bits,
                virtual_bits: &virtual_bits,
            },
            &mut transcript,
        )
        .unwrap();
    BitZVerifier::new(params, WINDOW)
        .verify_virtual(
            &statement,
            &pcs,
            root,
            verifier_transcript(&transcript.finish()),
        )
        .unwrap();
}
