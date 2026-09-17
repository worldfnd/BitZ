//! Circuit constraints over Q100, reduced by Spartan and opened through direct or virtual BitZ.

use circuit::{
    Circuit,
    constraints::{ConstraintGenerator, SparseBoolMatrix},
    matrix_transpose::{MTransposeGenerator, MaterializedMTranspose},
    witgen::{PackedWitness, ProductWitgen},
};
use common::{
    BitZParams, LinearClaim, OpeningQuery, Root, Shape, VirtualMap, VirtualStatement,
    shape::{MIN_LOG_BITS, PACK_BITS},
};
use field::{F128, FqDefault, Q100, gf128::smallest_generator};
use num_traits::{ConstOne, ConstZero};
use pcs::{CommitScheme, HashKind, LigeritoProfile, Pcs, ProverData, StatementBinding};
use poly::{DenseMultilinearExtension, ScaledMleEvaluationClaim};
use prover::{BitZProver, VirtualWitness};
use transcript::{PublicTranscript, build_prover, build_verifier};
use verifier::BitZVerifier;

use spartan::{
    PreparedConstraintMatrices, R1csProductMles, SpartanPiopProof, bigint_to_fq,
    build_assignment_mle, build_product_mles, prove_spartan_piop, verify_spartan_proof,
};

const SESSION: &[u8] = b"bitz/circuit-e2e/v1";
const WINDOW: u32 = 8;

/// A trusted, deterministic circuit and its public inputs. Implementations must
/// emit identical operations for symbolic and concrete backends and constrain
/// every public input/output. The proved constraints are interpreted modulo Q100.
pub trait CircuitStatement {
    fn domain(&self) -> &'static [u8];
    fn public_bytes(&self) -> Vec<u8>;
    fn input_bits(&self) -> usize;
    fn synthesize<C: Circuit>(&self, cs: &mut C, inputs: &[C::Bool]) -> Result<(), Error>;
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid circuit input: {0}")]
    Input(&'static str),
    #[error("invalid proof configuration: {0}")]
    Configuration(&'static str),
    #[error("matrix preparation failed: {0:?}")]
    Matrix(spartan::SpartanMatrixError),
    #[error("circuit witness does not satisfy the constraints")]
    Unsatisfied,
    #[error("Spartan failed: {0:?}")]
    Spartan(spartan::SpartanError),
    #[error("commitment failed: {0:?}")]
    Commit(pcs::CommitError),
    #[error("constant-one opening failed: {0:?}")]
    ConstantProve(pcs::ProveError),
    #[error("constant-one verification failed: {0:?}")]
    ConstantVerify(pcs::VerifyError),
    #[error("BitZ proving failed: {0:?}")]
    Prove(prover::ProveError),
    #[error("BitZ verification failed: {0:?}")]
    Verify(verifier::VerifyError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OpeningPath {
    Direct = 0,
    Virtual = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitStats {
    pub opening_path: OpeningPath,
    pub constraints: usize,
    pub assignment_bits: usize,
    /// Meaningful committed witness bits before PCS zero padding.
    pub committed_bits: usize,
    pub padded_committed_bits: usize,
}

#[derive(Debug)]
pub struct CircuitProofSystem<S> {
    statement: S,
    opening_path: OpeningPath,
    matrices: PreparedConstraintMatrices<FqDefault>,
    map: MaterializedMTranspose,
    params: BitZParams<Q100>,
    committed_shape: Shape,
    pcs: Pcs,
}

#[derive(Debug)]
pub struct Witness {
    committed: Vec<F128>,
    assignment_bits: Vec<F128>,
    assignment: DenseMultilinearExtension<FqDefault>,
    products: R1csProductMles<FqDefault>,
}

#[derive(Clone, Debug)]
pub struct Proof {
    pub root: Root,
    pub spartan: SpartanPiopProof<FqDefault>,
    pub opening: transcript::Proof,
}

impl<S: CircuitStatement> CircuitProofSystem<S> {
    #[tracing::instrument(name = "setup", skip_all)]
    pub fn new(statement: S) -> Result<Self, Error> {
        let mut constraints = ConstraintGenerator::new(statement.input_bits());
        let inputs: Vec<_> = (0..statement.input_bits())
            .map(|i| constraints.input(i))
            .collect();
        statement.synthesize(&mut constraints, &inputs)?;
        let matrices = PreparedConstraintMatrices::new(
            constraints
                .into_matrices()
                .map_coefficients(|c| bigint_to_fq(&c)),
        )
        .map_err(Error::Matrix)?;
        let mut generator = MTransposeGenerator::new(statement.input_bits());
        let inputs = generator.take_inputs();
        statement.synthesize(&mut generator, &inputs)?;
        let map = generator.finish();
        if map.h_len() != matrices.matrices().a.column_count() {
            return Err(Error::Configuration("map and assignment dimensions differ"));
        }
        if map.f_len() != matrices.matrices().m.column_count() {
            return Err(Error::Configuration("map and witness dimensions differ"));
        }
        let claim_shape = shape_for(map.h_len())?;
        let opening_path = if is_identity(&matrices.matrices().m) {
            OpeningPath::Direct
        } else {
            OpeningPath::Virtual
        };
        let committed_shape = match opening_path {
            OpeningPath::Direct => claim_shape,
            OpeningPath::Virtual => shape_for(map.f_len() - 1)?,
        };
        let params = BitZParams::new(claim_shape, smallest_generator())
            .map_err(|_| Error::Configuration("inadmissible BitZ parameters"))?;
        let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3)
            .map_err(|_| Error::Configuration("unsupported PCS shape"))?;
        Ok(Self {
            statement,
            opening_path,
            matrices,
            map,
            params,
            committed_shape,
            pcs,
        })
    }

    pub fn stats(&self) -> CircuitStats {
        CircuitStats {
            opening_path: self.opening_path,
            constraints: self.matrices.matrices().a.row_count(),
            assignment_bits: self.map.h_len(),
            committed_bits: match self.opening_path {
                OpeningPath::Direct => self.map.h_len(),
                OpeningPath::Virtual => self.map.f_len() - 1,
            },
            padded_committed_bits: self.pcs.bit_len(),
        }
    }

    #[tracing::instrument(name = "witness", skip_all)]
    pub fn witness(&self, inputs: &[bool]) -> Result<Witness, Error> {
        if inputs.len() != self.statement.input_bits() {
            return Err(Error::Input("wrong witness input length"));
        }
        let mut generator = ProductWitgen::with_inputs_and_capacity(inputs, self.map.f_len() - 1);
        self.statement.synthesize(&mut generator, inputs)?;
        let (f, h, products) = generator.into_parts();
        if f.bit_len() + 1 != self.map.f_len() || h.bit_len() != self.map.h_len() {
            return Err(Error::Input("circuit replay changed witness dimensions"));
        }
        let products = build_product_mles(&products, self.matrices.matrices().a.row_count())
            .map_err(Error::Matrix)?;
        if products
            .az
            .iter()
            .zip(products.bz.iter())
            .zip(products.cz.iter())
            .any(|((&a, &b), &c)| a * b != c)
        {
            return Err(Error::Unsatisfied);
        }
        let assignment = build_assignment_mle(&h, self.map.h_len()).map_err(Error::Matrix)?;
        let assignment_bits = pack(&h, *self.params.shape());
        let (committed, assignment_bits) = match self.opening_path {
            OpeningPath::Direct => (assignment_bits, Vec::new()),
            OpeningPath::Virtual => (pack(&f, self.committed_shape), assignment_bits),
        };
        Ok(Witness {
            committed,
            assignment_bits,
            assignment,
            products,
        })
    }

    #[tracing::instrument(name = "commit", skip_all)]
    pub fn commit(&self, witness: &Witness) -> Result<ProverData, Error> {
        self.pcs
            .commit(&witness.committed)
            .map(|(_, data)| data)
            .map_err(Error::Commit)
    }

    #[tracing::instrument(name = "prove", skip_all, fields(opening_path = ?self.opening_path))]
    pub fn prove(&self, witness: Witness, data: &ProverData) -> Result<Proof, Error> {
        let root = data.root();
        let mut transcript = build_prover(SESSION, self.statement.domain());
        self.bind(&mut transcript, root);
        if self.opening_path == OpeningPath::Direct {
            self.pcs
                .prove_lin(
                    data,
                    witness.committed.clone(),
                    &self.constant_query(),
                    StatementBinding::Bind,
                    &mut transcript,
                )
                .map_err(Error::ConstantProve)?;
        }
        let (spartan, terminal) = prove_spartan_piop(
            &mut transcript,
            &self.matrices,
            &witness.products,
            &witness.assignment,
        )
        .map_err(Error::Spartan)?;
        let claim = opening_claim(&self.params, &terminal)?;
        let prover = BitZProver::new(self.params, WINDOW);
        match self.opening_path {
            OpeningPath::Direct => {
                prover.prove(&claim, &self.pcs, data, witness.committed, &mut transcript)
            }
            OpeningPath::Virtual => {
                let statement =
                    VirtualStatement::new(self.params, self.committed_shape, &self.map, &claim)
                        .map_err(|_| Error::Configuration("invalid virtual statement"))?;
                prover.prove_virtual(
                    &statement,
                    &self.pcs,
                    data,
                    VirtualWitness {
                        committed_bits: witness.committed,
                        virtual_bits: &witness.assignment_bits,
                    },
                    &mut transcript,
                )
            }
        }
        .map_err(Error::Prove)?;
        Ok(Proof {
            root,
            spartan,
            opening: transcript.finish(),
        })
    }

    #[tracing::instrument(name = "verify", skip_all)]
    pub fn verify(&self, proof: &Proof) -> Result<(), Error> {
        let mut transcript = build_verifier(SESSION, self.statement.domain(), &proof.opening);
        self.bind(&mut transcript, proof.root);
        if self.opening_path == OpeningPath::Direct {
            self.pcs
                .verify_lin(
                    &proof.root,
                    &self.constant_query(),
                    StatementBinding::Bind,
                    &mut transcript,
                )
                .map_err(Error::ConstantVerify)?;
        }
        let terminal = verify_spartan_proof(&mut transcript, &self.matrices, &proof.spartan)
            .map_err(Error::Spartan)?;
        let claim = opening_claim(&self.params, &terminal)?;
        let verifier = BitZVerifier::new(self.params, WINDOW);
        match self.opening_path {
            OpeningPath::Direct => verifier.verify(&claim, &self.pcs, proof.root, transcript),
            OpeningPath::Virtual => {
                let statement =
                    VirtualStatement::new(self.params, self.committed_shape, &self.map, &claim)
                        .map_err(|_| Error::Configuration("invalid virtual statement"))?;
                verifier.verify_virtual(&statement, &self.pcs, proof.root, transcript)
            }
        }
        .map_err(Error::Verify)
    }

    // Direct commitment includes h[0]; unlike the virtual map, it does not
    // supply that coordinate as a fixed one. Opening at zero enforces h[0] = 1.
    fn constant_query(&self) -> OpeningQuery {
        OpeningQuery::Mle {
            point: vec![F128::ZERO; self.committed_shape.log_bits()],
            target: F128::ONE,
        }
    }

    fn bind(&self, transcript: &mut impl PublicTranscript, root: Root) {
        let public = self.statement.public_bytes();
        transcript.public_message(&(public.len() as u64));
        transcript.public_message(public.as_slice());
        transcript.public_message(&root.0);
        transcript.public_message(&self.params);
        transcript.public_message(&self.pcs);
        transcript.public_message(&self.map.digest());
        transcript.public_message(b"bitz/circuit-opening-path/v1");
        transcript.public_message(&[self.opening_path as u8]);
    }
}

fn is_identity(map: &SparseBoolMatrix) -> bool {
    map.row_count() == map.column_count()
        && map
            .rows()
            .iter()
            .enumerate()
            .all(|(i, row)| row.positions() == [i])
}

fn shape_for(bits: usize) -> Result<Shape, Error> {
    let padded = bits
        .checked_next_power_of_two()
        .ok_or(Error::Configuration("witness too large"))?;
    let log_bits = (padded.ilog2() as usize).max(MIN_LOG_BITS);
    let log_rows = PACK_BITS as usize;
    Shape::new(log_rows, log_bits - log_rows)
        .map_err(|_| Error::Configuration("witness shape outside supported range"))
}

fn pack(witness: &PackedWitness, shape: Shape) -> Vec<F128> {
    let mut packed: Vec<_> = witness
        .words()
        .chunks(2)
        .map(|words| F128::new(words[0], words.get(1).copied().unwrap_or(0)))
        .collect();
    packed.resize(1 << shape.log_packed_len(), F128::ZERO);
    packed
}

fn opening_claim(
    params: &BitZParams<Q100>,
    terminal: &ScaledMleEvaluationClaim<FqDefault>,
) -> Result<LinearClaim<FqDefault>, Error> {
    let shape = params.shape();
    if terminal.point().len() > shape.log_bits() {
        return Err(Error::Configuration("Spartan point exceeds virtual shape"));
    }
    // Zero high coordinates select the original assignment inside its zero padding.
    // Put the scale in one factor, avoiding division even when the scale is zero.
    let mut point = terminal.point().to_vec();
    point.resize(shape.log_bits(), FqDefault::ZERO);
    let rows = poly::eq_table(&point[..shape.log_rows()])
        .into_iter()
        .map(|weight| terminal.scale() * weight)
        .collect();
    let columns = poly::eq_table(&point[shape.log_rows()..]);
    LinearClaim::new(params, rows, columns, terminal.value())
        .map_err(|_| Error::Configuration("invalid terminal claim dimensions"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_exact_identity_maps_select_direct_opening() {
        let identity = SparseBoolMatrix::try_from_rows(3, vec![vec![0], vec![1], vec![2]]).unwrap();
        assert!(is_identity(&identity));
        for rows in [
            vec![vec![0], vec![2], vec![1]],
            vec![vec![0], vec![1], vec![1, 2]],
            vec![vec![0], vec![1], vec![0, 2]],
            vec![vec![0], vec![1], vec![1]],
            vec![vec![0], vec![1]],
        ] {
            assert!(!is_identity(
                &SparseBoolMatrix::try_from_rows(3, rows).unwrap()
            ));
        }
    }

    struct IdentityBit;

    impl CircuitStatement for IdentityBit {
        fn domain(&self) -> &'static [u8] {
            b"test/identity-bit/v1"
        }
        fn public_bytes(&self) -> Vec<u8> {
            vec![1]
        }
        fn input_bits(&self) -> usize {
            1
        }
        fn synthesize<C: Circuit>(&self, cs: &mut C, inputs: &[C::Bool]) -> Result<(), Error> {
            let bit = cs.bitz::<1>(inputs[0].clone());
            let one = C::Z::<1>::from(C::Coefficient::<1>::from(1u64));
            cs.assert_r1c::<1>(one.clone(), bit, one);
            Ok(())
        }
    }

    #[test]
    fn direct_opening_requires_constant_one_on_both_sides() {
        let mut system = CircuitProofSystem::new(IdentityBit).unwrap();
        let witness = system.witness(&[true]).unwrap();
        let data = system.commit(&witness).unwrap();
        let mut proof = system.prove(witness, &data).unwrap();
        system.verify(&proof).unwrap();
        system.opening_path = OpeningPath::Virtual;
        assert!(system.verify(&proof).is_err());
        system.opening_path = OpeningPath::Direct;

        let mut bad_witness = system.witness(&[true]).unwrap();
        bad_witness.committed.fill(F128::ZERO);
        let bad_data = system.commit(&bad_witness).unwrap();
        assert!(matches!(
            system.prove(bad_witness, &bad_data),
            Err(Error::ConstantProve(_))
        ));

        // A valid opening to zero must not substitute for the required one.
        let packed = vec![F128::ZERO; 1 << system.committed_shape.log_packed_len()];
        let mut transcript = build_prover(SESSION, system.statement.domain());
        system.bind(&mut transcript, bad_data.root());
        let query = OpeningQuery::Mle {
            point: vec![F128::ZERO; system.committed_shape.log_bits()],
            target: F128::ZERO,
        };
        system
            .pcs
            .prove_lin(
                &bad_data,
                packed,
                &query,
                StatementBinding::Bind,
                &mut transcript,
            )
            .unwrap();
        proof.root = bad_data.root();
        proof.opening = transcript.finish();
        assert!(matches!(
            system.verify(&proof),
            Err(Error::ConstantVerify(_))
        ));
    }

    #[test]
    fn scaled_claim_conversion_preserves_values_and_zero_scale() {
        let params = BitZParams::new(Shape::new(7, 15).unwrap(), smallest_generator()).unwrap();
        let assignment = DenseMultilinearExtension::from_evaluations(
            2,
            [0u128, 1, 1, 0].map(FqDefault::from).to_vec(),
        )
        .unwrap();
        let point = vec![FqDefault::from(3u128), FqDefault::from(5u128)];
        let evaluation = assignment.evaluate(&point).unwrap();
        for scale in [FqDefault::ZERO, FqDefault::from(7u128)] {
            let terminal = ScaledMleEvaluationClaim::new(
                point.clone().into_boxed_slice(),
                scale,
                scale * evaluation,
            );
            let claim = opening_claim(&params, &terminal).unwrap();
            let value: FqDefault = assignment
                .iter()
                .enumerate()
                .map(|(i, bit)| *bit * claim.row_weights()[i] * claim.column_weights()[0])
                .sum();
            assert_eq!(value, claim.target());
            assert!(
                claim.row_weights()[4..]
                    .iter()
                    .all(|w| *w == FqDefault::ZERO)
            );
            assert!(
                claim.column_weights()[1..]
                    .iter()
                    .all(|w| *w == FqDefault::ZERO)
            );
        }
    }
}
