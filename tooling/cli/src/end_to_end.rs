//! Circuit constraints over Q100, reduced by Spartan and opened through virtual BitZ.

use circuit::{
    Circuit,
    constraints::ConstraintGenerator,
    matrix_transpose::{MTransposeGenerator, MaterializedMTranspose},
    witgen::{PackedWitness, ProductWitgen},
};
use common::{BitZParams, LinearClaim, Root, Shape, VirtualMap, VirtualStatement};
use field::{F128, FqDefault, Q100, gf128::smallest_generator};
use num_traits::ConstZero;
use pcs::{HashKind, LigeritoProfile, Pcs, ProverData};
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
    #[error("BitZ proving failed: {0:?}")]
    Prove(prover::ProveError),
    #[error("BitZ verification failed: {0:?}")]
    Verify(verifier::VerifyError),
}

#[derive(Debug)]
pub struct Prepared<S> {
    statement: S,
    matrices: PreparedConstraintMatrices<FqDefault>,
    map: MaterializedMTranspose,
    params: BitZParams<Q100>,
    committed_shape: Shape,
    pcs: Pcs,
}

#[derive(Debug)]
pub struct Witness {
    committed: Vec<F128>,
    virtual_bits: Vec<F128>,
    assignment: DenseMultilinearExtension<FqDefault>,
    products: R1csProductMles<FqDefault>,
}

#[derive(Clone, Debug)]
pub struct Proof {
    pub root: Root,
    pub spartan: SpartanPiopProof<FqDefault>,
    pub opening: transcript::Proof,
}

impl<S: CircuitStatement> Prepared<S> {
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
        let claim_shape = shape_for(map.h_len())?;
        let committed_shape = shape_for(map.f_len() - 1)?;
        let params = BitZParams::new(claim_shape, smallest_generator())
            .map_err(|_| Error::Configuration("inadmissible BitZ parameters"))?;
        let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3)
            .map_err(|_| Error::Configuration("unsupported PCS shape"))?;
        Ok(Self {
            statement,
            matrices,
            map,
            params,
            committed_shape,
            pcs,
        })
    }

    pub fn witness(&self, inputs: &[bool]) -> Result<Witness, Error> {
        if inputs.len() != self.statement.input_bits() {
            return Err(Error::Input("wrong witness input length"));
        }
        let mut generator = ProductWitgen::with_inputs(inputs);
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
        Ok(Witness {
            committed: pack(&f, self.committed_shape),
            virtual_bits: pack(&h, *self.params.shape()),
            assignment,
            products,
        })
    }

    pub fn commit(&self, witness: &Witness) -> Result<ProverData, Error> {
        self.pcs
            .commit(&witness.committed)
            .map(|(_, data)| data)
            .map_err(Error::Commit)
    }

    pub fn prove(&self, witness: Witness, data: &ProverData) -> Result<Proof, Error> {
        let root = data.root();
        let mut transcript = build_prover(SESSION, self.statement.domain());
        self.bind(&mut transcript, root);
        let (spartan, terminal) = prove_spartan_piop(
            &mut transcript,
            &self.matrices,
            &witness.products,
            &witness.assignment,
        )
        .map_err(Error::Spartan)?;
        let claim = opening_claim(&self.params, &terminal)?;
        let statement = VirtualStatement::new(self.params, self.committed_shape, &self.map, &claim)
            .map_err(|_| Error::Configuration("invalid virtual statement"))?;
        BitZProver::new(self.params, WINDOW)
            .prove_virtual(
                &statement,
                &self.pcs,
                data,
                VirtualWitness {
                    committed_bits: witness.committed,
                    virtual_bits: &witness.virtual_bits,
                },
                &mut transcript,
            )
            .map_err(Error::Prove)?;
        Ok(Proof {
            root,
            spartan,
            opening: transcript.finish(),
        })
    }

    pub fn verify(&self, proof: &Proof) -> Result<(), Error> {
        let mut transcript = build_verifier(SESSION, self.statement.domain(), &proof.opening);
        self.bind(&mut transcript, proof.root);
        let terminal = verify_spartan_proof(&mut transcript, &self.matrices, &proof.spartan)
            .map_err(Error::Spartan)?;
        let claim = opening_claim(&self.params, &terminal)?;
        let statement = VirtualStatement::new(self.params, self.committed_shape, &self.map, &claim)
            .map_err(|_| Error::Configuration("invalid virtual statement"))?;
        BitZVerifier::new(self.params, WINDOW)
            .verify_virtual(&statement, &self.pcs, proof.root, transcript)
            .map_err(Error::Verify)
    }

    fn bind(&self, transcript: &mut impl PublicTranscript, root: Root) {
        let public = self.statement.public_bytes();
        transcript.public_message(&(public.len() as u64));
        transcript.public_message(public.as_slice());
        transcript.public_message(&root.0);
        transcript.public_message(&self.params);
        transcript.public_message(&self.pcs);
        transcript.public_message(&self.map.digest());
    }
}

/// The shape of a bit vector of `bits` entries: padded to a power of two, no
/// smaller than the protocol's minimum, split by [`Shape::for_log_bits`] —
/// the reference split (`t = ⌈0.6·n⌉` rows) that the BitZ measurements and
/// the f2z-pcs parity prover use, so an end-to-end proof exercises the same
/// PCS shapes as a direct one. Both the claim shape (over `h`) and the
/// committed shape (over `f`) come from here, each from its own length.
fn shape_for(bits: usize) -> Result<Shape, Error> {
    let padded = bits
        .checked_next_power_of_two()
        .ok_or(Error::Configuration("witness too large"))?;
    let log_bits = (padded.ilog2() as usize).max(common::shape::MIN_LOG_BITS);
    Shape::for_log_bits(log_bits)
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
    fn shapes_follow_the_reference_split() {
        // Below the minimum everything lands on the smallest shape, (14, 8).
        assert_eq!(shape_for(1).unwrap(), Shape::new(14, 8).unwrap());
        assert_eq!(shape_for(20_457).unwrap(), Shape::new(14, 8).unwrap());
        assert_eq!(shape_for(1 << 22).unwrap(), Shape::new(14, 8).unwrap());
        // Above it the split follows the bit count: 2^22 + 1 pads to 2^23.
        assert_eq!(shape_for((1 << 22) + 1).unwrap(), Shape::for_log_bits(23).unwrap());
        assert_eq!(shape_for(12_000_000).unwrap(), Shape::for_log_bits(24).unwrap());
        assert_eq!(Shape::for_log_bits(24).unwrap(), Shape::new(15, 9).unwrap());
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
