//! Circuit constraints over a prime field, reduced by Spartan and opened
//! through direct or virtual BitZ.
//!
//! The field's modulus is sampled at a runtime once the commitment and the
//! statement are absorbed.

use circuit::matrix_products::StoredInteger;
use circuit::{
    BitWidth, Circuit, IntoWords,
    constraints::{ConstraintGenerator, SparseBoolMatrix},
    matrix_products::IntegerProducts,
    matrix_transpose::{MTransposeGenerator, MaterializedMTranspose},
    witgen::{PackedWitness, ProductWitgen},
};
use common::{
    BitZParams, BitzClaimField, BitzConstraintRing, LinearClaim, OpeningQuery, Root, Shape,
    VirtualMap, VirtualStatement,
    shape::{MIN_LOG_BITS, PACK_BITS},
};
use field::{F128, FieldWithDynamicModulus, gf128::smallest_generator};
use num_traits::{ConstOne, ConstZero};
use pcs::{CommitScheme, HashKind, LigeritoProfile, Pcs, ProverData, StatementBinding};
use poly::ScaledMleEvaluationClaim;
use prover::{BitZProver, VirtualWitness};
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError};
use transcript::{
    ProverState, PublicTranscript, SqueezableTranscript, build_prover, build_verifier,
};
use verifier::BitZVerifier;

use crate::ProjectConstraint;
use spartan::{
    PreparedConstraintMatrices, PreparedIntegerMatrices, SpartanMatrixError, SpartanPiopProof,
    build_assignment_mle, build_product_mles, prove_spartan_piop, verify_spartan_proof,
};

const SESSION: &[u8] = b"bitz/circuit-e2e/v2";
const WINDOW: u32 = 8;

/// Held for the whole of a proof or a verification: the modulus they install
/// is process-wide.
static DYNAMIC_MODULUS_LOCK: Mutex<()> = Mutex::new(());

type ModulusLock<'a> = MutexGuard<'a, ()>;

/// A trusted, deterministic circuit and its public inputs. Implementations must
/// emit identical operations for symbolic and concrete backends and constrain
/// every public input/output. The proved constraints are interpreted modulo
/// the prime the proof draws.
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
    Matrix(SpartanMatrixError),
    #[error("circuit witness does not satisfy the constraints")]
    Unsatisfied,
    #[error("Spartan failed: {0:?}")]
    Spartan(spartan::SpartanError),
    #[error("commitment failed: {0:?}")]
    Commit(pcs::CommitError),
    #[error("OOD commitment binding verification failed: {0:?}")]
    OodVerify(pcs::VerifyError),
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
    /// Width of the fingerprint prime each proof draws.
    pub prime_bits: u32,
}

/// A circuit prepared for proving over the prime field `F`, whose modulus
/// each proof draws; `Proj` projects the constraints from `R` onto it.
#[derive(Debug)]
pub struct CircuitProofSystem<S, F, R, Proj> {
    statement: S,
    opening_path: OpeningPath,
    /// Prepared once over the integers; projected onto `F` under the drawn prime.
    matrices: PreparedIntegerMatrices<R>,
    map: MaterializedMTranspose,
    claim_shape: Shape,
    generator: F128,
    /// Width of the fingerprint prime, fixed by the claim shape.
    prime_bits: u32,
    committed_shape: Shape,
    pcs: Pcs,
    _phantom: PhantomData<(F, Proj)>,
}

/// A witness kept exact: it takes field values once the prime is drawn.
#[derive(Debug)]
pub struct Witness {
    committed_bits: Vec<F128>,
    /// Only exist on a virtual opening path
    virtual_bits: Option<Vec<F128>>,
    assignment: PackedWitness,
    products: IntegerProducts,
}

/// Commitment data and the transcript that sampled its OOD claim.
pub struct CommittedWitness {
    data: ProverData,
    transcript: ProverState,
}

/// The residues are under the prime the transcript yields for `root`.
#[derive(Clone, Debug)]
pub struct Proof<F> {
    pub root: Root,
    pub spartan: SpartanPiopProof<F>,
    pub opening: transcript::Proof,
}

impl<S, F, R, Proj> CircuitProofSystem<S, F, R, Proj>
where
    S: CircuitStatement,
    F: BitzClaimField + FieldWithDynamicModulus,
    F::Integer: BitWidth + IntoWords,
    R: BitzConstraintRing + for<'a> From<&'a StoredInteger>,
    for<'a> StoredInteger: From<&'a R>,
    Proj: ProjectConstraint<R, F>,
{
    #[tracing::instrument(name = "setup", skip_all)]
    pub fn new(statement: S) -> Result<Self, Error> {
        let mut constraints = ConstraintGenerator::<R>::new(statement.input_bits());
        let inputs: Vec<_> = (0..statement.input_bits())
            .map(|i| constraints.input(i))
            .collect();
        statement.synthesize(&mut constraints, &inputs)?;
        let matrices =
            PreparedIntegerMatrices::new(constraints.into_matrices()).map_err(Error::Matrix)?;
        let mut generator = MTransposeGenerator::new(statement.input_bits());
        let inputs = generator.take_inputs();
        statement.synthesize(&mut generator, &inputs)?;
        let map = generator.finish();
        if map.h_len() != matrices.column_count() {
            return Err(Error::Configuration("map and assignment dimensions differ"));
        }
        if map.f_len() != matrices.m().column_count() {
            return Err(Error::Configuration("map and witness dimensions differ"));
        }
        let claim_shape = shape_for(map.h_len())?;
        let opening_path = if is_identity(matrices.m()) {
            OpeningPath::Direct
        } else {
            OpeningPath::Virtual
        };
        let committed_shape = match opening_path {
            OpeningPath::Direct => claim_shape,
            OpeningPath::Virtual => shape_for(map.f_len() - 1)?,
        };
        let prime_bits = common::prime_bits(&claim_shape)
            .map_err(|_| Error::Configuration("shape too tall for the fingerprint prime"))?;
        let pcs = Pcs::new(&committed_shape, LigeritoProfile::Fast, HashKind::Blake3)
            .map_err(|_| Error::Configuration("unsupported PCS shape"))?;
        Ok(Self {
            statement,
            opening_path,
            matrices,
            map,
            claim_shape,
            generator: smallest_generator(),
            prime_bits,
            committed_shape,
            pcs,
            _phantom: PhantomData,
        })
    }

    pub fn stats(&self) -> CircuitStats {
        CircuitStats {
            opening_path: self.opening_path,
            constraints: self.matrices.row_count(),
            assignment_bits: self.map.h_len(),
            committed_bits: match self.opening_path {
                OpeningPath::Direct => self.map.h_len(),
                OpeningPath::Virtual => self.map.f_len() - 1,
            },
            padded_committed_bits: self.pcs.bit_len(),
            prime_bits: self.prime_bits,
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
        if products.a_mw.len() != self.matrices.row_count() {
            return Err(Error::Input("circuit replay changed constraint count"));
        }
        // Over the integers, so modulo whichever prime is drawn.
        if !products.is_satisfied::<R>() {
            return Err(Error::Unsatisfied);
        }
        let assignment_bits = pack(&h, self.claim_shape);
        let (committed_bits, virtual_bits) = match self.opening_path {
            OpeningPath::Direct => (assignment_bits, None),
            OpeningPath::Virtual => (pack(&f, self.committed_shape), Some(assignment_bits)),
        };
        Ok(Witness {
            committed_bits,
            virtual_bits,
            assignment: h,
            products,
        })
    }

    /// Commits and sends the initial OOD evaluation before any PIOP challenge.
    /// The returned state retains both PCS data and the transcript for proving.
    #[tracing::instrument(name = "commit", skip_all)]
    pub fn commit(&self, witness: &Witness) -> Result<CommittedWitness, Error> {
        let mut transcript = build_prover(SESSION, self.statement.domain());
        let (_, data) = self
            .pcs
            .commit(&witness.committed_bits, &mut transcript)
            .map_err(Error::Commit)?;
        Ok(CommittedWitness { data, transcript })
    }

    /// Continues the commitment transcript through the prime draw, Spartan
    /// and the BitZ opening.
    #[tracing::instrument(name = "prove", skip_all, fields(opening_path = ?self.opening_path))]
    pub fn prove(&self, witness: Witness, commitment: CommittedWitness) -> Result<Proof<F>, Error> {
        let prime_lock = DYNAMIC_MODULUS_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let CommittedWitness {
            data,
            mut transcript,
        } = commitment;
        let root = data.root();
        self.bind(&mut transcript, root);
        if self.opening_path == OpeningPath::Direct {
            self.pcs
                .prove_lin(
                    &data,
                    witness.committed_bits.clone(),
                    &self.constant_query(),
                    StatementBinding::Bind,
                    &mut transcript,
                )
                .map_err(Error::ConstantProve)?;
        }
        let prime = transcript.squeeze_prime(self.prime_bits);
        let (params, matrices) = self.under_prime(prime, &prime_lock, &mut transcript)?;
        let products =
            build_product_mles(&witness.products, matrices.row_count()).map_err(Error::Matrix)?;
        let assignment =
            build_assignment_mle(&witness.assignment, self.map.h_len()).map_err(Error::Matrix)?;
        let (spartan, terminal) =
            prove_spartan_piop(&mut transcript, &matrices, &products, &assignment)
                .map_err(Error::Spartan)?;
        let claim = opening_claim(&params, &terminal)?;
        let prover = BitZProver::new(params, WINDOW);
        match self.opening_path {
            OpeningPath::Direct => prover.prove(
                &claim,
                &self.pcs,
                &data,
                witness.committed_bits,
                &mut transcript,
            ),
            OpeningPath::Virtual => {
                let statement =
                    VirtualStatement::new(params, self.committed_shape, &self.map, &claim)
                        .map_err(|_| Error::Configuration("invalid virtual statement"))?;
                prover.prove_virtual(
                    &statement,
                    &self.pcs,
                    &data,
                    VirtualWitness {
                        committed_bits: witness.committed_bits,
                        virtual_bits: witness
                            .virtual_bits
                            .as_ref()
                            .expect("virtual_bits must exist on virtual path"),
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
    pub fn verify(&self, proof: &Proof<F>) -> Result<(), Error> {
        let prime_lock = DYNAMIC_MODULUS_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut transcript = build_verifier(SESSION, self.statement.domain(), &proof.opening);
        let commitment = self
            .pcs
            .receive_commitment(proof.root, &mut transcript)
            .map_err(Error::OodVerify)?;
        self.bind(&mut transcript, proof.root);
        if self.opening_path == OpeningPath::Direct {
            self.pcs
                .verify_lin(
                    &commitment,
                    &self.constant_query(),
                    StatementBinding::Bind,
                    &mut transcript,
                )
                .map_err(Error::ConstantVerify)?;
        }
        let prime = transcript.squeeze_prime(self.prime_bits);
        let (params, matrices) = self.under_prime(prime, &prime_lock, &mut transcript)?;
        let terminal = verify_spartan_proof(&mut transcript, &matrices, &proof.spartan)
            .map_err(Error::Spartan)?;
        let claim = opening_claim(&params, &terminal)?;
        let verifier = BitZVerifier::new(params, WINDOW);
        match self.opening_path {
            OpeningPath::Direct => {
                verifier.verify_with_commitment(&claim, &self.pcs, &commitment, transcript)
            }
            OpeningPath::Virtual => {
                let statement =
                    VirtualStatement::new(params, self.committed_shape, &self.map, &claim)
                        .map_err(|_| Error::Configuration("invalid virtual statement"))?;
                verifier.verify_virtual_with_commitment(
                    &statement,
                    &self.pcs,
                    &commitment,
                    transcript,
                )
            }
        }
        .map_err(Error::Verify)
    }

    /// Installs the sampled prime and builds the parameters and the matrices
    /// under it. The parameters' frame carries the prime into the transcript.
    fn under_prime(
        &self,
        prime: u128,
        _prime_lock: &ModulusLock<'_>,
        transcript: &mut impl PublicTranscript,
    ) -> Result<(BitZParams<F>, PreparedConstraintMatrices<F>), Error> {
        tracing::info!(bits = self.prime_bits, prime = %prime, "Fingerprint prime drawn");
        // SAFETY: No value of `F` exists yet, and `DYNAMIC_MODULUS_LOCK` is held, so
        // no operation is in flight elsewhere.
        unsafe { F::set_modulus(prime) };
        let params = BitZParams::new(self.claim_shape, self.generator)
            .map_err(|_| Error::Configuration("inadmissible BitZ parameters"))?;
        transcript.public_message(&params);
        let projection = Proj::prepare();
        let matrices = self.matrices.project(|c| projection.project(c));
        Ok((params, matrices))
    }

    // Direct commitment includes h[0]; unlike the virtual map, it does not
    // supply that coordinate as a fixed one. Opening at zero enforces h[0] = 1.
    fn constant_query(&self) -> OpeningQuery {
        OpeningQuery::Mle {
            point: vec![F128::ZERO; self.committed_shape.log_bits()],
            target: F128::ONE,
        }
    }

    /// Everything public and fixed before the prime is drawn.
    fn bind(&self, transcript: &mut impl PublicTranscript, root: Root) {
        let public = self.statement.public_bytes();
        transcript.public_message(&(public.len() as u64));
        transcript.public_message(public.as_slice());
        transcript.public_message(&root.0);
        transcript.public_message(&(self.claim_shape.log_rows() as u64));
        transcript.public_message(&(self.claim_shape.log_columns() as u64));
        transcript.public_message(&self.generator);
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

fn opening_claim<F: BitzClaimField>(
    params: &BitZParams<F>,
    terminal: &ScaledMleEvaluationClaim<F>,
) -> Result<LinearClaim<F>, Error> {
    let shape = params.shape();
    if terminal.point().len() > shape.log_bits() {
        return Err(Error::Configuration("Spartan point exceeds virtual shape"));
    }
    // Zero high coordinates select the original assignment inside its zero padding.
    // Put the scale in one factor, avoiding division even when the scale is zero.
    let mut point = terminal.point().to_vec();
    point.resize(shape.log_bits(), F::zero());
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
    use crate::ProjectBigIntToField;
    use poly::DenseMultilinearExtension;

    type F = field::DynField;
    type R = num_bigint::BigInt;
    type Proj = ProjectBigIntToField;
    type System<S> = CircuitProofSystem<S, F, R, Proj>;

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
    fn commitment_sends_ood_before_proving() {
        let system = System::new(IdentityBit).unwrap();
        let witness = system.witness(&[true]).unwrap();
        let committed = system.commit(&witness).unwrap();
        let proof = committed.transcript.finish();
        assert_eq!(proof.narg_string.len(), 16);
        assert!(proof.hints.is_empty());
        let mut verifier = build_verifier(SESSION, system.statement.domain(), &proof);
        system
            .pcs
            .receive_commitment(committed.data.root(), &mut verifier)
            .unwrap();
        verifier.check_eof().unwrap();
    }

    #[test]
    fn direct_opening_requires_constant_one_on_both_sides() {
        let mut system = System::new(IdentityBit).unwrap();
        let witness = system.witness(&[true]).unwrap();
        let data = system.commit(&witness).unwrap();
        let mut proof = system.prove(witness, data).unwrap();
        system.verify(&proof).unwrap();
        system.opening_path = OpeningPath::Virtual;
        assert!(system.verify(&proof).is_err());
        system.opening_path = OpeningPath::Direct;

        let mut bad_witness = system.witness(&[true]).unwrap();
        bad_witness.committed_bits.fill(F128::ZERO);
        let bad_data = system.commit(&bad_witness).unwrap();
        assert!(matches!(
            system.prove(bad_witness, bad_data),
            Err(Error::ConstantProve(_))
        ));

        // A valid opening to zero must not substitute for the required one.
        let packed = vec![F128::ZERO; 1 << system.committed_shape.log_packed_len()];
        let mut transcript = build_prover(SESSION, system.statement.domain());
        let (_, bad_data) = system.pcs.commit(&packed, &mut transcript).unwrap();
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

    /// The prime is the transcript's: the same statement and commitment
    /// yield it again, another statement or commitment yields another.
    #[test]
    fn the_prime_follows_the_transcript() {
        let system = System::new(IdentityBit).unwrap();
        let witness = system.witness(&[true]).unwrap();
        let root = system.commit(&witness).unwrap().data.root();
        let draw = |domain: &'static [u8], root: Root| {
            let mut transcript = build_prover(SESSION, domain);
            system.bind(&mut transcript, root);
            transcript.squeeze_prime(system.prime_bits)
        };
        let prime = draw(system.statement.domain(), root);
        assert_eq!(u128::BITS - prime.leading_zeros(), system.prime_bits);
        assert!(field::helpers::is_prime(prime));
        assert_eq!(draw(system.statement.domain(), root), prime);
        assert_ne!(draw(b"another/statement", root), prime);
        assert_ne!(draw(system.statement.domain(), Root([0; 32])), prime);
    }

    /// `opening_claim` is generic, so a fixed field keeps this test off the
    /// installed modulus.
    #[test]
    fn scaled_claim_conversion_preserves_values_and_zero_scale() {
        type F = field::FqDefault;
        let params =
            BitZParams::<F>::new(Shape::new(7, 15).unwrap(), smallest_generator()).unwrap();
        let assignment =
            DenseMultilinearExtension::from_evaluations(2, [0u128, 1, 1, 0].map(F::from).to_vec())
                .unwrap();
        let point = vec![F::from(3u128), F::from(5u128)];
        let evaluation = assignment.evaluate(&point).unwrap();
        for scale in [F::ZERO, F::from(7u128)] {
            let terminal = ScaledMleEvaluationClaim::new(
                point.clone().into_boxed_slice(),
                scale,
                scale * evaluation,
            );
            let claim = opening_claim(&params, &terminal).unwrap();
            let value: F = assignment
                .iter()
                .enumerate()
                .map(|(i, bit)| *bit * claim.row_weights()[i] * claim.column_weights()[0])
                .sum();
            assert_eq!(value, claim.target());
            assert!(claim.row_weights()[4..].iter().all(|w| *w == F::ZERO));
            assert!(claim.column_weights()[1..].iter().all(|w| *w == F::ZERO));
        }
    }
}
