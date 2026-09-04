//! A circuit proved end to end: Spartan over the integer witness, F2Z over
//! the committed bits.
//!
//! Three replays of the same circuit produce the three things the proof needs:
//! [`ConstraintGenerator`] emits `M`, `A`, `B` and `C`; [`MTransposeGenerator`]
//! emits the compact `M^T` the virtual path applies; [`ProductWitgen`] emits
//! the bit witness `f`, the integer witness `h = M (1 ‖ f)`, and the exact
//! products `Ah`, `Bh`, `Ch`.
//!
//! Spartan then reduces R1CS satisfaction to one evaluation claim about `h`,
//! and F2Z discharges that claim against a commitment to `f`.

use circuit::constraints::ConstraintGenerator;
use circuit::matrix_transpose::{MTransposeGenerator, MaterializedMTranspose};
use circuit::sha256::sha256_block_aligned_circuit;
use circuit::witgen::{PackedWitness, ProductWitgen};
use common::{ClaimError, F2ZParams, LinearClaim, Root, Shape, VirtualParams, VirtualParamsError};
use field::{F128, FqDefault, Q100, gf128::smallest_generator};
use pcs::{CommitError, HashKind, LigeritoProfile, Pcs, ProverData};
use poly::DenseMultilinearExtension;
use prover::{VirtualProveError, VirtualProver};
use reduction::{GrandProduct, ReductionError};
use spartan::{
    PreparedConstraintMatrices, R1csProductMles, SpartanError, SpartanPiopProof, bigint_to_fq,
    build_assignment_mle, build_product_mles, prove_spartan_piop, verify_spartan_proof,
};
use transcript::{Proof, VerifierState};
use verifier::{VirtualVerifier, VirtualVerifyError};

/// The comb window. Trades table size against multiplies per exponentiation.
pub const WINDOW: u32 = 8;

/// The inner-product opening is defined only for the secure profile, and the
/// virtual path always reaches it: `M^T v` is not an equality weight.
pub const PROFILE: LigeritoProfile = LigeritoProfile::Secure;

/// Anything that stops a statement from being built or proved.
#[derive(Debug)]
pub enum E2eError {
    Shape(common::ShapeError),
    Params(common::ParamsError),
    VirtualParams(VirtualParamsError),
    Claim(ClaimError),
    Spartan(SpartanError),
    Commit(CommitError),
    Prove(VirtualProveError<ReductionError>),
    Verify(VirtualVerifyError<ReductionError>),
    /// The witness does not fit the shape it was given.
    WitnessTooLarge {
        bits: usize,
        shape_bits: usize,
    },
}

/// Everything a proof of one SHA-256 message needs, built once.
pub struct Sha256Statement {
    matrices: PreparedConstraintMatrices<FqDefault>,
    transpose: MaterializedMTranspose,
    products: R1csProductMles<FqDefault>,
    assignment: DenseMultilinearExtension<FqDefault>,
    committed_packed: Vec<F128>,
    integer_packed: Vec<F128>,
    params: VirtualParams<Q100>,
    pcs: Pcs,
    com: Root,
    data: ProverData,
}

/// The circuit's own sizes, before any shape is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitSize {
    /// Committed bits.
    pub witness_bits: usize,
    /// Entries of the integer witness.
    pub assignment_bits: usize,
    /// Rank-1 constraints.
    pub r1cs_rows: usize,
}

/// Runs the constraint generator alone, which is enough to size the shapes.
pub fn size_of(message_bits: usize) -> CircuitSize {
    let mut generator = ConstraintGenerator::new(message_bits);
    let inputs: Vec<_> = (0..message_bits).map(|i| generator.input(i)).collect();
    let _ = sha256_block_aligned_circuit(&mut generator, message_bits, |bit| inputs[bit].clone());
    let matrices = generator.into_matrices();

    CircuitSize {
        witness_bits: matrices.m.column_count() - 1,
        assignment_bits: matrices.m.row_count(),
        r1cs_rows: matrices.a.row_count(),
    }
}

/// The smallest admissible shape that holds `bits`, as square as the row floor
/// allows.
///
/// Both witnesses are zero-padded up to their shape, so a shape only has to be
/// large enough. Splitting evenly keeps the fold's two sides comparable.
pub fn shape_for(bits: usize) -> Result<Shape, common::ShapeError> {
    let needed = bits.max(1).next_power_of_two().trailing_zeros() as usize;
    let log_bits = needed.max(common::shape::MIN_LOG_BITS);
    let log_rows = (log_bits / 2).max(common::shape::PACK_BITS as usize);
    Shape::new(log_rows, log_bits - log_rows)
}

/// Packs a witness into a shape, zero-padding to its full width.
fn pack(witness: &PackedWitness, shape: &Shape) -> Result<Vec<F128>, E2eError> {
    let shape_bits = 1usize << shape.log_bits();
    if witness.bit_len() > shape_bits {
        return Err(E2eError::WitnessTooLarge {
            bits: witness.bit_len(),
            shape_bits,
        });
    }

    let mut words = witness.words().to_vec();
    words.resize(shape_bits / 64, 0);
    Ok(words
        .chunks_exact(2)
        .map(|pair| F128::new(pair[0], pair[1]))
        .collect())
}

impl Sha256Statement {
    /// Replays the circuit three ways and commits to the bit witness.
    ///
    /// `message_bits` must be a multiple of 512; the circuit adds SHA-256's
    /// own padding block.
    pub fn build(message: &[bool]) -> Result<Self, E2eError> {
        let message_bits = message.len();

        let _guard = prof::scope("build");
        let matrices_guard = prof::scope("build/constraint-matrices");
        let mut generator = ConstraintGenerator::new(message_bits);
        let symbolic: Vec<_> = (0..message_bits).map(|i| generator.input(i)).collect();
        let _ =
            sha256_block_aligned_circuit(&mut generator, message_bits, |bit| symbolic[bit].clone());
        let integer_matrices = generator.into_matrices();
        drop(matrices_guard);

        let transpose_guard = prof::scope("build/m-transpose");
        let mut transpose_generator = MTransposeGenerator::new(message_bits);
        let transpose_inputs = transpose_generator.take_inputs();
        let _ = sha256_block_aligned_circuit(&mut transpose_generator, message_bits, |bit| {
            transpose_inputs[bit]
        });
        let transpose = transpose_generator.finish();
        drop(transpose_guard);

        let witgen_guard = prof::scope("build/witgen");
        let mut witgen = ProductWitgen::with_inputs(message);
        let _ = sha256_block_aligned_circuit(&mut witgen, message_bits, |bit| message[bit]);
        let (witness, assignment_bits, exact_products) = witgen.into_parts();
        drop(witgen_guard);

        let lower_guard = prof::scope("build/lower-to-field");
        let matrices = integer_matrices.map_coefficients(|c| bigint_to_fq(&c));
        let products = build_product_mles(&exact_products, matrices.a.row_count())
            .map_err(SpartanError::Matrix)
            .map_err(E2eError::Spartan)?;
        let assignment = build_assignment_mle(&assignment_bits, matrices.a.column_count())
            .map_err(SpartanError::Matrix)
            .map_err(E2eError::Spartan)?;
        let matrices = PreparedConstraintMatrices::new(matrices)
            .map_err(SpartanError::Matrix)
            .map_err(E2eError::Spartan)?;

        drop(lower_guard);

        let committed_shape = shape_for(witness.bit_len()).map_err(E2eError::Shape)?;
        let claim_shape = shape_for(assignment_bits.bit_len()).map_err(E2eError::Shape)?;
        let committed_packed = pack(&witness, &committed_shape)?;
        let integer_packed = pack(&assignment_bits, &claim_shape)?;

        let claim_params =
            F2ZParams::<Q100>::new(claim_shape, smallest_generator()).map_err(E2eError::Params)?;
        let params = VirtualParams::new(claim_params, committed_shape, &transpose)
            .map_err(E2eError::VirtualParams)?;

        let pcs =
            Pcs::new(&committed_shape, PROFILE, HashKind::Blake3).map_err(E2eError::Commit)?;
        let (com, data) = {
            let _guard = prof::scope("build/commit");
            pcs.commit(&committed_packed).map_err(E2eError::Commit)?
        };

        Ok(Self {
            matrices,
            transpose,
            products,
            assignment,
            committed_packed,
            integer_packed,
            params,
            pcs,
            com,
            data,
        })
    }

    pub fn params(&self) -> &VirtualParams<Q100> {
        &self.params
    }

    pub fn commitment(&self) -> Root {
        self.com
    }

    /// Spartan, then F2Z, on one transcript.
    ///
    /// The PIOP proof rides back separately: its messages are absorbed as
    /// public values rather than written to the narg channel, so the verifier
    /// needs the typed proof alongside the byte stream.
    pub fn prove(&self) -> Result<(Proof, SpartanPiopProof<FqDefault>), E2eError> {
        let _guard = prof::scope("prove");
        let mut transcript = transcript::build_prover(SESSION, INSTANCE);
        let spartan_guard = prof::scope("prove/spartan");
        let (piop, evaluation) = prove_spartan_piop(
            &mut transcript,
            &self.matrices,
            self.products.clone(),
            self.assignment.clone(),
        )
        .map_err(E2eError::Spartan)?;

        drop(spartan_guard);

        let claim = LinearClaim::from_evaluation(self.params.claim(), &evaluation)
            .map_err(E2eError::Claim)?;

        let _f2z = prof::scope("prove/f2z");
        VirtualProver::new(self.params, WINDOW)
            .prove(
                &self.transpose,
                &claim,
                &self.pcs,
                &self.data,
                &self.integer_packed,
                self.committed_packed.clone(),
                &GrandProduct,
                &mut transcript,
            )
            .map_err(E2eError::Prove)?;

        Ok((transcript.finish(), piop))
    }

    /// Replays both halves. Only the commitment, the matrices and the shapes
    /// are read here; no witness is touched.
    pub fn verify(
        &self,
        proof: &Proof,
        piop: &SpartanPiopProof<FqDefault>,
    ) -> Result<(), E2eError> {
        let _guard = prof::scope("verify");
        let mut transcript: VerifierState<'_> =
            transcript::build_verifier(SESSION, INSTANCE, proof);
        let spartan_guard = prof::scope("verify/spartan");
        let evaluation = verify_spartan_proof(&mut transcript, &self.matrices, piop)
            .map_err(E2eError::Spartan)?;

        drop(spartan_guard);

        let claim = LinearClaim::from_evaluation(self.params.claim(), &evaluation)
            .map_err(E2eError::Claim)?;

        let _f2z = prof::scope("verify/f2z");
        VirtualVerifier::new(self.params, WINDOW)
            .verify(
                &self.transpose,
                &claim,
                &self.pcs,
                self.com,
                &GrandProduct,
                transcript,
            )
            .map_err(E2eError::Verify)
    }
}

const SESSION: &[u8] = b"f2z/sha256/v1";
const INSTANCE: &[u8] = b"block-aligned";
