//! Spartan sumcheck proving and verification.
//!
//! This module implements the reusable sumcheck reduction and Spartan's
//! equality-weighted outer sumcheck.
//!
//! The prover implements parts of [*More Optimizations to Sum-Check Proving*].
//!
//! # Prover optimizations
//!
//! 1. **Missing-at-one coefficient reconstruction.** The round kernel computes
//!    `[c0, c2, c3]` and derives `c1` from `g_i(0) + g_i(1) = claim`. This avoids
//!    an independent evaluation at `1` and requires neither interpolation nor
//!    inversion.
//! 2. **Split equality tables.** `eq(tau, x)` is represented by low/high
//!    factors. A balanced split stores roughly `2 * 2^(n/2)` entries instead of
//!    `2^n`, then folds the low and high factors separately.
//! 3. **Parallel round accumulation.** Large round sums use Rayon, while sums
//!    below the `2^12` contribution threshold remain sequential.
//! 4. **Fold/round fusion.** Once round `i` has been absorbed and its challenge
//!    sampled, the prover folds the active tables and accumulates round
//!    `i + 1` from those freshly folded values in the same traversal.
//! 5. **Reusable scratch tables.** Product and equality folds ping-pong between
//!    preallocated buffers instead of allocating a new destination each round.
//!
//! The reusable verifier lives on [`SumcheckProof`]. Protocol-specific code is
//! responsible for checking the terminal claim produced by that reduction.
//!
//! [*More Optimizations to Sum-Check Proving*]: https://eprint.iacr.org/2024/1210.pdf

use crypto_primitives::ConstField;
use field::FqDefault;
use poly::DenseMultilinearExtension;
use rayon::prelude::*;
use transcript::{Encoding, ProverState, TranscriptChallenge, VerifierState};

/// Failures produced while reducing or checking a sumcheck claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckError {
    EmptyRoundPolynomial,
    InvalidRoundCount { expected: usize, actual: usize },
    InvalidRoundClaim { round: usize },
    InvalidTerminalClaim,
    InvalidProductDimensions,
    InvalidEqualityDimensions,
    InvalidMleOperation,
}

/// Sumcheck round polynomials in coefficient form.
///
/// `COEFFS` is the maximum degree plus one. For example, a cubic outer
/// sumcheck uses four coefficients and a quadratic inner sumcheck uses three.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof<F, const COEFFS: usize> {
    pub round_polynomials: Vec<[F; COEFFS]>,
}

impl<F, const COEFFS: usize> SumcheckProof<F, COEFFS>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
    /// Verifies the round reductions and returns `(r, final_claim)`.
    ///
    /// The caller supplies the expected number of rounds from the statement.
    /// This method does not check a protocol-specific terminal identity.
    pub fn verify(
        &self,
        transcript: &mut VerifierState<'_>,
        initial_claim: F,
        expected_rounds: usize,
    ) -> Result<(Vec<F>, F), SumcheckError> {
        if COEFFS == 0 {
            return Err(SumcheckError::EmptyRoundPolynomial);
        }

        let actual_rounds = self.round_polynomials.len();
        if actual_rounds != expected_rounds {
            return Err(SumcheckError::InvalidRoundCount {
                expected: expected_rounds,
                actual: actual_rounds,
            });
        }

        let zero = F::ZERO;
        let mut current_claim = initial_claim;
        let mut eval_points = Vec::with_capacity(expected_rounds);

        for (round, coefficients) in self.round_polynomials.iter().enumerate() {
            // The typed proof owns the message bytes, so the transcript only
            // absorbs their canonical encoding; it does not deserialize a
            // second copy from the narg string.
            transcript.public_message(coefficients);

            let at_zero = coefficients[0];
            let at_one = coefficients
                .iter()
                .copied()
                .fold(zero, |sum, coefficient| sum + coefficient);

            if at_zero + at_one != current_claim {
                return Err(SumcheckError::InvalidRoundClaim { round });
            }

            let challenge = transcript.squeeze::<F>();
            current_claim = coefficients
                .iter()
                .rev()
                .copied()
                .fold(zero, |value, coefficient| value * challenge + coefficient);
            eval_points.push(challenge);
        }

        Ok((eval_points, current_claim))
    }
}

/// Local output produced while writing a sumcheck proof to the transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProverOutput<F, const COEFFS: usize> {
    pub proof: SumcheckProof<F, COEFFS>,

    /// Transcript-derived challenges.
    pub eval_points: Vec<F>,

    /// Running claim after the final round.
    pub final_claim: F,
}

/// Dense Boolean-row MLEs for `Az`, `Bz`, and `Cz`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csProductMles<F> {
    pub az: DenseMultilinearExtension<F>,
    pub bz: DenseMultilinearExtension<F>,
    pub cz: DenseMultilinearExtension<F>,
}

/// Proof of the equality-weighted R1CS residual sum.
///
/// The cubic rounds reduce
/// `sum_x eq(tau, x) * (Az(x) * Bz(x) - Cz(x))` to the transcript-derived
/// point `r_x`. The terminal claims are then absorbed in `Az`, `Bz`, `Cz`
/// order and must satisfy
///
/// `final_claim = eq(tau, r_x) * (Az(r_x) * Bz(r_x) - Cz(r_x))`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckProof<F> {
    /// Cubic round proof for the equality-weighted R1CS residual.
    ///
    /// Each round polynomial is stored in coefficient form as
    /// `[c0, c1, c2, c3]`.
    pub sumcheck: SumcheckProof<F, 4>,

    /// Claimed terminal evaluation `Az(r_x)`.
    pub az_mle_claim: F,

    /// Claimed terminal evaluation `Bz(r_x)`.
    pub bz_mle_claim: F,

    /// Claimed terminal evaluation `Cz(r_x)`.
    pub cz_mle_claim: F,
}

/// Local result of the outer-sumcheck prover.
///
/// In each round, the prover absorbs a cubic round polynomial and then samples
/// one evaluation coordinate from the transcript. In round order, these
/// coordinates form `r_x = eval_points`. At termination,
///
/// `final_claim = eq(tau, r_x) * (Az(r_x) * Bz(r_x) - Cz(r_x))`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckOutput<F> {
    /// All transcript-bound outer-sumcheck messages: the cubic round proof and
    /// terminal evaluations `[Az(r_x), Bz(r_x), Cz(r_x)]`.
    pub proof: OuterSumcheckProof<F>,

    /// Transcript-sampled outer evaluation point `r_x`, in round order.
    ///
    /// This is prover-local derived output, not a separately encoded proof
    /// message.
    pub eval_points: Vec<F>,

    /// Terminal running claim after evaluating the last round polynomial at
    /// the final coordinate of `r_x`.
    ///
    /// This value is derived from the round messages and transcript challenges;
    /// it is not a separate outer-sumcheck proof message.
    pub final_claim: F,
}

/// Transcript-derived point and terminal claims returned after verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckVerifierOutput<F> {
    /// Transcript-sampled outer evaluation point `r_x`, replayed while checking
    /// the cubic round proof.
    pub eval_points: Vec<F>,

    /// Claimed terminal evaluation `Az(r_x)`.
    pub az_mle_claim: F,

    /// Claimed terminal evaluation `Bz(r_x)`.
    pub bz_mle_claim: F,

    /// Claimed terminal evaluation `Cz(r_x)`.
    pub cz_mle_claim: F,
}

/// Local result of the Spartan inner-sumcheck prover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerSumcheckOutput<F> {
    /// Quadratic round proof, transcript challenges, and final running claim.
    pub sumcheck: SumcheckProverOutput<F, 3>,

    /// `D(r_y)`, where `D` is the batched matrix MLE.
    pub batched_matrix_evaluation: F,

    /// `witness(r_y)` for the field-valued witness MLE.
    pub witness_evaluation: F,
}

impl<F> OuterSumcheckProof<F>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
    /// Verifies the outer reduction and its terminal R1CS identity.
    pub fn verify(
        &self,
        transcript: &mut VerifierState<'_>,
        initial_claim: F,
        tau: &[F],
    ) -> Result<OuterSumcheckVerifierOutput<F>, SumcheckError> {
        let (eval_points, final_claim) =
            self.sumcheck.verify(transcript, initial_claim, tau.len())?;

        transcript.public_message(&[self.az_mle_claim, self.bz_mle_claim, self.cz_mle_claim]);
        let expected_claim = poly::eq_eval(tau, &eval_points)
            * (self.az_mle_claim * self.bz_mle_claim - self.cz_mle_claim);

        if final_claim != expected_claim {
            return Err(SumcheckError::InvalidTerminalClaim);
        }

        Ok(OuterSumcheckVerifierOutput {
            eval_points,
            az_mle_claim: self.az_mle_claim,
            bz_mle_claim: self.bz_mle_claim,
            cz_mle_claim: self.cz_mle_claim,
        })
    }
}

/// Proves a Spartan outer-sumcheck claim.
///
/// When a protocol supports more than one choice of `F`, its transcript
/// session or instance must bind that choice so proofs from different fields
/// occupy distinct Fiat–Shamir domains.
pub fn prove_outer_sumcheck<F>(
    transcript: &mut ProverState,
    initial_claim: F,
    (eq_low, eq_high): (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
    products: R1csProductMles<F>,
) -> Result<OuterSumcheckOutput<F>, SumcheckError>
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
    let num_vars = products.az.num_vars();
    if products.bz.num_vars() != num_vars || products.cz.num_vars() != num_vars {
        return Err(SumcheckError::InvalidProductDimensions);
    }

    if eq_low
        .num_vars()
        .checked_add(eq_high.num_vars())
        .is_none_or(|eq_vars| eq_vars != num_vars)
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }

    let zero = F::ZERO;
    let mut eq_low: Vec<_> = eq_low.into_iter().collect();
    let mut eq_high: Vec<_> = eq_high.into_iter().collect();
    let mut products = R1csProductTableBuffers::from_mles(products);

    // Each destination is allocated once at half the initial table size. After
    // a fold, swapping makes the old input allocation the next scratch table.
    // Every active scratch entry is overwritten before it is read.
    let mut product_scratch = R1csProductTableBuffers::filled(products.len() / 2, zero);
    let mut eq_low_scratch = vec![zero; eq_low.len() / 2];
    let mut eq_high_scratch = vec![zero; eq_high.len() / 2];

    let mut current_claim = initial_claim;
    let mut eval_points = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);
    let mut coefficients_without_linear =
        compute_coefficients_without_linear(&products, EqualityPairs::new(&eq_low, &eq_high));

    // Bind the low equality variables first. The small equality factor is
    // folded separately because each of its entries is shared by every high
    // suffix; the product traversal then both folds and prepares the next
    // round polynomial from register-resident outputs.
    while eq_low.len() > 1 {
        let challenge = recover_full_round_polynomial_and_sample_next_challenge(
            transcript,
            &mut current_claim,
            coefficients_without_linear,
            &mut round_polynomials,
            &mut eval_points,
        );

        let next_product_len = products.len() / 2;
        let next_eq_low_len = eq_low.len() / 2;
        product_scratch.truncate(next_product_len);
        eq_low_scratch.truncate(next_eq_low_len);
        fold_table(&eq_low, &mut eq_low_scratch, challenge);

        if next_product_len > 1 {
            coefficients_without_linear = fold_products_and_compute_next(
                &products,
                &mut product_scratch,
                challenge,
                EqualityPairs::new(&eq_low_scratch, &eq_high),
            );
        } else {
            debug_assert_eq!(next_product_len, 1);
            fold_product_tables(&products, &mut product_scratch, challenge);
        }

        products.swap(&mut product_scratch);
        std::mem::swap(&mut eq_low, &mut eq_low_scratch);
    }

    debug_assert_eq!(eq_low.len(), 1);
    debug_assert_eq!(products.len(), eq_high.len());

    while eq_high.len() > 1 {
        let challenge = recover_full_round_polynomial_and_sample_next_challenge(
            transcript,
            &mut current_claim,
            coefficients_without_linear,
            &mut round_polynomials,
            &mut eval_points,
        );

        let next_eq_high_len = eq_high.len() / 2;
        debug_assert_eq!(products.len() / 2, next_eq_high_len);
        product_scratch.truncate(next_eq_high_len);
        eq_high_scratch.truncate(next_eq_high_len);

        if next_eq_high_len == 1 {
            fold_products_and_eq(
                &products,
                &mut product_scratch,
                &eq_high,
                &mut eq_high_scratch,
                challenge,
            );
        } else {
            fold_table(&eq_high, &mut eq_high_scratch, challenge);
            coefficients_without_linear = fold_products_and_compute_next(
                &products,
                &mut product_scratch,
                challenge,
                EqualityPairs::new(&eq_low, &eq_high_scratch),
            );
        }

        products.swap(&mut product_scratch);
        std::mem::swap(&mut eq_high, &mut eq_high_scratch);
    }

    let az_mle_claim = products.az[0];
    let bz_mle_claim = products.bz[0];
    let cz_mle_claim = products.cz[0];
    debug_assert_eq!(
        current_claim,
        eq_low[0] * eq_high[0] * (az_mle_claim * bz_mle_claim - cz_mle_claim)
    );

    transcript.public_message(&[az_mle_claim, bz_mle_claim, cz_mle_claim]);

    Ok(OuterSumcheckOutput {
        proof: OuterSumcheckProof {
            sumcheck: SumcheckProof { round_polynomials },
            az_mle_claim,
            bz_mle_claim,
            cz_mle_claim,
        },
        eval_points,
        final_claim: current_claim,
    })
}

/// Owned evaluation tables used by the fused outer-sumcheck kernels.
struct R1csProductTableBuffers<F> {
    az: Vec<F>,
    bz: Vec<F>,
    cz: Vec<F>,
}

impl<F: Copy> R1csProductTableBuffers<F> {
    fn from_mles(products: R1csProductMles<F>) -> Self {
        Self {
            az: products.az.into_iter().collect(),
            bz: products.bz.into_iter().collect(),
            cz: products.cz.into_iter().collect(),
        }
    }

    fn filled(len: usize, value: F) -> Self {
        Self {
            az: vec![value; len],
            bz: vec![value; len],
            cz: vec![value; len],
        }
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.az.len(), self.bz.len());
        debug_assert_eq!(self.az.len(), self.cz.len());
        self.az.len()
    }

    fn truncate(&mut self, len: usize) {
        debug_assert!(self.az.len() >= len);
        debug_assert!(self.bz.len() >= len);
        debug_assert!(self.cz.len() >= len);
        self.az.truncate(len);
        self.bz.truncate(len);
        self.cz.truncate(len);
    }

    fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.az, &mut other.az);
        std::mem::swap(&mut self.bz, &mut other.bz);
        std::mem::swap(&mut self.cz, &mut other.cz);
    }
}

/// Proves the Spartan inner-sumcheck claim
///
/// `initial_claim = sum_y batched_matrix(y) * witness(y)`
///
/// over the shared Boolean domain of the two MLEs.
pub fn prove_inner_sumcheck(
    transcript: &mut ProverState,
    initial_claim: FqDefault,
    batched_matrix_mle: DenseMultilinearExtension<FqDefault>,
    witness_mle: DenseMultilinearExtension<FqDefault>,
) -> Result<InnerSumcheckOutput<FqDefault>, SumcheckError> {
    let num_vars = batched_matrix_mle.num_vars();
    if witness_mle.num_vars() != num_vars {
        return Err(SumcheckError::InvalidProductDimensions);
    }

    let zero = FqDefault::from(0u128);
    let mut batched_matrix = batched_matrix_mle.into_evaluations();
    let mut witness = witness_mle.into_evaluations();
    let mut current_claim = initial_claim;
    let mut eval_points = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);

    if num_vars > 0 {
        // Each round halves both evaluation tables. Allocate the output buffers
        // once, then reuse the previous input buffers as scratch space after
        // swapping them with the newly folded tables.
        let mut batched_matrix_scratch = vec![zero; batched_matrix.len() / 2];
        let mut witness_scratch = vec![zero; witness.len() / 2];

        // The loop below pipelines the rounds so each table fold can also
        // prepare the following round polynomial. `coefficients_without_linear`
        // always holds `[c0, c2]` for the round about to run; the missing `c1`
        // is reconstructed from `current_claim`. Round zero is prepared here.
        // After sampling a challenge, every non-final iteration folds both
        // tables at that challenge and uses the freshly folded values to prepare
        // `[c0, c2]` for the next iteration. This operator fusion avoids a second
        // scan of the folded tables. The final iteration has no next polynomial,
        // so it only interpolates the last pair in each table.
        let mut coefficients_without_linear =
            sum_inner_round_coefficients_without_linear(&batched_matrix, &witness);

        for _round in 0..num_vars {
            let challenge = recover_full_round_polynomial_and_sample_next_challenge::<
                FqDefault,
                2,
                3,
            >(
                transcript,
                &mut current_claim,
                coefficients_without_linear,
                &mut round_polynomials,
                &mut eval_points,
            );

            let next_len = batched_matrix.len() / 2;
            debug_assert_eq!(witness.len() / 2, next_len);
            debug_assert!(batched_matrix_scratch.len() >= next_len);
            debug_assert!(witness_scratch.len() >= next_len);
            batched_matrix_scratch.truncate(next_len);
            witness_scratch.truncate(next_len);

            if next_len == 1 {
                // In the last round, each table has one pair left. Interpolating
                // those pairs at the final challenge produces `D(r_y)` and
                // `witness(r_y)`. No next-round polynomial needs to be prepared.
                batched_matrix_scratch[0] =
                    interpolate_pair([batched_matrix[0], batched_matrix[1]], challenge);
                witness_scratch[0] = interpolate_pair([witness[0], witness[1]], challenge);
            } else {
                // More rounds remain. Evaluate every adjacent pair at this
                // challenge to halve both tables. The fused helper also uses
                // those new values to prepare the next round's coefficients.
                coefficients_without_linear =
                    fold_and_compute_next_inner_round_coefficients_without_linear(
                        &batched_matrix,
                        &witness,
                        &mut batched_matrix_scratch,
                        &mut witness_scratch,
                        challenge,
                    );
            }

            // Use the folded tables as the next round's inputs and recycle the
            // old input buffers as scratch space.
            std::mem::swap(&mut batched_matrix, &mut batched_matrix_scratch);
            std::mem::swap(&mut witness, &mut witness_scratch);
        }
    }

    let batched_matrix_evaluation = batched_matrix[0];
    let witness_evaluation = witness[0];
    debug_assert_eq!(
        current_claim,
        batched_matrix_evaluation * witness_evaluation
    );

    Ok(InnerSumcheckOutput {
        sumcheck: SumcheckProverOutput {
            proof: SumcheckProof { round_polynomials },
            eval_points,
            final_claim: current_claim,
        },
        batched_matrix_evaluation,
        witness_evaluation,
    })
}

#[inline]
fn compute_inner_pair_coefficients_without_linear(
    batched_matrix: [FqDefault; 2],
    witness: [FqDefault; 2],
) -> [FqDefault; 2] {
    let [matrix_zero, matrix_one] = batched_matrix;
    let [witness_zero, witness_one] = witness;

    [
        matrix_zero * witness_zero,
        (matrix_one - matrix_zero) * (witness_one - witness_zero),
    ]
}

fn sum_inner_round_coefficients_without_linear(
    batched_matrix: &[FqDefault],
    witness: &[FqDefault],
) -> [FqDefault; 2] {
    debug_assert_eq!(batched_matrix.len(), witness.len());
    debug_assert!(batched_matrix.len() >= 2);

    let zero = FqDefault::from(0u128);
    let pair_count = batched_matrix.len() / 2;

    if should_parallelize(pair_count) {
        batched_matrix
            .par_chunks_exact(2)
            .zip(witness.par_chunks_exact(2))
            .fold(
                || [zero; 2],
                |sum, (matrix, witness)| {
                    add_coefficients(
                        sum,
                        compute_inner_pair_coefficients_without_linear(
                            [matrix[0], matrix[1]],
                            [witness[0], witness[1]],
                        ),
                    )
                },
            )
            .reduce(|| [zero; 2], add_coefficients::<FqDefault, 2>)
    } else {
        batched_matrix
            .chunks_exact(2)
            .zip(witness.chunks_exact(2))
            .fold([zero; 2], |sum, (matrix, witness)| {
                add_coefficients(
                    sum,
                    compute_inner_pair_coefficients_without_linear(
                        [matrix[0], matrix[1]],
                        [witness[0], witness[1]],
                    ),
                )
            })
    }
}

#[inline]
fn fold_inner_chunk(
    batched_matrix: &[FqDefault],
    witness: &[FqDefault],
    batched_matrix_output: &mut [FqDefault],
    witness_output: &mut [FqDefault],
    challenge: FqDefault,
) -> [FqDefault; 2] {
    debug_assert_eq!(batched_matrix.len(), 4);
    debug_assert_eq!(witness.len(), 4);
    debug_assert_eq!(batched_matrix_output.len(), 2);
    debug_assert_eq!(witness_output.len(), 2);

    let folded_matrix = [
        interpolate_pair([batched_matrix[0], batched_matrix[1]], challenge),
        interpolate_pair([batched_matrix[2], batched_matrix[3]], challenge),
    ];
    let folded_witness = [
        interpolate_pair([witness[0], witness[1]], challenge),
        interpolate_pair([witness[2], witness[3]], challenge),
    ];

    batched_matrix_output.copy_from_slice(&folded_matrix);
    witness_output.copy_from_slice(&folded_witness);
    compute_inner_pair_coefficients_without_linear(folded_matrix, folded_witness)
}

/// Binds the current variable in both tables and simultaneously prepares the
/// next round's `[c0, c2]`. The challenge has already been sampled, so this
/// does not move any work across the Fiat-Shamir boundary.
fn fold_and_compute_next_inner_round_coefficients_without_linear(
    batched_matrix: &[FqDefault],
    witness: &[FqDefault],
    batched_matrix_output: &mut [FqDefault],
    witness_output: &mut [FqDefault],
    challenge: FqDefault,
) -> [FqDefault; 2] {
    debug_assert_eq!(batched_matrix.len(), witness.len());
    debug_assert!(batched_matrix.len() >= 4);
    debug_assert_eq!(batched_matrix_output.len(), batched_matrix.len() / 2);
    debug_assert_eq!(witness_output.len(), witness.len() / 2);

    let zero = FqDefault::from(0u128);
    let chunk_count = batched_matrix.len() / 4;

    if should_parallelize(chunk_count) {
        batched_matrix
            .par_chunks_exact(4)
            .zip(witness.par_chunks_exact(4))
            .zip(batched_matrix_output.par_chunks_exact_mut(2))
            .zip(witness_output.par_chunks_exact_mut(2))
            .fold(
                || [zero; 2],
                |sum, (((matrix, witness), matrix_output), witness_output)| {
                    add_coefficients(
                        sum,
                        fold_inner_chunk(matrix, witness, matrix_output, witness_output, challenge),
                    )
                },
            )
            .reduce(|| [zero; 2], add_coefficients::<FqDefault, 2>)
    } else {
        batched_matrix
            .chunks_exact(4)
            .zip(witness.chunks_exact(4))
            .zip(batched_matrix_output.chunks_exact_mut(2))
            .zip(witness_output.chunks_exact_mut(2))
            .fold(
                [zero; 2],
                |sum, (((matrix, witness), matrix_output), witness_output)| {
                    add_coefficients(
                        sum,
                        fold_inner_chunk(matrix, witness, matrix_output, witness_output, challenge),
                    )
                },
            )
    }
}

#[inline]
fn add_coefficients<F, const COEFFS: usize>(left: [F; COEFFS], right: [F; COEFFS]) -> [F; COEFFS]
where
    F: ConstField + Copy,
{
    std::array::from_fn(|index| left[index] + right[index])
}

fn sum_coefficients<F, const COEFFS: usize>(
    len: usize,
    contribution: impl Fn(usize) -> [F; COEFFS] + Sync,
) -> [F; COEFFS]
where
    F: ConstField + Copy,
{
    let zero = F::ZERO;

    if should_parallelize(len) {
        (0..len)
            .into_par_iter()
            .map(&contribution)
            .reduce(|| [zero; COEFFS], add_coefficients::<F, COEFFS>)
    } else {
        (0..len).fold([zero; COEFFS], |sum, index| {
            add_coefficients(sum, contribution(index))
        })
    }
}

const PARALLEL_SUMCHECK_THRESHOLD: usize = 1 << 12;

#[inline]
fn should_parallelize(work_items: usize) -> bool {
    work_items >= PARALLEL_SUMCHECK_THRESHOLD && rayon::current_num_threads() > 1
}

#[inline]
fn cubic_contribution<F>(eq: [F; 2], az: [F; 2], bz: [F; 2], cz: [F; 2]) -> [F; 3]
where
    F: ConstField + Copy,
{
    let [eq_zero, eq_one] = eq;
    let [az_zero, az_one] = az;
    let [bz_zero, bz_one] = bz;
    let [cz_zero, cz_one] = cz;

    let eq_delta = eq_one - eq_zero;
    let az_delta = az_one - az_zero;
    let bz_delta = bz_one - bz_zero;
    let cz_delta = cz_one - cz_zero;

    let residual_zero = az_zero * bz_zero - cz_zero;
    let residual_linear = az_zero * bz_delta + az_delta * bz_zero - cz_delta;
    let residual_quadratic = az_delta * bz_delta;

    [
        eq_zero * residual_zero,
        eq_zero * residual_quadratic + eq_delta * residual_linear,
        eq_delta * residual_quadratic,
    ]
}

/// Restores the omitted linear coefficient of a sumcheck round polynomial.
///
/// Given `coefficients_without_linear = [c0, c2, ..., cD]`, this returns the
/// complete coefficient array `[c0, c1, c2, ..., cD]`. The sumcheck round invariant
///
/// `current_claim = g(0) + g(1) = 2*c0 + c1 + c2 + ... + cD`
///
/// uniquely determines
///
/// `c1 = current_claim - 2*c0 - c2 - ... - cD`.
///
/// Consequently, `INPUT_COEFFS` must equal the polynomial degree and `COEFFS`
/// must equal `INPUT_COEFFS + 1`.
#[inline]
fn reconstruct_round_coefficients<F, const INPUT_COEFFS: usize, const COEFFS: usize>(
    current_claim: F,
    coefficients_without_linear: [F; INPUT_COEFFS],
) -> [F; COEFFS]
where
    F: ConstField + Copy,
{
    assert!(INPUT_COEFFS >= 1);
    assert_eq!(COEFFS, INPUT_COEFFS + 1);

    let zero = F::ZERO;
    let mut coefficients = [zero; COEFFS];
    coefficients[0] = coefficients_without_linear[0];
    coefficients[2..].copy_from_slice(&coefficients_without_linear[1..]);

    let at_one_without_c1 = coefficients
        .iter()
        .copied()
        .fold(zero, |sum, coefficient| sum + coefficient);
    coefficients[1] = current_claim - coefficients[0] - at_one_without_c1;
    coefficients
}

#[inline]
fn evaluate_polynomial<F, const COEFFS: usize>(coefficients: &[F; COEFFS], point: F) -> F
where
    F: ConstField + Copy,
{
    let zero = F::ZERO;
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(zero, |value, coefficient| value * point + coefficient)
}

/// Completes and records one sumcheck round. Given `[c0, c2, ..., cD]`, it
/// reconstructs `c1` from
///
/// `g_i(0) + g_i(1) = current_claim`,
///
/// absorbs the completed polynomial, samples `r_i`, and updates
/// `current_claim` to `g_i(r_i)`.
fn recover_full_round_polynomial_and_sample_next_challenge<
    F,
    const INPUT_COEFFS: usize,
    const COEFFS: usize,
>(
    transcript: &mut ProverState,
    current_claim: &mut F,
    coefficients_without_linear: [F; INPUT_COEFFS],
    round_polynomials: &mut Vec<[F; COEFFS]>,
    eval_points: &mut Vec<F>,
) -> F
where
    F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
{
    let zero = F::ZERO;
    let coefficients = reconstruct_round_coefficients(*current_claim, coefficients_without_linear);
    let at_one = coefficients
        .iter()
        .copied()
        .fold(zero, |sum, coefficient| sum + coefficient);

    debug_assert_eq!(*current_claim, coefficients[0] + at_one);

    transcript.public_message(&coefficients);
    let challenge = transcript.squeeze::<F>();
    *current_claim = evaluate_polynomial(&coefficients, challenge);
    round_polynomials.push(coefficients);
    eval_points.push(challenge);
    challenge
}

/// Allocation-free adjacent-pair view of the low/high equality-table product.
///
/// The low factor varies fastest. A singleton low factor naturally covers the
/// case where all low variables have already been bound.
struct EqualityPairs<'a, F> {
    low: &'a [F],
    high: &'a [F],
    low_bits: usize,
    low_mask: usize,
}

impl<'a, F> EqualityPairs<'a, F>
where
    F: ConstField + Copy,
{
    fn new(low: &'a [F], high: &'a [F]) -> Self {
        debug_assert!(low.len().is_power_of_two());
        debug_assert!(high.len().is_power_of_two());

        Self {
            low,
            high,
            low_bits: low.len().ilog2() as usize,
            low_mask: low.len() - 1,
        }
    }

    #[inline]
    fn pair(&self, pair: usize) -> [F; 2] {
        if self.low_bits == 0 {
            let scale = self.low[0];
            let index = 2 * pair;
            return [scale * self.high[index], scale * self.high[index + 1]];
        }

        let low_pair_bits = self.low_bits - 1;
        let low_pair = pair & (self.low_mask >> 1);
        let high_weight = self.high[pair >> low_pair_bits];
        [
            self.low[2 * low_pair] * high_weight,
            self.low[2 * low_pair + 1] * high_weight,
        ]
    }
}

/// Computes `[c0, c2, c3]` from adjacent pairs in the current product tables.
fn compute_coefficients_without_linear<F>(
    products: &R1csProductTableBuffers<F>,
    equality_pairs: EqualityPairs<'_, F>,
) -> [F; 3]
where
    F: ConstField + Copy,
{
    let pair_count = products.len() / 2;

    sum_coefficients(pair_count, |pair| {
        let index = 2 * pair;
        cubic_contribution(
            equality_pairs.pair(pair),
            [products.az[index], products.az[index + 1]],
            [products.bz[index], products.bz[index + 1]],
            [products.cz[index], products.cz[index + 1]],
        )
    })
}

#[inline]
fn interpolate_pair<F>(pair: [F; 2], challenge: F) -> F
where
    F: ConstField + Copy,
{
    let [zero, one] = pair;
    zero + challenge * (one - zero)
}

#[inline]
fn fold_two_pairs<F>(values: &[F], challenge: F) -> [F; 2]
where
    F: ConstField + Copy,
{
    debug_assert_eq!(values.len(), 4);
    [
        interpolate_pair([values[0], values[1]], challenge),
        interpolate_pair([values[2], values[3]], challenge),
    ]
}

#[inline]
fn fold_product_chunk<F>(
    az: &[F],
    bz: &[F],
    cz: &[F],
    az_output: &mut [F],
    bz_output: &mut [F],
    cz_output: &mut [F],
    challenge: F,
) -> [[F; 2]; 3]
where
    F: ConstField + Copy,
{
    debug_assert_eq!(az_output.len(), 2);
    debug_assert_eq!(bz_output.len(), 2);
    debug_assert_eq!(cz_output.len(), 2);

    let folded = [
        fold_two_pairs(az, challenge),
        fold_two_pairs(bz, challenge),
        fold_two_pairs(cz, challenge),
    ];
    az_output.copy_from_slice(&folded[0]);
    bz_output.copy_from_slice(&folded[1]);
    cz_output.copy_from_slice(&folded[2]);
    folded
}

/// Folds one evaluation table into an already initialized destination.
fn fold_table<F>(input: &[F], output: &mut [F], challenge: F)
where
    F: ConstField + Copy,
{
    debug_assert_eq!(input.len(), 2 * output.len());

    let fold = |(pair, value): (&[F], &mut F)| {
        *value = interpolate_pair([pair[0], pair[1]], challenge);
    };
    if should_parallelize(output.len()) {
        input
            .par_chunks_exact(2)
            .zip(output.par_iter_mut())
            .for_each(fold);
    } else {
        input.chunks_exact(2).zip(output.iter_mut()).for_each(fold);
    }
}

/// Folds all three product tables into reusable scratch storage.
fn fold_product_tables<F>(
    input: &R1csProductTableBuffers<F>,
    output: &mut R1csProductTableBuffers<F>,
    challenge: F,
) where
    F: ConstField + Copy,
{
    debug_assert_eq!(input.len(), 2 * output.len());

    fold_table(&input.az, &mut output.az, challenge);
    fold_table(&input.bz, &mut output.bz, challenge);
    fold_table(&input.cz, &mut output.cz, challenge);
}

/// Folds all three product tables and accumulates the next round polynomial
/// from the freshly folded pairs.
fn fold_products_and_compute_next<F>(
    input: &R1csProductTableBuffers<F>,
    output: &mut R1csProductTableBuffers<F>,
    challenge: F,
    equality_pairs: EqualityPairs<'_, F>,
) -> [F; 3]
where
    F: ConstField + Copy,
{
    debug_assert_eq!(input.len(), 2 * output.len());

    let zero = F::ZERO;
    let accumulate = |sum: [F; 3],
                      chunk: usize,
                      az: &[F],
                      bz: &[F],
                      cz: &[F],
                      az_output: &mut [F],
                      bz_output: &mut [F],
                      cz_output: &mut [F]| {
        let [az, bz, cz] =
            fold_product_chunk(az, bz, cz, az_output, bz_output, cz_output, challenge);
        add_coefficients(
            sum,
            cubic_contribution(equality_pairs.pair(chunk), az, bz, cz),
        )
    };

    let chunk_count = output.len() / 2;
    if should_parallelize(chunk_count) {
        (
            input.az.par_chunks_exact(4),
            input.bz.par_chunks_exact(4),
            input.cz.par_chunks_exact(4),
            output.az.par_chunks_exact_mut(2),
            output.bz.par_chunks_exact_mut(2),
            output.cz.par_chunks_exact_mut(2),
        )
            .into_par_iter()
            .enumerate()
            .fold(
                || [zero; 3],
                |sum, (chunk, (az, bz, cz, az_output, bz_output, cz_output))| {
                    accumulate(sum, chunk, az, bz, cz, az_output, bz_output, cz_output)
                },
            )
            .reduce(|| [zero; 3], add_coefficients::<F, 3>)
    } else {
        let mut sum = [zero; 3];
        for chunk in 0..chunk_count {
            let input_start = 4 * chunk;
            let output_start = 2 * chunk;
            sum = accumulate(
                sum,
                chunk,
                &input.az[input_start..input_start + 4],
                &input.bz[input_start..input_start + 4],
                &input.cz[input_start..input_start + 4],
                &mut output.az[output_start..output_start + 2],
                &mut output.bz[output_start..output_start + 2],
                &mut output.cz[output_start..output_start + 2],
            );
        }
        sum
    }
}

/// Folds the high equality factor and all product tables together.
fn fold_products_and_eq<F>(
    products: &R1csProductTableBuffers<F>,
    product_output: &mut R1csProductTableBuffers<F>,
    eq: &[F],
    eq_output: &mut [F],
    challenge: F,
) where
    F: ConstField + Copy,
{
    debug_assert_eq!(products.len(), eq.len());
    debug_assert_eq!(products.len(), 2 * product_output.len());
    debug_assert_eq!(eq.len(), 2 * eq_output.len());

    fold_product_tables(products, product_output, challenge);
    fold_table(eq, eq_output, challenge);
}

#[cfg(test)]
mod tests {
    use field::{F128, FqDefault};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use transcript::{build_prover, build_verifier};

    use super::*;

    const SESSION: &[u8] = b"spartan/outer-sumcheck/test";
    const F128_SESSION: &[u8] = b"spartan/outer-sumcheck/f128/test";
    const INNER_SESSION: &[u8] = b"spartan/inner-sumcheck/test";

    fn fq(value: u128) -> FqDefault {
        FqDefault::from(value)
    }

    #[test]
    fn full_coefficient_sumcheck_verifies_and_replays_challenges() {
        let second_round = [fq(1), fq(2), fq(3), fq(3)];
        let sumcheck = SumcheckProof {
            round_polynomials: vec![[fq(10), fq(0), fq(0), fq(0)], second_round],
        };

        let instance = b"full-coefficient-sumcheck";
        let mut prover = build_prover(SESSION, instance);
        let prover_points: Vec<_> = sumcheck
            .round_polynomials
            .iter()
            .map(|round| {
                prover.public_message(round);
                prover.squeeze::<FqDefault>()
            })
            .collect();
        let next_prover_challenge = prover.squeeze::<FqDefault>();
        let transcript_proof = prover.finish();

        let mut verifier = build_verifier(SESSION, instance, &transcript_proof);
        let (verifier_points, final_claim) = sumcheck.verify(&mut verifier, fq(20), 2).unwrap();
        let next_verifier_challenge = verifier.squeeze::<FqDefault>();

        let expected_final_claim = second_round
            .iter()
            .rev()
            .copied()
            .fold(fq(0), |value, coefficient| {
                value * verifier_points[1] + coefficient
            });

        assert_eq!(verifier_points, prover_points);
        assert_eq!(final_claim, expected_final_claim);
        assert_eq!(next_verifier_challenge, next_prover_challenge);
        verifier.check_eof().unwrap();
    }

    #[test]
    fn sumcheck_rejects_an_explicit_bad_c1() {
        let sumcheck = SumcheckProof {
            round_polynomials: vec![[fq(10), fq(0), fq(0), fq(0)], [fq(1), fq(3), fq(3), fq(3)]],
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"bad-c1", &transcript_proof);

        assert_eq!(
            sumcheck.verify(&mut verifier, fq(20), 2),
            Err(SumcheckError::InvalidRoundClaim { round: 1 })
        );
        verifier.check_eof().unwrap();
    }

    #[test]
    fn zero_round_sumcheck_preserves_the_initial_claim() {
        let sumcheck_proof = SumcheckProof::<FqDefault, 4> {
            round_polynomials: vec![],
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"zero-rounds", &transcript_proof);

        assert_eq!(
            sumcheck_proof.verify(&mut verifier, fq(42), 0),
            Ok((vec![], fq(42)))
        );
        verifier.check_eof().unwrap();
    }

    #[test]
    fn sumcheck_rejects_zero_coefficient_rounds() {
        let sumcheck = SumcheckProof::<FqDefault, 0> {
            round_polynomials: vec![[]],
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"zero-coefficients", &transcript_proof);

        assert_eq!(
            sumcheck.verify(&mut verifier, fq(0), 1),
            Err(SumcheckError::EmptyRoundPolynomial)
        );
        verifier.check_eof().unwrap();
    }

    struct OuterSumcheckTestInputs<F> {
        tau: Vec<F>,
        eq_factors: (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
        products: R1csProductMles<F>,
    }

    /// Builds random product MLEs and equality factors from `poly::eq_table`.
    fn build_outer_sumcheck_inputs<F>(num_vars: usize) -> OuterSumcheckTestInputs<F>
    where
        F: ConstField + Copy,
    {
        build_outer_sumcheck_inputs_with_split(num_vars, num_vars / 2)
    }

    fn build_outer_sumcheck_inputs_with_split<F>(
        num_vars: usize,
        split: usize,
    ) -> OuterSumcheckTestInputs<F>
    where
        F: ConstField + Copy,
    {
        assert!(num_vars < usize::BITS as usize);
        assert!(split <= num_vars);

        let table_len = 1usize << num_vars;
        let mut rng = Pcg64::seed_from_u64(0x5a17_a11c);
        let az_values: Vec<F> = (0..table_len)
            .map(|_| F::from(rng.random::<u128>()))
            .collect();
        let bz_values: Vec<F> = (0..table_len)
            .map(|_| F::from(rng.random::<u128>()))
            .collect();
        let cz_values = az_values
            .iter()
            .zip(&bz_values)
            .map(|(&az, &bz)| az * bz)
            .collect();

        let products = R1csProductMles {
            az: DenseMultilinearExtension::from_evaluations(num_vars, az_values).unwrap(),
            bz: DenseMultilinearExtension::from_evaluations(num_vars, bz_values).unwrap(),
            cz: DenseMultilinearExtension::from_evaluations(num_vars, cz_values).unwrap(),
        };

        let tau: Vec<_> = (0..num_vars)
            .map(|index| F::from(2 * index as u128 + 2))
            .collect();
        let (tau_low, tau_high) = tau.split_at(split);
        let eq_factors = (
            DenseMultilinearExtension::from_evaluations(tau_low.len(), poly::eq_table(tau_low))
                .unwrap(),
            DenseMultilinearExtension::from_evaluations(tau_high.len(), poly::eq_table(tau_high))
                .unwrap(),
        );

        OuterSumcheckTestInputs {
            tau,
            eq_factors,
            products,
        }
    }

    /// Runs only the outer prover and outer verifier on prebuilt inputs.
    fn check_outer_sumcheck<F>(session: &[u8], inputs: OuterSumcheckTestInputs<F>)
    where
        F: ConstField + Copy + Encoding<[u8]> + TranscriptChallenge,
    {
        let OuterSumcheckTestInputs {
            tau,
            eq_factors,
            products,
        } = inputs;
        let num_vars = tau.len();
        let instance = (num_vars as u64).to_le_bytes();
        let expected_products = products.clone();

        let mut prover = build_prover(session, &instance);
        let prover_output =
            prove_outer_sumcheck(&mut prover, F::ZERO, eq_factors, products).unwrap();
        let proof = prover.finish();

        let mut verifier = build_verifier(session, &instance, &proof);
        let verifier_output = prover_output
            .proof
            .verify(&mut verifier, F::ZERO, &tau)
            .unwrap();
        verifier.check_eof().unwrap();

        assert_eq!(prover_output.eval_points, verifier_output.eval_points);
        assert_eq!(
            prover_output.proof.az_mle_claim,
            verifier_output.az_mle_claim
        );
        assert_eq!(
            prover_output.proof.bz_mle_claim,
            verifier_output.bz_mle_claim
        );
        assert_eq!(
            prover_output.proof.cz_mle_claim,
            verifier_output.cz_mle_claim
        );
        assert_eq!(
            prover_output.proof.sumcheck.round_polynomials.len(),
            num_vars
        );
        assert_eq!(
            prover_output.proof.az_mle_claim,
            expected_products
                .az
                .evaluate(&prover_output.eval_points)
                .unwrap()
        );
        assert_eq!(
            prover_output.proof.bz_mle_claim,
            expected_products
                .bz
                .evaluate(&prover_output.eval_points)
                .unwrap()
        );
        assert_eq!(
            prover_output.proof.cz_mle_claim,
            expected_products
                .cz
                .evaluate(&prover_output.eval_points)
                .unwrap()
        );
    }

    fn check_outer_sumcheck_input_construction<F>()
    where
        F: ConstField + Copy,
    {
        for num_vars in [0, 1, 3, 10] {
            let inputs = build_outer_sumcheck_inputs::<F>(num_vars);
            let table_len = 1usize << num_vars;

            assert_eq!(inputs.products.az.len(), table_len);
            assert_eq!(inputs.products.bz.len(), table_len);
            assert_eq!(inputs.products.cz.len(), table_len);
            for ((&az, &bz), &cz) in inputs
                .products
                .az
                .iter()
                .zip(inputs.products.bz.iter())
                .zip(inputs.products.cz.iter())
            {
                assert_eq!(cz, az * bz);
            }

            let split = inputs.eq_factors.0.num_vars();
            let low_index_mask = (1usize << split) - 1;
            let full_eq = poly::eq_table(&inputs.tau);
            let (eq_low, eq_high) = &inputs.eq_factors;
            for (index, &expected) in full_eq.iter().enumerate() {
                let actual = eq_low[index & low_index_mask] * eq_high[index >> split];
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn outer_sumcheck_inputs_have_pointwise_products_and_factored_eq() {
        check_outer_sumcheck_input_construction::<FqDefault>();
        check_outer_sumcheck_input_construction::<F128>();
    }

    #[test]
    fn outer_sumcheck_supports_every_equality_factor_split() {
        for split in 0..=5 {
            check_outer_sumcheck(
                SESSION,
                build_outer_sumcheck_inputs_with_split::<FqDefault>(5, split),
            );
            check_outer_sumcheck(
                F128_SESSION,
                build_outer_sumcheck_inputs_with_split::<F128>(5, split),
            );
        }
    }

    #[test]
    fn outer_sumcheck_rejects_dimensions_before_mutating_the_transcript() {
        let instance = b"invalid-outer-dimensions";
        let mut invalid_product_prover = build_prover(SESSION, instance);
        let invalid_products = R1csProductMles {
            az: DenseMultilinearExtension::zero_vars(fq(1)),
            bz: DenseMultilinearExtension::from_evaluations(1, vec![fq(2), fq(3)]).unwrap(),
            cz: DenseMultilinearExtension::zero_vars(fq(4)),
        };
        assert_eq!(
            prove_outer_sumcheck(
                &mut invalid_product_prover,
                fq(0),
                (
                    DenseMultilinearExtension::zero_vars(fq(1)),
                    DenseMultilinearExtension::zero_vars(fq(1)),
                ),
                invalid_products,
            ),
            Err(SumcheckError::InvalidProductDimensions)
        );
        let challenge_after_product_error = invalid_product_prover.squeeze::<FqDefault>();

        let mut invalid_equality_prover = build_prover(SESSION, instance);
        let inputs = build_outer_sumcheck_inputs::<FqDefault>(1);
        assert_eq!(
            prove_outer_sumcheck(
                &mut invalid_equality_prover,
                fq(0),
                (
                    DenseMultilinearExtension::zero_vars(fq(1)),
                    DenseMultilinearExtension::zero_vars(fq(1)),
                ),
                inputs.products,
            ),
            Err(SumcheckError::InvalidEqualityDimensions)
        );
        let challenge_after_equality_error = invalid_equality_prover.squeeze::<FqDefault>();

        let mut clean_prover = build_prover(SESSION, instance);
        let clean_challenge = clean_prover.squeeze::<FqDefault>();
        assert_eq!(challenge_after_product_error, clean_challenge);
        assert_eq!(challenge_after_equality_error, clean_challenge);
    }

    #[test]
    fn outer_verifier_checks_the_zero_round_terminal_claim() {
        let proof = OuterSumcheckProof {
            sumcheck: SumcheckProof {
                round_polynomials: vec![],
            },
            az_mle_claim: fq(2),
            bz_mle_claim: fq(3),
            cz_mle_claim: fq(6),
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"outer-zero-rounds", &transcript_proof);

        assert_eq!(
            proof.verify(&mut verifier, fq(0), &[]),
            Ok(OuterSumcheckVerifierOutput {
                eval_points: vec![],
                az_mle_claim: proof.az_mle_claim,
                bz_mle_claim: proof.bz_mle_claim,
                cz_mle_claim: proof.cz_mle_claim,
            })
        );
        verifier.check_eof().unwrap();
    }

    #[test]
    fn outer_verifier_rejects_bad_terminal_evaluations() {
        let proof = OuterSumcheckProof {
            sumcheck: SumcheckProof {
                round_polynomials: vec![],
            },
            az_mle_claim: fq(2),
            bz_mle_claim: fq(3),
            cz_mle_claim: fq(5),
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"outer-bad-terminal", &transcript_proof);

        assert_eq!(
            proof.verify(&mut verifier, fq(0), &[]),
            Err(SumcheckError::InvalidTerminalClaim)
        );
        verifier.check_eof().unwrap();
    }

    #[test]
    fn outer_sumcheck_zero_vars() {
        check_outer_sumcheck(SESSION, build_outer_sumcheck_inputs::<FqDefault>(0));
        check_outer_sumcheck(F128_SESSION, build_outer_sumcheck_inputs::<F128>(0));
    }

    #[test]
    fn outer_sumcheck_one_var() {
        check_outer_sumcheck(SESSION, build_outer_sumcheck_inputs::<FqDefault>(1));
        check_outer_sumcheck(F128_SESSION, build_outer_sumcheck_inputs::<F128>(1));
    }

    #[test]
    fn outer_sumcheck_three_vars() {
        check_outer_sumcheck(SESSION, build_outer_sumcheck_inputs::<FqDefault>(3));
        check_outer_sumcheck(F128_SESSION, build_outer_sumcheck_inputs::<F128>(3));
    }

    #[test]
    fn outer_sumcheck_ten_vars() {
        check_outer_sumcheck(SESSION, build_outer_sumcheck_inputs::<FqDefault>(10));
        check_outer_sumcheck(F128_SESSION, build_outer_sumcheck_inputs::<F128>(10));
    }

    #[test]
    fn inner_sumcheck_one_variable_has_expected_quadratic() {
        let batched_matrix = DenseMultilinearExtension::from_evaluations(
            1,
            vec![FqDefault::from(2u128), FqDefault::from(5u128)],
        )
        .unwrap();
        let witness = DenseMultilinearExtension::from_evaluations(
            1,
            vec![FqDefault::from(3u128), FqDefault::from(7u128)],
        )
        .unwrap();
        let mut prover = build_prover(INNER_SESSION, b"one-variable");

        let output = prove_inner_sumcheck(
            &mut prover,
            FqDefault::from(41u128),
            batched_matrix,
            witness,
        )
        .unwrap();

        assert_eq!(
            output.sumcheck.proof.round_polynomials,
            vec![[
                FqDefault::from(6u128),
                FqDefault::from(17u128),
                FqDefault::from(12u128),
            ]]
        );
    }

    #[test]
    fn inner_sumcheck_binds_lowest_index_variable_first() {
        let batched_matrix = DenseMultilinearExtension::from_evaluations(
            2,
            [2u128, 5, 11, 17]
                .into_iter()
                .map(FqDefault::from)
                .collect(),
        )
        .unwrap();
        let witness = DenseMultilinearExtension::from_evaluations(
            2,
            [3u128, 7, 13, 19]
                .into_iter()
                .map(FqDefault::from)
                .collect(),
        )
        .unwrap();
        let mut prover = build_prover(INNER_SESSION, b"lowest-variable-first");

        let output = prove_inner_sumcheck(
            &mut prover,
            FqDefault::from(507u128),
            batched_matrix,
            witness,
        )
        .unwrap();

        assert_eq!(
            output.sumcheck.proof.round_polynomials[0],
            [
                FqDefault::from(149u128),
                FqDefault::from(161u128),
                FqDefault::from(48u128),
            ]
        );
    }

    struct InnerSumcheckTestInputs {
        initial_claim: FqDefault,
        batched_matrix_mle: DenseMultilinearExtension<FqDefault>,
        witness_mle: DenseMultilinearExtension<FqDefault>,
    }

    #[test]
    fn inner_sumcheck_proves_matrix_products() {
        for num_column_vars in [0, 1, 3, 12, 13] {
            check_inner_sumcheck(build_inner_sumcheck_inputs(2, num_column_vars));
        }
    }

    fn build_inner_sumcheck_inputs(
        num_row_vars: usize,
        num_column_vars: usize,
    ) -> InnerSumcheckTestInputs {
        assert!(num_row_vars < usize::BITS as usize);
        assert!(num_column_vars < usize::BITS as usize);

        let num_rows = 1usize << num_row_vars;
        let num_columns = 1usize << num_column_vars;
        let matrix_len = num_rows.checked_mul(num_columns).unwrap();
        let zero = FqDefault::from(0u128);
        let mut rng = Pcg64::seed_from_u64(
            0x1a2b_3c4d ^ ((num_row_vars as u64) << 32) ^ num_column_vars as u64,
        );

        let matrix: Vec<FqDefault> = (0..matrix_len).map(|_| rng.random()).collect();
        let witness_values: Vec<FqDefault> = (0..num_columns).map(|_| rng.random()).collect();
        let row_point: Vec<FqDefault> = (0..num_row_vars).map(|_| rng.random()).collect();
        let row_weights = poly::eq_table(&row_point);

        // Compute A * w, then evaluate its row MLE at r_x.
        let matrix_times_witness: Vec<FqDefault> = matrix
            .chunks_exact(num_columns)
            .map(|row| {
                row.iter()
                    .zip(&witness_values)
                    .fold(zero, |sum, (&coefficient, &witness)| {
                        sum + coefficient * witness
                    })
            })
            .collect();
        let initial_claim = row_weights
            .iter()
            .zip(&matrix_times_witness)
            .fold(zero, |sum, (&weight, &value)| sum + weight * value);

        // Bind the row variables: D = A^T * eq(r_x).
        let batched_matrix_values: Vec<FqDefault> = (0..num_columns)
            .map(|column| {
                (0..num_rows).fold(zero, |sum, row| {
                    sum + matrix[row * num_columns + column] * row_weights[row]
                })
            })
            .collect();

        let inner_claim = batched_matrix_values
            .iter()
            .zip(&witness_values)
            .fold(zero, |sum, (&matrix, &witness)| sum + matrix * witness);
        assert_eq!(initial_claim, inner_claim);

        InnerSumcheckTestInputs {
            initial_claim,
            batched_matrix_mle: DenseMultilinearExtension::from_evaluations(
                num_column_vars,
                batched_matrix_values,
            )
            .unwrap(),
            witness_mle: DenseMultilinearExtension::from_evaluations(
                num_column_vars,
                witness_values,
            )
            .unwrap(),
        }
    }

    fn check_inner_sumcheck(inputs: InnerSumcheckTestInputs) {
        let InnerSumcheckTestInputs {
            initial_claim,
            batched_matrix_mle,
            witness_mle,
        } = inputs;
        let num_vars = batched_matrix_mle.num_vars();
        assert_eq!(witness_mle.num_vars(), num_vars);

        let expected_batched_matrix = batched_matrix_mle.clone();
        let expected_witness = witness_mle.clone();
        let instance = (num_vars as u64).to_le_bytes();
        let mut prover = build_prover(INNER_SESSION, &instance);

        let output =
            prove_inner_sumcheck(&mut prover, initial_claim, batched_matrix_mle, witness_mle)
                .unwrap();
        let next_prover_challenge = prover.squeeze::<FqDefault>();
        let transcript_proof = prover.finish();

        let mut verifier = build_verifier(INNER_SESSION, &instance, &transcript_proof);
        let (verifier_points, verifier_final_claim) = output
            .sumcheck
            .proof
            .verify(&mut verifier, initial_claim, num_vars)
            .unwrap();
        let next_verifier_challenge = verifier.squeeze::<FqDefault>();
        verifier.check_eof().unwrap();

        assert_eq!(output.sumcheck.proof.round_polynomials.len(), num_vars);
        assert_eq!(output.sumcheck.eval_points, verifier_points);
        assert_eq!(output.sumcheck.final_claim, verifier_final_claim);
        assert_eq!(next_prover_challenge, next_verifier_challenge);
        assert_eq!(
            output.batched_matrix_evaluation,
            expected_batched_matrix.evaluate(&verifier_points).unwrap()
        );
        assert_eq!(
            output.witness_evaluation,
            expected_witness.evaluate(&verifier_points).unwrap()
        );
        assert_eq!(
            verifier_final_claim,
            output.batched_matrix_evaluation * output.witness_evaluation
        );
    }

    #[test]
    fn inner_sumcheck_supports_zero_variables() {
        let batched_matrix = DenseMultilinearExtension::zero_vars(FqDefault::from(5u128));
        let witness = DenseMultilinearExtension::zero_vars(FqDefault::from(7u128));
        let initial_claim = FqDefault::from(35u128);
        let mut prover = build_prover(INNER_SESSION, b"zero-variables");

        let output =
            prove_inner_sumcheck(&mut prover, initial_claim, batched_matrix, witness).unwrap();

        assert!(output.sumcheck.proof.round_polynomials.is_empty());
        assert!(output.sumcheck.eval_points.is_empty());
        assert_eq!(output.sumcheck.final_claim, initial_claim);
        assert_eq!(output.batched_matrix_evaluation, FqDefault::from(5u128));
        assert_eq!(output.witness_evaluation, FqDefault::from(7u128));

        let mut control = build_prover(INNER_SESSION, b"zero-variables");
        assert_eq!(
            prover.squeeze::<FqDefault>(),
            control.squeeze::<FqDefault>()
        );
    }

    #[test]
    fn inner_sumcheck_rejects_mismatched_dimensions() {
        let batched_matrix = DenseMultilinearExtension::from_evaluations(
            1,
            vec![FqDefault::from(1u128), FqDefault::from(2u128)],
        )
        .unwrap();
        let witness = DenseMultilinearExtension::from_evaluations(
            2,
            vec![
                FqDefault::from(1u128),
                FqDefault::from(2u128),
                FqDefault::from(3u128),
                FqDefault::from(4u128),
            ],
        )
        .unwrap();
        let mut prover = build_prover(INNER_SESSION, b"mismatched-dimensions");

        assert_eq!(
            prove_inner_sumcheck(&mut prover, FqDefault::from(0u128), batched_matrix, witness,),
            Err(SumcheckError::InvalidProductDimensions)
        );

        let mut control = build_prover(INNER_SESSION, b"mismatched-dimensions");
        assert_eq!(
            prover.squeeze::<FqDefault>(),
            control.squeeze::<FqDefault>()
        );
    }
}
