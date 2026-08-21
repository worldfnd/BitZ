//! Shared Spartan sumcheck interfaces.
//!
//! The reusable verifier lives on [`SumcheckProof`]. Protocol-specific code is
//! responsible for checking the terminal claim produced by that reduction.

use field::{FqDefault, Q100};
use poly::DenseMultilinearExtension;
use rayon::prelude::*;
use transcript::{ProverState, VerifierState};

const PARALLEL_SUMCHECK_THRESHOLD: usize = 1 << 12;

/// Failures produced while reducing or checking a sumcheck claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckError {
    EmptyRoundPolynomial,
    InvalidRoundCount { expected: usize, actual: usize },
    InvalidRoundClaim { round: usize },
    InvalidPointLength { expected: usize, actual: usize },
    InvalidTerminalClaim,
    InvalidProductDimensions,
    InvalidEqualityDimensions,
    InvalidMleOperation,
}

impl From<poly::EqEvalError> for SumcheckError {
    fn from(error: poly::EqEvalError) -> Self {
        match error {
            poly::EqEvalError::PointLengthMismatch { expected, actual } => {
                Self::InvalidPointLength { expected, actual }
            }
        }
    }
}

/// Sumcheck round polynomials in coefficient form.
///
/// `COEFFS` is the maximum degree plus one. For example, a cubic outer
/// sumcheck uses four coefficients and a quadratic inner sumcheck uses three.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof<F, const COEFFS: usize> {
    pub round_polynomials: Vec<[F; COEFFS]>,
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

/// Prover messages for the complete Spartan outer sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckProof<F> {
    /// Cubic round polynomials, stored as `[c0, c1, c2, c3]`.
    pub sumcheck: SumcheckProof<F, 4>,

    /// `[Az(r_x), Bz(r_x), Cz(r_x)]`.
    pub product_evaluations: [F; 3],
}

/// Local result of the outer-sumcheck prover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckOutput<F> {
    pub proof: OuterSumcheckProof<F>,

    /// Transcript-derived outer evaluation point `r_x`.
    pub eval_points: Vec<F>,

    /// Running claim after the final round.
    pub final_claim: F,
}

/// Result returned after the verifier checks the outer terminal equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterSumcheckVerifierOutput<F> {
    /// Transcript-derived outer evaluation point `r_x`.
    pub eval_points: Vec<F>,

    /// `[Az(r_x), Bz(r_x), Cz(r_x)]` supplied by and absorbed from the proof.
    pub product_evaluations: [F; 3],
}

pub(crate) trait FqChallengeSource {
    fn squeeze_u128(&mut self) -> u128;
}

impl FqChallengeSource for ProverState {
    fn squeeze_u128(&mut self) -> u128 {
        self.verifier_message()
    }
}

impl FqChallengeSource for VerifierState<'_> {
    fn squeeze_u128(&mut self) -> u128 {
        self.verifier_message()
    }
}

const REJECTION_REMAINDER: u128 = (u128::MAX % Q100 + 1) % Q100;
const MAX_ACCEPTED_CHALLENGE: u128 = u128::MAX - REJECTION_REMAINDER;

/// Draws an exactly uniform Q100 element from 128-bit transcript squeezes.
pub(crate) fn challenge_fq(transcript: &mut impl FqChallengeSource) -> FqDefault {
    loop {
        let candidate = transcript.squeeze_u128();
        if candidate <= MAX_ACCEPTED_CHALLENGE {
            return FqDefault::from(candidate);
        }
    }
}

impl<const COEFFS: usize> SumcheckProof<FqDefault, COEFFS> {
    /// Verifies the round reductions and returns `(r, final_claim)`.
    ///
    /// The caller supplies the expected number of rounds from the statement.
    /// This method does not check a protocol-specific terminal identity.
    pub fn verify(
        &self,
        transcript: &mut VerifierState<'_>,
        initial_claim: FqDefault,
        expected_rounds: usize,
    ) -> Result<(Vec<FqDefault>, FqDefault), SumcheckError> {
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

        let zero = FqDefault::from(0u128);
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

            let challenge = challenge_fq(transcript);
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

/// Proves a Spartan outer-sumcheck claim.
///
pub fn prove_outer_sumcheck(
    transcript: &mut ProverState,
    initial_claim: FqDefault,
    (mut eq_low, mut eq_high): (
        DenseMultilinearExtension<FqDefault>,
        DenseMultilinearExtension<FqDefault>,
    ),
    mut products: R1csProductMles<FqDefault>,
) -> Result<OuterSumcheckOutput<FqDefault>, SumcheckError> {
    let num_vars = products.az.num_vars();
    if products.bz.num_vars() != num_vars || products.cz.num_vars() != num_vars {
        return Err(SumcheckError::InvalidProductDimensions);
    }

    let split = eq_low.num_vars();
    if split
        .checked_add(eq_high.num_vars())
        .is_none_or(|eq_vars| eq_vars != num_vars)
    {
        return Err(SumcheckError::InvalidEqualityDimensions);
    }

    let mut current_claim = initial_claim;
    let mut eval_points = Vec::with_capacity(num_vars);
    let mut round_polynomials = Vec::with_capacity(num_vars);

    for _round in 0..split {
        let partial = split_round_partial(&eq_low, &eq_high, &products);
        let challenge = prove_round(
            transcript,
            &mut current_claim,
            partial,
            &mut round_polynomials,
            &mut eval_points,
        );

        fold_product_mles(&mut products, challenge)?;
        eq_low
            .fold(&[challenge])
            .map_err(|_| SumcheckError::InvalidMleOperation)?;
    }

    let expected_bound_len = 1usize << (num_vars - split);
    debug_assert_eq!(products.az.len(), expected_bound_len);
    debug_assert_eq!(products.az.len(), products.bz.len());
    debug_assert_eq!(products.az.len(), products.cz.len());
    debug_assert_eq!(eq_low.len(), 1);

    let eq_scale = eq_low[0];
    for _round in split..num_vars {
        let partial = folded_round_partial(eq_scale, &eq_high, &products);
        let challenge = prove_round(
            transcript,
            &mut current_claim,
            partial,
            &mut round_polynomials,
            &mut eval_points,
        );

        fold_product_mles(&mut products, challenge)?;
        eq_high
            .fold(&[challenge])
            .map_err(|_| SumcheckError::InvalidMleOperation)?;
    }

    let product_evaluations = [products.az[0], products.bz[0], products.cz[0]];
    let [a, b, c] = product_evaluations;
    debug_assert_eq!(current_claim, eq_low[0] * eq_high[0] * (a * b - c));

    transcript.public_message(&product_evaluations);

    Ok(OuterSumcheckOutput {
        proof: OuterSumcheckProof {
            sumcheck: SumcheckProof { round_polynomials },
            product_evaluations,
        },
        eval_points,
        final_claim: current_claim,
    })
}

/// Verifies the outer reduction and its terminal R1CS identity.
pub fn verify_outer_sumcheck(
    transcript: &mut VerifierState<'_>,
    initial_claim: FqDefault,
    proof: &OuterSumcheckProof<FqDefault>,
    tau: &[FqDefault],
) -> Result<OuterSumcheckVerifierOutput<FqDefault>, SumcheckError> {
    let (eval_points, final_claim) = proof
        .sumcheck
        .verify(transcript, initial_claim, tau.len())?;

    transcript.public_message(&proof.product_evaluations);
    let [a, b, c] = proof.product_evaluations;
    let expected_claim = poly::eq_eval(tau, &eval_points)? * (a * b - c);

    if final_claim != expected_claim {
        return Err(SumcheckError::InvalidTerminalClaim);
    }

    Ok(OuterSumcheckVerifierOutput {
        eval_points,
        product_evaluations: proof.product_evaluations,
    })
}

#[inline]
fn add_coefficients<const COEFFS: usize>(
    left: [FqDefault; COEFFS],
    right: [FqDefault; COEFFS],
) -> [FqDefault; COEFFS] {
    std::array::from_fn(|index| left[index] + right[index])
}

fn sum_coefficients<const COEFFS: usize>(
    len: usize,
    contribution: impl Fn(usize) -> [FqDefault; COEFFS] + Sync,
) -> [FqDefault; COEFFS] {
    let zero = FqDefault::from(0u128);

    if len >= PARALLEL_SUMCHECK_THRESHOLD {
        (0..len)
            .into_par_iter()
            .map(&contribution)
            .reduce(|| [zero; COEFFS], add_coefficients::<COEFFS>)
    } else {
        (0..len).fold([zero; COEFFS], |sum, index| {
            add_coefficients(sum, contribution(index))
        })
    }
}

#[inline]
fn cubic_contribution(
    eq: [FqDefault; 2],
    az: [FqDefault; 2],
    bz: [FqDefault; 2],
    cz: [FqDefault; 2],
) -> [FqDefault; 3] {
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
/// Given `partial = [c0, c2, ..., cD]`, this returns the complete coefficient
/// array `[c0, c1, c2, ..., cD]`. The sumcheck round invariant
///
/// `current_claim = g(0) + g(1) = 2*c0 + c1 + c2 + ... + cD`
///
/// uniquely determines
///
/// `c1 = current_claim - 2*c0 - c2 - ... - cD`.
///
/// Consequently, `PARTIAL_COEFFS` must equal the polynomial degree and
/// `COEFFS` must equal `PARTIAL_COEFFS + 1`.
#[inline]
fn complete_round<const PARTIAL_COEFFS: usize, const COEFFS: usize>(
    current_claim: FqDefault,
    partial: [FqDefault; PARTIAL_COEFFS],
) -> [FqDefault; COEFFS] {
    assert!(PARTIAL_COEFFS >= 1);
    assert_eq!(COEFFS, PARTIAL_COEFFS + 1);

    let zero = FqDefault::from(0u128);
    let mut coefficients = [zero; COEFFS];
    coefficients[0] = partial[0];
    coefficients[2..].copy_from_slice(&partial[1..]);

    let at_one_without_c1 = coefficients
        .iter()
        .copied()
        .fold(zero, |sum, coefficient| sum + coefficient);
    coefficients[1] = current_claim - coefficients[0] - at_one_without_c1;
    coefficients
}

#[inline]
fn evaluate_polynomial<const COEFFS: usize>(
    coefficients: &[FqDefault; COEFFS],
    point: FqDefault,
) -> FqDefault {
    let zero = FqDefault::from(0u128);
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
fn prove_round<const PARTIAL_COEFFS: usize, const COEFFS: usize>(
    transcript: &mut ProverState,
    current_claim: &mut FqDefault,
    partial: [FqDefault; PARTIAL_COEFFS],
    round_polynomials: &mut Vec<[FqDefault; COEFFS]>,
    eval_points: &mut Vec<FqDefault>,
) -> FqDefault {
    let zero = FqDefault::from(0u128);
    let coefficients = complete_round(*current_claim, partial);
    let at_one = coefficients
        .iter()
        .copied()
        .fold(zero, |sum, coefficient| sum + coefficient);

    debug_assert_eq!(*current_claim, coefficients[0] + at_one);

    transcript.public_message(&coefficients);
    let challenge = challenge_fq(transcript);
    *current_claim = evaluate_polynomial(&coefficients, challenge);
    round_polynomials.push(coefficients);
    eval_points.push(challenge);
    challenge
}

/// Computes `[c0, c2, c3]` from the currently folded product MLEs while the
/// active variable belongs to `eq_low`.
///
/// After challenges `r_{<i}` have been folded in place, each adjacent pair is
/// the pair of endpoints at the current variable:
///
/// `P_pair(T) = P(r_{<i}, T, s)` for `P in {Az, Bz, Cz}`.
///
/// The equality endpoints are reconstructed from the corresponding low and
/// high suffix indices, and the helper sums
///
/// `eq(tau, (r_{<i}, T, s)) * (Az_pair(T) Bz_pair(T) - Cz_pair(T))`
///
/// over every remaining Boolean suffix `s`.
fn split_round_partial(
    eq_low: &DenseMultilinearExtension<FqDefault>,
    eq_high: &DenseMultilinearExtension<FqDefault>,
    products: &R1csProductMles<FqDefault>,
) -> [FqDefault; 3] {
    debug_assert!(eq_low.num_vars() > 0);

    let pair_count = products.az.len() / 2;
    let low_tail_bits = eq_low.num_vars() - 1;
    let low_tail_mask = if low_tail_bits == 0 {
        0
    } else {
        (1usize << low_tail_bits) - 1
    };

    sum_coefficients(pair_count, |pair| {
        let index = 2 * pair;
        let low_tail = pair & low_tail_mask;
        let high_index = pair >> low_tail_bits;
        let high_weight = eq_high[high_index];

        cubic_contribution(
            [
                eq_low[2 * low_tail] * high_weight,
                eq_low[2 * low_tail + 1] * high_weight,
            ],
            [products.az[index], products.az[index + 1]],
            [products.bz[index], products.bz[index + 1]],
            [products.cz[index], products.cz[index + 1]],
        )
    })
}

/// Binds the current (lowest-index) variable of all three product MLEs to the
/// transcript challenge.
fn fold_product_mles(
    products: &mut R1csProductMles<FqDefault>,
    challenge: FqDefault,
) -> Result<(), SumcheckError> {
    products
        .az
        .fold(&[challenge])
        .map_err(|_| SumcheckError::InvalidMleOperation)?;
    products
        .bz
        .fold(&[challenge])
        .map_err(|_| SumcheckError::InvalidMleOperation)?;
    products
        .cz
        .fold(&[challenge])
        .map_err(|_| SumcheckError::InvalidMleOperation)?;
    Ok(())
}

fn folded_round_partial(
    eq_scale: FqDefault,
    eq_high: &DenseMultilinearExtension<FqDefault>,
    products: &R1csProductMles<FqDefault>,
) -> [FqDefault; 3] {
    let pair_count = products.az.len() / 2;

    sum_coefficients(pair_count, |pair| {
        let index = 2 * pair;
        cubic_contribution(
            [eq_scale * eq_high[index], eq_scale * eq_high[index + 1]],
            [products.az[index], products.az[index + 1]],
            [products.bz[index], products.bz[index + 1]],
            [products.cz[index], products.cz[index + 1]],
        )
    })
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use transcript::{build_prover, build_verifier};

    use super::*;

    const SESSION: &[u8] = b"spartan/outer-sumcheck/test";

    fn fq(value: u128) -> FqDefault {
        FqDefault::from(value)
    }

    struct OuterSumcheckTestInputs {
        tau: Vec<FqDefault>,
        eq_factors: (
            DenseMultilinearExtension<FqDefault>,
            DenseMultilinearExtension<FqDefault>,
        ),
        products: R1csProductMles<FqDefault>,
    }

    /// Builds random product MLEs and equality factors from `poly::eq_table`.
    fn build_outer_sumcheck_inputs(num_vars: usize) -> OuterSumcheckTestInputs {
        build_outer_sumcheck_inputs_with_split(num_vars, num_vars / 2)
    }

    fn build_outer_sumcheck_inputs_with_split(
        num_vars: usize,
        split: usize,
    ) -> OuterSumcheckTestInputs {
        assert!(num_vars < usize::BITS as usize);
        assert!(split <= num_vars);

        let table_len = 1usize << num_vars;
        let mut rng = Pcg64::seed_from_u64(0x5a17_a11c);
        let az_values: Vec<FqDefault> = (0..table_len).map(|_| rng.random()).collect();
        let bz_values: Vec<FqDefault> = (0..table_len).map(|_| rng.random()).collect();
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
            .map(|index| fq(2 * index as u128 + 2))
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
    fn check_outer_sumcheck(inputs: OuterSumcheckTestInputs) {
        let OuterSumcheckTestInputs {
            tau,
            eq_factors,
            products,
        } = inputs;
        let num_vars = tau.len();
        let instance = (num_vars as u64).to_le_bytes();
        let expected_products = products.clone();

        let mut prover = build_prover(SESSION, &instance);
        let prover_output =
            prove_outer_sumcheck(&mut prover, FqDefault::from(0u128), eq_factors, products)
                .unwrap();
        let proof = prover.finish();

        let mut verifier = build_verifier(SESSION, &instance, &proof);
        let verifier_output =
            verify_outer_sumcheck(&mut verifier, fq(0), &prover_output.proof, &tau).unwrap();
        verifier.check_eof().unwrap();

        assert_eq!(prover_output.eval_points, verifier_output.eval_points);
        assert_eq!(
            prover_output.proof.product_evaluations,
            verifier_output.product_evaluations
        );
        assert_eq!(
            prover_output.proof.sumcheck.round_polynomials.len(),
            num_vars
        );
        assert_eq!(
            prover_output.proof.product_evaluations,
            [
                expected_products
                    .az
                    .evaluate(&prover_output.eval_points)
                    .unwrap(),
                expected_products
                    .bz
                    .evaluate(&prover_output.eval_points)
                    .unwrap(),
                expected_products
                    .cz
                    .evaluate(&prover_output.eval_points)
                    .unwrap(),
            ]
        );
    }

    #[test]
    fn outer_sumcheck_inputs_have_pointwise_products_and_factored_eq() {
        for num_vars in [0, 1, 3, 10] {
            let inputs = build_outer_sumcheck_inputs(num_vars);
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
    fn outer_sumcheck_supports_every_equality_factor_split() {
        for split in 0..=5 {
            check_outer_sumcheck(build_outer_sumcheck_inputs_with_split(5, split));
        }
    }

    #[test]
    fn generic_round_prover_supports_quadratics() {
        let initial_claim = fq(20);
        let mut prover = build_prover(SESSION, b"generic-quadratic-round");
        let mut current_claim = initial_claim;
        let mut round_polynomials = Vec::<[FqDefault; 3]>::new();
        let mut prover_points = Vec::new();

        let prover_challenge = prove_round(
            &mut prover,
            &mut current_claim,
            [fq(3), fq(5)],
            &mut round_polynomials,
            &mut prover_points,
        );

        assert_eq!(round_polynomials, vec![[fq(3), fq(9), fq(5)]]);
        assert_eq!(prover_points, vec![prover_challenge]);

        let transcript_proof = prover.finish();
        let sumcheck = SumcheckProof { round_polynomials };
        let mut verifier = build_verifier(SESSION, b"generic-quadratic-round", &transcript_proof);
        let (verifier_points, final_claim) =
            sumcheck.verify(&mut verifier, initial_claim, 1).unwrap();

        assert_eq!(verifier_points, prover_points);
        assert_eq!(final_claim, current_claim);
        verifier.check_eof().unwrap();
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
                challenge_fq(&mut prover)
            })
            .collect();
        let next_prover_challenge = challenge_fq(&mut prover);
        let transcript_proof = prover.finish();

        let mut verifier = build_verifier(SESSION, instance, &transcript_proof);
        let (verifier_points, final_claim) = sumcheck.verify(&mut verifier, fq(20), 2).unwrap();
        let next_verifier_challenge = challenge_fq(&mut verifier);

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
    fn sumcheck_rejects_a_proof_owned_round_count() {
        let sumcheck = SumcheckProof {
            round_polynomials: vec![[fq(1), fq(0), fq(0), fq(0)]],
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"round-count", &transcript_proof);

        assert_eq!(
            sumcheck.verify(&mut verifier, fq(2), 2),
            Err(SumcheckError::InvalidRoundCount {
                expected: 2,
                actual: 1,
            })
        );
        verifier.check_eof().unwrap();
    }

    #[test]
    fn zero_round_sumcheck_preserves_the_initial_claim() {
        let sumcheck = SumcheckProof::<FqDefault, 4> {
            round_polynomials: vec![],
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"zero-rounds", &transcript_proof);

        assert_eq!(
            sumcheck.verify(&mut verifier, fq(42), 0),
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

    #[test]
    fn fq_challenge_rejects_the_incomplete_tail() {
        struct ScriptedChallenges {
            values: Vec<u128>,
            next: usize,
        }

        impl FqChallengeSource for ScriptedChallenges {
            fn squeeze_u128(&mut self) -> u128 {
                let value = self.values[self.next];
                self.next += 1;
                value
            }
        }

        let mut source = ScriptedChallenges {
            values: vec![MAX_ACCEPTED_CHALLENGE + 1, 7],
            next: 0,
        };

        assert_eq!(challenge_fq(&mut source), fq(7));
        assert_eq!(source.next, 2);
    }

    #[test]
    fn outer_verifier_checks_the_zero_round_terminal_claim() {
        let proof = OuterSumcheckProof {
            sumcheck: SumcheckProof {
                round_polynomials: vec![],
            },
            product_evaluations: [fq(2), fq(3), fq(6)],
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"outer-zero-rounds", &transcript_proof);

        assert_eq!(
            verify_outer_sumcheck(&mut verifier, fq(0), &proof, &[]),
            Ok(OuterSumcheckVerifierOutput {
                eval_points: vec![],
                product_evaluations: proof.product_evaluations,
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
            product_evaluations: [fq(2), fq(3), fq(5)],
        };
        let transcript_proof = transcript::Proof::default();
        let mut verifier = build_verifier(SESSION, b"outer-bad-terminal", &transcript_proof);

        assert_eq!(
            verify_outer_sumcheck(&mut verifier, fq(0), &proof, &[]),
            Err(SumcheckError::InvalidTerminalClaim)
        );
        verifier.check_eof().unwrap();
    }

    #[test]
    fn outer_sumcheck_zero_vars() {
        check_outer_sumcheck(build_outer_sumcheck_inputs(0));
    }

    #[test]
    fn outer_sumcheck_one_var() {
        check_outer_sumcheck(build_outer_sumcheck_inputs(1));
    }

    #[test]
    fn outer_sumcheck_three_vars() {
        check_outer_sumcheck(build_outer_sumcheck_inputs(3));
    }

    #[test]
    fn outer_sumcheck_ten_vars() {
        check_outer_sumcheck(build_outer_sumcheck_inputs(10));
    }
}
