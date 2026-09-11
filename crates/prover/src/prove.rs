//! `ProveBitZ`.

use common::{BitTable, LinearClaim, OpeningQuery, ReductionInput, Root, TableError};
use field::{F128, Fq};
use pcs::{CommitScheme, Pcs, ProveError as OpeningProveError, ProverData, StatementBinding};

use common::Fold;
use gkr::{GrandProductCircuit, gpgkr_prove};
use num_traits::{ConstOne, ConstZero, identities::Zero};
use poly::eq_table;
use transcript::ProverState;

use crate::{BitZProver, SendError};

/// A proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError<E> {
    /// The witness is not the length the shape calls for.
    Witness(TableError),
    /// The fold round failed.
    Fold(SendError),
    /// The reduction failed.
    Reduction(E),
    /// The opening failed, so the reduction's claim was never discharged.
    Opening(OpeningProveError),
}

/// Step 4: the grand product, and the sumcheck that turns its affine leaf
/// into a claim on the committed bits.
///
/// TODO: #8 implements this. Until then the round trip stubs it, so the
/// transcript order below is exercised and the reduction's argument is not.
///
/// Xander: This trait only for mock testing?
pub trait Reduction<const Q: u128> {
    type Error;

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
        // Don't think this is still necessary.
    ) -> Result<OpeningQuery, Self::Error>;
}

/// Stands in for #8: runs the grand-product circuit and its sumcheck, but
/// does not yet turn the resulting claim into a discharged [`OpeningQuery`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Reduce;

impl<const Q: u128> Reduction<Q> for Reduce {
    type Error = ();

    fn reduce(
        &self,
        input: &ReductionInput<'_, Q>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
    ) -> Result<OpeningQuery, Self::Error> {
        let fold = input.fold;

        let (_u1, _m, _claim) = gkr_reduce(transcript, fold, table);

        // turn the sumcheck's output claim into a discharged `OpeningQuery`.
        todo!(
            "u1, u2 and claim into a linear claim. Requires changing LinearClaim::new to take Vec<F128> instead of Vec<Fq<Q>>"
        );
    }
}

fn init_circuit(table: &BitTable, fold: &Fold) -> GrandProductCircuit {
    let dim = table.shape().columns() * table.shape().rows();
    let mut leafs: Vec<_> = vec![F128::zero(); dim];

    // TODO optimisation: Handle the leafs and the two layers above it lazily.
    for b in 0..table.shape().rows() {
        for c in 0..table.shape().columns() {
            leafs[b * table.shape().columns() + c] = if table.bit(c, b) {
                fold.row_images[b]
            } else {
                F128::ONE
            };
        }
    }

    GrandProductCircuit::new(leafs)
}

pub fn gkr_reduce(
    transcript: &mut ProverState,
    fold: &Fold,
    table: &BitTable,
) -> (Vec<F128>, Vec<F128>, F128) {
    let circuit = init_circuit(table, fold);
    let (_last_value, witnesses) = circuit.batched_eval(table.shape().columns());

    let (mut point, claim) = gpgkr_prove(transcript, &fold.zeta, witnesses);

    let inner_product_claim = claim - F128::ONE;

    let r1 = fold.row_images.len().max(1).ilog2();
    let r2 = point.len() - r1 as usize;
    let alfa_b = point.split_off(r2);
    let alfa_c = point;

    let u1: Vec<_> = fold
        .row_images
        .iter()
        .zip(poly::eq_table(&alfa_b))
        .map(|(a, b)| (*a - F128::ONE) * b) // Does the later step benefit from wide mul?
        .collect();

    let u2 = eq_table(&alfa_c);
    todo!("Move m_table into pcs::lin and return u2");

    // Inner product / eq_table approach to prevent blowup
    fn m_table(u2: Vec<F128>, table: &BitTable) -> Vec<F128> {
        let columns = table.shape().columns();
        let rows = table.shape().rows();
        debug_assert_eq!(u2.len(), columns);

        let mut m = vec![F128::ZERO; rows];
        for (j, factor) in u2.iter().enumerate() {
            for (i, b) in table.column_bits(j).enumerate() {
                if b {
                    m[i] += factor;
                }
            }
        }

        m
    }

    let m = m_table(u2, table);

    (u1, m, inner_product_claim)
}

impl<const Q: u128> BitZProver<Q> {
    /// Proves the caller's linear claim about the committed bits.
    ///
    /// The caller commits first and passes what that produced: the `data` the
    /// opening reads and the packed witness itself. The root is read back off
    /// `data` rather than passed alongside it, so the two cannot disagree.
    /// `pcs` must be the scheme that committed, or the opening will not verify.
    ///
    /// The witness arrives owned because the opening consumes it. The transcript
    /// arrives carrying the caller's events; this appends and hands it back.
    pub fn prove<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<Fq<Q>>,
        pcs: &Pcs,
        data: &ProverData,
        packed: Vec<F128>,
        reduction: &R,
        transcript: &mut ProverState,
    ) -> Result<(), ProveError<R::Error>> {
        let com = data.root();
        let table = self.params().table(&packed).map_err(ProveError::Witness)?;

        // Step 1: bind. Absorbing the root here is not redundant with the opening
        // scheme, whose batched opening binds it only in its own statement mode --
        // and that fires at step 6, long after the fold has squeezed.
        //
        // The claim itself is not bound: neither `v^(1)`, `v^(2)` nor `mu` reaches
        // the sponge here, only the parameters. They enter through the caller's
        // own events.
        transcript.public_message(&com.0);
        transcript.public_message(self.params());

        // Steps 2 to 5: the modulus reduction, the column fold, the grand
        // product, and the batching a merged forest does not need.
        let query = self.fold_and_reduce(claim, com, &table, reduction, transcript)?;

        // Step 6: the ring switch and the opening, which discharge the claim
        // step 4 handed over. `Bind` rather than `AlreadyBound`: the opening's
        // own parameters are not in the frame step 1 absorbed, and binding them
        // here is what puts them in the sponge before the opener's first
        // squeeze.
        pcs.prove_lin(data, packed, &query, StatementBinding::Bind, transcript)
            .map_err(ProveError::Opening)
    }

    /// Steps 2 to 5, which every entry point runs identically.
    ///
    /// Step 1 and step 6 stay with the caller: it absorbs its own statement
    /// frame, shapes its own table, and decides what to do with the query,
    /// which is a claim about whatever `table` holds.
    pub(crate) fn fold_and_reduce<R: Reduction<Q>>(
        &self,
        claim: &LinearClaim<field::Fq<Q>>,
        com: Root,
        table: &BitTable<'_>,
        reduction: &R,
        transcript: &mut ProverState,
    ) -> Result<OpeningQuery, ProveError<R::Error>> {
        // TODO: step 2, reducing the modulus, is absent. It runs when q is too
        // large for the shape, and a const modulus parameter cannot express its
        // `q <- q'`. Callers must supply an admissible q; LinearClaim::new
        // rejects anything else.

        // Step 3: fold each column into an integer exponent.
        let fold = self
            .send_fold(claim, table, transcript)
            .map_err(ProveError::Fold)?;

        // Step 4: the grand product over the folds, then the sumcheck that
        // turns its affine leaf into a claim on the committed bits.
        let input = ReductionInput {
            params: self.params(),
            claim,
            commitment: com,
            fold: &fold,
        };

        // Step 5, batching the per-column claims, is conditional and a
        // merged-forest grand product does not need it: it draws its challenge
        // once across all columns, so the claims never separate.
        reduction
            .reduce(&input, table, transcript)
            .map_err(ProveError::Reduction)
    }
}

#[cfg(test)]
mod order_check_ai_test {
    //! AI generated
    //! Checks that `gkr_reduce`'s leaf construction and
    //! its later alfa_b/alfa_c/u1/m extraction agree with each other, by
    //! verifying the identity <u1,m> == inner_product_claim on concrete data
    //! with high entropy in both the row and column dimension (so a swapped
    //! or misordered point would break it with overwhelming probability).
    use super::*;
    use common::{BitZParams, Fold, Shape};
    use field::gf128::smallest_generator;

    const Q: u128 = (1 << 114) - 11;

    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    fn params() -> BitZParams<Q> {
        BitZParams::new(shape(), smallest_generator()).unwrap()
    }

    fn packed_witness(shape: &Shape, bit_fn: impl Fn(usize, usize) -> bool) -> Vec<F128> {
        let mut packed = vec![F128::ZERO; (1 << shape.log_bits()) / 128];
        for column in 0..shape.columns() {
            for row in 0..shape.rows() {
                if bit_fn(column, row) {
                    let index = (column << shape.log_rows()) | row;
                    let element = &mut packed[index >> 7];
                    let offset = index % 128;
                    if offset < 64 {
                        element.lo |= 1u64 << offset;
                    } else {
                        element.hi |= 1u64 << (offset - 64);
                    }
                }
            }
        }
        packed
    }

    #[test]
    fn u1_dot_m_matches_the_circuits_own_claim() {
        let shape = shape();

        // High-entropy bit pattern mixing row and column, so a b/c swap or a
        // reversed/misplaced point coordinate breaks the identity below.
        let packed = packed_witness(&shape, |c, b| {
            let h =
                (b as u64).wrapping_mul(2654435761) ^ (c as u64).wrapping_mul(0x9E3779B97F4A7C15);
            (h >> 5) & 1 == 1
        });
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        let images = vec![F128::ONE; shape.columns()];
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();

        let mut prover = transcript::build_prover("order-check", &F128::ZERO);
        let (u1, m, inner_product_claim) = gkr_reduce(&mut prover, &fold, &table);

        let dot: F128 = u1
            .iter()
            .zip(m.iter())
            .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);

        assert_eq!(
            dot, inner_product_claim,
            "<u1,m> did not match the circuit's own inner_product_claim -- \
             row/column order is mixed somewhere between the leaf construction \
             and the u1/m extraction"
        );
    }

    /// Isolates the row/`b` pathway: every column has the identical bit
    /// pattern (a function of `b` alone), so every column's grand product is
    /// the same value and the column/`c` dimension drops out of the picture
    /// entirely. If this still fails, the bug is in the b-pathway (leaf
    /// construction, circuit reduction, or u1/alfa_b) independent of c.
    #[test]
    fn b_only_pattern_isolates_the_row_pathway() {
        let shape = shape();

        let packed = packed_witness(&shape, |_c, b| (b * 2654435761) >> 5 & 1 == 1);
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        let images = vec![F128::ONE; shape.columns()];
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();

        let mut prover = transcript::build_prover("order-check-b", &F128::ZERO);
        let (u1, m, inner_product_claim) = gkr_reduce(&mut prover, &fold, &table);

        let dot: F128 = u1
            .iter()
            .zip(m.iter())
            .fold(F128::ZERO, |acc, (&a, &b)| acc + a * b);

        assert_eq!(
            dot, inner_product_claim,
            "row-only pathway is already broken"
        );
    }
}
