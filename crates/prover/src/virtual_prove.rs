//! Proving a claim about a vector that was never committed.
//!
//! The claim arrives about `h`, the oracle holds `f`, and a public 0/1 map
//! relates them by `h = M (1 ‖ f)`. Steps 2 to 5 never read the oracle, so
//! they run against `h` exactly as the plain path runs against `f`. Step 6
//! does read it, so the claim is rewritten first:
//!
//! ```text
//! <v, h> = <v, M (1 || f)> = <M^T v, (1 || f)>.
//! ```
//!
//! `M^T v` is not an equality weight for any point, so the opening it goes to
//! is the inner-product one rather than the evaluation one.

use common::{
    LinearClaim, TableError, VirtualMap, VirtualMapError, VirtualParams, transpose_query,
};
use field::F128;
use pcs::{CommitError, CommitScheme, Pcs, ProverData, StatementBinding};
use transcript::ProverState;

use crate::{F2ZProver, Reduction, prove::ProveError};

/// A virtual proof the prover cannot produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VirtualProveError<E> {
    /// The integer witness is not the length the claim's shape calls for.
    IntegerWitness(TableError),
    /// Steps 2 to 5 failed.
    Core(ProveError<E>),
    /// The transposition rejected the reduction's claim.
    Map(VirtualMapError),
    /// The opening failed, so the transposed claim was never discharged.
    Opening(CommitError),
}

/// The prover for the virtual path, carrying both shapes.
///
/// Wrapping the core prover rather than taking its parameters alongside is
/// what stops the two from naming different shapes.
#[derive(Debug)]
pub struct VirtualProver<const Q: u128> {
    params: VirtualParams<Q>,
    core: F2ZProver<Q>,
}

impl<const Q: u128> VirtualProver<Q> {
    /// # Panics
    ///
    /// [`F2ZProver::new`] requires `window` in `1..=16`.
    pub fn new(params: VirtualParams<Q>, window: u32) -> Self {
        let core = F2ZProver::new(*params.claim(), window);
        Self { params, core }
    }

    pub fn params(&self) -> &VirtualParams<Q> {
        &self.params
    }

    /// Proves the caller's linear claim about `h`, opening against `f`.
    ///
    /// `integer_packed` is `h` in the claim's shape and is only read;
    /// `committed_packed` is `f` in the committed shape and the opening
    /// consumes it. `map` must be the one `data` was committed under: nothing
    /// here can tell, which is why its digest goes into the frame.
    #[allow(clippy::too_many_arguments)]
    pub fn prove<R: Reduction<Q>>(
        &self,
        map: &impl VirtualMap,
        claim: &LinearClaim<Q>,
        pcs: &Pcs,
        data: &ProverData,
        integer_packed: &[F128],
        committed_packed: Vec<F128>,
        reduction: &R,
        transcript: &mut ProverState,
    ) -> Result<(), VirtualProveError<R::Error>> {
        let com = data.root();
        let table = self
            .core
            .params()
            .table(integer_packed)
            .map_err(VirtualProveError::IntegerWitness)?;

        // Step 1. The frame is the virtual one, which encodes to a different
        // width than the plain parameters, and the map rides with it: two runs
        // over the same shapes but different maps are different statements.
        transcript.public_message(&com.0);
        transcript.public_message(&self.params);
        transcript.public_message(&map.digest());

        // Steps 2 to 5, against `h`.
        let query = {
            let _guard = prof::scope("f2z/fold-and-reduce");
            self.core
                .fold_and_reduce(claim, com, &table, reduction, transcript)
                .map_err(VirtualProveError::Core)?
        };

        let committed_bits = 1usize << self.params.committed_shape().log_bits();
        let query = {
            let _guard = prof::scope("f2z/transpose");
            transpose_query(map, committed_bits, query).map_err(VirtualProveError::Map)?
        };

        // Step 6, against `f`.
        let _guard = prof::scope("f2z/open");
        pcs.prove_lin(
            data,
            committed_packed,
            &query,
            StatementBinding::Bind,
            transcript,
        )
        .map_err(VirtualProveError::Opening)
    }
}
