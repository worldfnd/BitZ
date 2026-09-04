//! Verifying a claim about a vector that was never committed.
//!
//! The mirror of the prover's virtual path. `M` is public, so the verifier
//! runs the same transposition rather than being told its result.

use common::{LinearClaim, Root, VirtualMap, VirtualMapError, VirtualParams, transpose_query};
use pcs::{CommitScheme, Pcs, StatementBinding};
use transcript::VerifierState;

use crate::{F2ZVerifier, Reduction, verify::VerifyError};

/// A virtual proof the verifier rejects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VirtualVerifyError<E> {
    /// Steps 2 to 5 failed.
    Core(VerifyError<E>),
    /// The transposition rejected the reduction's claim.
    Map(VirtualMapError),
    /// The opening did not discharge the transposed claim.
    Opening(pcs::CommitError),
    /// A stream held bytes the protocol never read.
    TrailingData,
}

/// The verifier for the virtual path, carrying both shapes.
#[derive(Debug)]
pub struct VirtualVerifier<const Q: u128> {
    params: VirtualParams<Q>,
    core: F2ZVerifier<Q>,
}

impl<const Q: u128> VirtualVerifier<Q> {
    /// # Panics
    ///
    /// [`F2ZVerifier::new`] requires `window` in `1..=16`.
    pub fn new(params: VirtualParams<Q>, window: u32) -> Self {
        let core = F2ZVerifier::new(*params.claim(), window);
        Self { params, core }
    }

    pub fn params(&self) -> &VirtualParams<Q> {
        &self.params
    }

    /// Replays the proof of the caller's linear claim about `h`.
    ///
    /// `map` is public input. Handing a different one than the prover used
    /// changes the frame, so the challenge stream diverges immediately.
    pub fn verify<R: Reduction<Q>>(
        &self,
        map: &impl VirtualMap,
        claim: &LinearClaim<Q>,
        pcs: &Pcs,
        com: Root,
        reduction: &R,
        mut transcript: VerifierState<'_>,
    ) -> Result<(), VirtualVerifyError<R::Error>> {
        // Step 1, replayed.
        transcript.public_message(&com.0);
        transcript.public_message(&self.params);
        transcript.public_message(&map.digest());

        // Steps 2 to 5, replayed against `h`.
        let query = {
            let _guard = prof::scope("f2z/fold-and-reduce");
            self.core
                .fold_and_reduce(claim, com, reduction, &mut transcript)
                .map_err(VirtualVerifyError::Core)?
        };

        let committed_bits = 1usize << self.params.committed_shape().log_bits();
        let query = transpose_query(map, committed_bits, query).map_err(VirtualVerifyError::Map)?;

        // Step 6, replayed against `f`.
        let _guard = prof::scope("f2z/open");
        pcs.verify_lin(&com, &query, StatementBinding::Bind, &mut transcript)
            .map_err(VirtualVerifyError::Opening)?;

        transcript
            .check_eof()
            .map_err(|_| VirtualVerifyError::TrailingData)?;

        Ok(())
    }
}
