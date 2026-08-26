use spongefish::{DomainSeparator, Encoding, protocol_id};

use crate::{Proof, ProverState, VerifierState};

/// What this protocol is. Changing it invalidates every existing proof.
pub const PROTOCOL_LABEL: &str = "f2z/v1";

/// Starts a prover transcript.
///
/// The session says where the protocol runs, the instance what is being
/// proven; both are absorbed as domain tags before the first message and
/// stay out of the narg string.
pub fn build_prover<S, I>(session: &S, instance: &I) -> ProverState
where
    S: Encoding<[u8]> + ?Sized,
    I: Encoding<[u8]> + ?Sized,
{
    let inner = DomainSeparator::new(protocol_id(format_args!("{PROTOCOL_LABEL}")))
        .session(session)
        .instance(instance)
        .std_prover();
    ProverState {
        inner,
        hints: Vec::new(),
        narg_records: 0,
        hint_records: 0,
    }
}

/// Starts a verifier transcript over a proof, with the same tags.
///
/// The proof's declared record counts are deliberately not read here: they
/// are transport metadata, and comparing them against the replay is the
/// caller's step once the replay is over.
pub fn build_verifier<'a, S, I>(session: &S, instance: &I, proof: &'a Proof) -> VerifierState<'a>
where
    S: Encoding<[u8]> + ?Sized,
    I: Encoding<[u8]> + ?Sized,
{
    let inner = DomainSeparator::new(protocol_id(format_args!("{PROTOCOL_LABEL}")))
        .session(session)
        .instance(instance)
        .std_verifier(&proof.narg_string);
    VerifierState {
        inner,
        hints: &proof.hints,
        narg_records: 0,
        hint_records: 0,
    }
}
