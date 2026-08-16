//! The verifier replays inside worker threads, so both states must be `Send`.

use transcript::{Proof, ProverState, VerifierState};

fn assert_send<T: Send>() {}

#[test]
fn transcript_states_are_send() {
    assert_send::<ProverState>();
    assert_send::<VerifierState<'static>>();
    assert_send::<Proof>();
}
