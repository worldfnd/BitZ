/// A finished transcript. Host serialization of the pair lives in `host`.
///
/// A hint is prover data that an already-absorbed message pins down. The
/// verifier usually cannot derive it — a Merkle sibling, an inverse `b` of
/// an absorbed `a` — but it can check it, against the root or by
/// `a * b == 1`. Absorbing it would add nothing the sponge does not already
/// bind, and absorption is not free: for a Merkle path it costs about what
/// verifying the path costs. So hints ride beside the narg string, and the
/// check has to land before the next challenge is squeezed, or a forged
/// hint would reach the sponge unchallenged. A value nothing absorbed pins
/// down is a `prover_message`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Proof {
    pub narg_string: Vec<u8>,
    pub hints: Vec<u8>,
}
