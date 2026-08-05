/// A finished transcript.
///
/// `narg_string` is what the sponge saw and the proof size that matters;
/// `hints` bypassed it. Host serialization of the pair lives elsewhere.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Proof {
    pub narg_string: Vec<u8>,
    pub hints: Vec<u8>,
}
