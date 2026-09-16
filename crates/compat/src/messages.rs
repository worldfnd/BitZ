//! Encodings and sampling for the deferred binary opening protocol.
use serde_json::{Value, json};
use std::io;
use transcript::reference::{Absorbable, ByteTranscript, Transcript, number};
pub fn binary_value(v: u128) -> Value {
    json!({"field":"GF(2^128)","basis":"polynomial","value_hex":number(v)})
}
/// Existing binary-field frames, without combining their hash updates.
pub struct BinaryFields<'a>(pub &'a [u128]);
impl Absorbable for BinaryFields<'_> {
    fn visit_chunks(&self, e: &mut dyn FnMut(&[u8])) {
        for v in self.0 {
            e(&[3]);
            e(&0x87u128.to_le_bytes());
            e(&[5]);
            e(&[1]);
            e(&v.to_le_bytes());
            e(&[3]);
        }
    }
    fn value(&self) -> Value {
        json!({"values":self.0.iter().copied().map(binary_value).collect::<Vec<_>>()})
    }
}

pub fn binary(t: &mut Transcript, purpose: &str, ctx: Value, feedback: bool) -> io::Result<u128> {
    t.squeeze(purpose, ctx, |t| {
        let mut bytes = [0; 16];
        t.challenge_bytes(&mut bytes)?;
        let v = u128::from_le_bytes(bytes);
        if feedback {
            t.absorb_object(&BinaryFields(&[v]))?;
        }
        Ok((v, binary_value(v)))
    })
}
