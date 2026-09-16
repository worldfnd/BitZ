//! Protocol-owned encodings for the F2Z Spartan profile.
use serde_json::{Value, json};
use transcript::reference::{Absorbable, Bytes, hex, number};
pub struct Message<'a> {
    pub tag: &'a [u8],
    pub payload: &'a [u8],
}
impl Absorbable for Message<'_> {
    fn visit_chunks(&self, emit: &mut dyn FnMut(&[u8])) {
        let mut v = b"f2z/spartan/transcript-frame/v1".to_vec();
        v.extend_from_slice(&(self.tag.len() as u64).to_le_bytes());
        v.extend_from_slice(self.tag);
        v.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        v.extend_from_slice(self.payload);
        Bytes(&v).visit_chunks(emit);
    }
    fn value(&self) -> Value {
        json!({"tag":String::from_utf8_lossy(self.tag),"bytes_hex":hex(self.payload)})
    }
}
pub struct Fields<'a> {
    pub values: &'a [u128],
    pub modulus: u128,
}
impl Absorbable for Fields<'_> {
    fn visit_chunks(&self, emit: &mut dyn FnMut(&[u8])) {
        let mut v = (self.values.len() as u64).to_le_bytes().to_vec();
        for x in self.values {
            v.extend_from_slice(&16u64.to_le_bytes());
            v.extend_from_slice(&x.to_le_bytes());
        }
        Message {
            tag: b"field-elements",
            payload: &v,
        }
        .visit_chunks(emit);
    }
    fn value(&self) -> Value {
        json!({"field":"prime","modulus_hex":number(self.modulus),"values_hex":self.values.iter().copied().map(number).collect::<Vec<_>>()})
    }
}
