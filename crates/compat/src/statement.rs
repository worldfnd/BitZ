use crate::messages::binary_value;
use field::runtime::PrimeContext;
use flock_core::pcs::ligerito::{LigeritoSecurityConfig, ProverConfig};
use serde_json::{Value, json};
use spartan::reference::{Claim, matrix_digest};
use spartan::reference_messages::Message;
use std::io;
use transcript::reference::{Absorbable, Bytes, Transcript, hex};

pub struct Configuration {
    policy: LigeritoSecurityConfig,
    prover: ProverConfig,
    min: u128,
    max: u128,
}
impl Configuration {
    pub fn policy(&self) -> &LigeritoSecurityConfig {
        &self.policy
    }
    pub fn prover(&self) -> &ProverConfig {
        &self.prover
    }
    pub fn prime_bounds(&self) -> (u128, u128) {
        (self.min, self.max)
    }
    pub fn from_public_statement(s: &Value) -> io::Result<Self> {
        // This milestone deliberately supports one complete public profile.
        // Commitments and binding digests vary by witness and are checked only
        // after independent commitment construction.
        let mut declared = s.clone();
        let object = declared
            .as_object_mut()
            .ok_or_else(|| io::Error::other("statement must be an object"))?;
        let binding = object
            .remove("assignment_binding_hex")
            .ok_or_else(|| io::Error::other("missing assignment binding"))?;
        let public = object
            .get_mut("public_instance")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| io::Error::other("missing public instance"))?;
        let root = public
            .remove("root_hex")
            .ok_or_else(|| io::Error::other("missing root"))?;
        for digest in [&root, &binding] {
            if transcript::reference::unhex(
                digest
                    .as_str()
                    .ok_or_else(|| io::Error::other("digest must be hex"))?,
            )?
            .len()
                != 32
            {
                return Err(io::Error::other("digest must have 32 bytes"));
            }
        }
        let supported: Value = serde_json::from_str(include_str!("supported_statement.json"))?;
        if declared != supported {
            return Err(io::Error::other(
                "unsupported or inconsistent public statement; expected the complete W1 plain-UDR Lambda100 profile",
            ));
        }
        let c = &s["configuration"];
        let policy: LigeritoSecurityConfig =
            serde_json::from_value(c["ligerito"]["configuration"].clone())?;
        policy.validate().map_err(io::Error::other)?;
        let min = 1u128 << 110;
        let max = (1u128 << 111) - 1;
        let (prover, _) = policy
            .to_prover_verifier_configs()
            .map_err(io::Error::other)?;
        Ok(Self {
            policy,
            prover,
            min,
            max,
        })
    }
    /// Reconstruct the public statement from the supported profile and local root.
    pub fn public_statement(&self, root: &[u8; 32]) -> io::Result<Value> {
        let mut s: Value = serde_json::from_str(include_str!("supported_statement.json"))?;
        s["public_instance"]["root_hex"] = json!(hex(root));
        s["assignment_binding_hex"] = json!(hex(&self.assignment_binding(root)));
        Ok(s)
    }
    pub fn policy_digest(&self) -> [u8; 32] {
        *blake3::hash(&bincode::serialize(&self.policy).expect("policy serialization")).as_bytes()
    }
    pub fn assignment_binding(&self, root: &[u8; 32]) -> [u8; 32] {
        let mut v = b"f2z/spartan-u32-mul/assignment/plain/v1".to_vec();
        v.extend_from_slice(root);
        sizes(&mut v, &[22, 1, 4]);
        v.extend_from_slice(&[0, 1]);
        sized_bytes(&mut v, b"lambda100");
        sizes(&mut v, &[100]);
        v.extend_from_slice(&self.min.to_le_bytes());
        v.extend_from_slice(&self.max.to_le_bytes());
        v.push(0);
        sizes(&mut v, &[0, 0, 0]);
        v.push(0);
        sizes(&mut v, &[0, 0, 100]);
        let c = &self.prover;
        sizes(
            &mut v,
            &[
                c.recursive_steps,
                c.initial_log_msg_cols,
                c.initial_log_num_interleaved,
                c.initial_k,
            ],
        );
        for a in [
            &c.log_inv_rates,
            &c.recursive_log_msg_cols,
            &c.recursive_ks,
            &c.queries,
            &c.grinding_bits,
            &c.fold_grinding_bits,
            &c.ood_samples,
        ] {
            sizes(&mut v, &[a.len()]);
            sizes(&mut v, a);
        }
        v.push(1);
        sizes(&mut v, &[32768, 32768, 131072, 15, 15, 7, 1, 0, 3]);
        *blake3::hash(&v).as_bytes()
    }
    pub fn bind(&self, root: &[u8; 32], t: &mut Transcript) -> io::Result<[u8; 32]> {
        let binding = self.assignment_binding(root);
        t.absorb(
            "statement.assignment_binding_digest",
            json!({}),
            &Message {
                tag: b"u32-statement",
                payload: &binding,
            },
        )?;
        t.absorb(
            "statement.ligerito_policy_digest",
            json!({}),
            &Bytes(b"f2z/ligerito-policy/early-ood/v1"),
        )?;
        t.absorb(
            "statement.ligerito_policy_digest",
            json!({}),
            &Bytes(&self.policy_digest()),
        )?;
        Ok(binding)
    }
}
fn sizes(v: &mut Vec<u8>, a: &[usize]) {
    for &n in a {
        v.extend_from_slice(&(n as u64).to_le_bytes());
    }
}
fn sized_bytes(v: &mut Vec<u8>, b: &[u8]) {
    sizes(v, &[b.len()]);
    v.extend_from_slice(b);
}
fn elements(v: &mut Vec<u8>, a: &[u128]) {
    sizes(v, &[a.len()]);
    for n in a {
        v.extend_from_slice(&n.to_le_bytes());
    }
}

pub struct Bitified {
    pub rows: Vec<u128>,
    pub columns: Vec<u128>,
    pub target: u128,
    pub digest: [u8; 32],
}
pub fn bitify(f: &PrimeContext, binding: &[u8; 32], claim: &Claim) -> io::Result<Bitified> {
    if claim.point.len() != 17
        || claim
            .point
            .iter()
            .chain([&claim.scale, &claim.value])
            .any(|&x| x >= f.modulus())
    {
        return Err(io::Error::other("invalid terminal claim"));
    }
    let point = &claim.point;
    let selectors = f.eq_table(&point[15..]);
    let gate_zero = point[..15].iter().fold(1, |a, &r| f.mul(a, f.sub(1, r)));
    let target = f.sub(
        claim.value,
        f.mul(claim.scale, f.mul(selectors[0], gate_zero)),
    );
    let mut factors = [selectors[1], selectors[2], selectors[3]];
    let dummy = factors.iter().all(|&x| x == 0);
    if dummy && target != 0 {
        return Err(io::Error::other("nonzero dummy claim"));
    }
    let column_scale = if dummy || claim.scale == 0 {
        0
    } else {
        if claim.scale != 1 {
            for x in &mut factors {
                *x = f.mul(*x, claim.scale);
            }
        }
        1
    };
    let low = &point[..7];
    let high = &point[7..15];
    let high_eq = f.eq_table(high);
    let mut rows = vec![0; 32768];
    if dummy {
        rows[0] = 1;
    } else {
        for slot in 0..128 {
            let (factor, bit) = if slot < 32 {
                (factors[0], slot)
            } else if slot < 64 {
                (factors[1], slot - 32)
            } else {
                (factors[2], slot - 64)
            };
            for i in 0..256 {
                rows[(slot << 8) | i] = f.mul(f.mul(factor, 1u128 << bit), high_eq[i]);
            }
        }
    }
    let columns = f
        .eq_table(low)
        .into_iter()
        .map(|x| f.mul(x, column_scale))
        .collect();
    let mut v = b"f2z/spartan-f2z/bitified-claim/v3".to_vec();
    v.extend_from_slice(binding);
    sized_bytes(&mut v, &f.modulus().to_le_bytes());
    v.extend_from_slice(&matrix_digest(f.modulus()));
    v.extend_from_slice(&f.modulus().to_le_bytes());
    sizes(
        &mut v,
        &[32768, 32768, 131072, 15, 15, 7, 1, 0, 32, 32, 32, 64, 64],
    );
    v.extend_from_slice(&[2, 0, 0, 1, 2, 3]);
    elements(&mut v, point);
    v.extend_from_slice(&claim.scale.to_le_bytes());
    v.extend_from_slice(&claim.value.to_le_bytes());
    elements(&mut v, low);
    elements(&mut v, high);
    v.push(dummy as u8);
    if !dummy {
        for x in factors {
            v.extend_from_slice(&x.to_le_bytes());
        }
    }
    v.extend_from_slice(&column_scale.to_le_bytes());
    v.extend_from_slice(&target.to_le_bytes());
    Ok(Bitified {
        rows,
        columns,
        target,
        digest: *blake3::hash(&v).as_bytes(),
    })
}

pub struct OpeningStatement<'a> {
    pub root: &'a [u8; 32],
    pub config: &'a ProverConfig,
    pub digest: &'a [u8; 32],
    pub bits: usize,
}
enum Field<'a> {
    Bytes(&'a [u8]),
    Byte(u8),
    Size(usize),
    Sizes(&'a [usize]),
    Generator,
}
impl Field<'_> {
    fn emit(&self, tag: u8, e: &mut dyn FnMut(&[u8])) {
        let (kind, count) = match self {
            Self::Bytes(v) => (1, v.len()),
            Self::Byte(_) => (2, 1),
            Self::Size(_) => (3, 1),
            Self::Sizes(v) => (3, v.len()),
            Self::Generator => (5, 1),
        };
        e(&[6]);
        e(&[tag, kind]);
        e(&(count as u64).to_le_bytes());
        match self {
            Self::Bytes(v) => e(v),
            Self::Byte(v) => e(&[*v]),
            Self::Size(v) => e(&(*v as u64).to_le_bytes()),
            Self::Sizes(v) => {
                for n in *v {
                    e(&(*n as u64).to_le_bytes());
                }
            }
            Self::Generator => {
                e(&2u64.to_le_bytes());
                e(&0u64.to_le_bytes());
            }
        }
        e(&[7]);
    }
}
impl Absorbable for OpeningStatement<'_> {
    fn visit_chunks(&self, e: &mut dyn FnMut(&[u8])) {
        transcript::reference::Bytes(b"f2z/ligerito-flock/statement-frame/v1").visit_chunks(e);
        transcript::reference::Bytes(b"f2z/spartan-f2z/u32-mod-q-opening/v2").visit_chunks(e);
        let c = self.config;
        use Field::*;
        for (tag, v) in [
            (1, Bytes(self.root)),
            (2, Size(22)),
            (3, Size(1)),
            (4, Size(4)),
            (5, Byte(0)),
            (6, Byte(1)),
            (8, Sizes(&c.log_inv_rates)),
            (9, Size(c.recursive_steps)),
            (10, Size(c.initial_log_msg_cols)),
            (11, Size(c.initial_log_num_interleaved)),
            (12, Size(c.initial_k)),
            (13, Sizes(&c.recursive_log_msg_cols)),
            (14, Sizes(&c.recursive_ks)),
            (15, Sizes(&c.queries)),
            (16, Sizes(&c.grinding_bits)),
            (17, Sizes(&c.fold_grinding_bits)),
            (18, Sizes(&c.ood_samples)),
            (19, Byte(1)),
            (32, Size(15)),
            (33, Size(7)),
            (34, Size(1)),
            (48, Bytes(self.digest)),
            (49, Size(self.bits)),
            (50, Generator),
        ] {
            v.emit(tag, e);
        }
    }
    fn value(&self) -> Value {
        let c = self.config;
        json!({"domains":{"frame":"f2z/ligerito-flock/statement-frame/v1","protocol":"f2z/spartan-f2z/u32-mod-q-opening/v2"},"commitment":{"root_hex":hex(self.root),"variables":22,"log_inverse_rate":1,"log_batch_size":4,"profile_code":0,"merkle_hash":"blake3"},"ligerito_configuration":{"log_inverse_rates":c.log_inv_rates,"recursive_steps":c.recursive_steps,"initial":{"log_message_columns":c.initial_log_msg_cols,"log_interleaving":c.initial_log_num_interleaved,"fold_count":c.initial_k},"recursive_log_message_columns":c.recursive_log_msg_cols,"recursive_fold_counts":c.recursive_ks,"query_counts":c.queries,"query_grinding_bits":c.grinding_bits,"fold_grinding_bits":c.fold_grinding_bits,"ood_sample_counts":c.ood_samples,"merkle_hash":"blake3"},"layout":{"row_variables":15,"column_variables":7,"word_bits":1},"opening_claim_digest_hex":hex(self.digest),"modulus_bit_length":self.bits,"generator":binary_value(2)})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn statement() -> Value {
        let mut s: Value = serde_json::from_str(include_str!("supported_statement.json")).unwrap();
        s["assignment_binding_hex"] = json!("00".repeat(32));
        s["public_instance"]["root_hex"] = json!("00".repeat(32));
        s
    }
    #[test]
    fn rejects_changed_public_profile_fields() {
        let s = statement();
        Configuration::from_public_statement(&s).unwrap();
        for (path, value) in [
            (
                "/public_instance/commitment_parameters/log_inv_rate",
                json!(2),
            ),
            ("/public_instance/commitment_parameters/m", json!(23)),
            ("/dimensions/assignment_len", json!(262144)),
            ("/configuration/transcript", json!("SHA256")),
            ("/configuration/piop_skip_degree", json!(4)),
            ("/configuration/generator_words", json!([3, 0])),
            ("/configuration/security/ligerito_target_bits", json!(90)),
            (
                "/configuration/resolved_ligerito_parameters/initial_k",
                json!(3),
            ),
            (
                "/configuration/ligerito/configuration_fingerprint",
                json!("00".repeat(32)),
            ),
            (
                "/configuration/ligerito/configuration/levels/0/queries",
                json!(1),
            ),
            ("/relation/N", json!(32767)),
            ("/diagnostic_metadata_is_absorbed", json!(true)),
            ("/configuration/security/projection_min", json!("3")),
        ] {
            let mut bad = s.clone();
            *bad.pointer_mut(path).unwrap() = value;
            assert!(
                Configuration::from_public_statement(&bad).is_err(),
                "{path}"
            );
        }
        let mut bad = s.clone();
        bad["unknown_protocol_setting"] = json!(true);
        assert!(Configuration::from_public_statement(&bad).is_err());
        let mut bad = s;
        bad["assignment_binding_hex"] = json!("wrong");
        assert!(Configuration::from_public_statement(&bad).is_err());
    }
}
