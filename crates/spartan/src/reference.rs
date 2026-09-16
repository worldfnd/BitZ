//! Ordinary U32 Spartan for the opt-in F2Z transcript profile.
use crypto_bigint::{Odd, U128};
use crypto_primes::{Flavor, hazmat::MillerRabin, is_prime};
use field::{F128, runtime::PrimeContext};
use serde_json::{Value, json};
use std::io;
use transcript::reference::{Absorbable, ByteTranscript, Described, Transcript, hex, number};

pub use crate::reference_messages::{Fields, Message};

pub const ROWS: usize = 1 << 15;
/// Products and committed bits can only be derived from the same U32 operands.
pub struct Witness {
    x: Vec<u128>,
    y: Vec<u128>,
    product: Vec<u128>,
    packed: Vec<F128>,
}
impl Witness {
    pub fn deterministic() -> Self {
        Self::from_operands((0..ROWS).map(|i| {
            (
                (i as u32).wrapping_mul(0x9e3779b9) | 1,
                (i as u32).wrapping_mul(0x85ebca6b) | 1,
            )
        }))
        .expect("fixed fixture dimensions")
    }
    pub fn from_operands(operands: impl IntoIterator<Item = (u32, u32)>) -> io::Result<Self> {
        let mut x = Vec::with_capacity(ROWS);
        let mut y = Vec::with_capacity(ROWS);
        let mut product = Vec::with_capacity(ROWS);
        let mut packed = vec![F128::default(); ROWS];
        for (i, (a, b)) in operands.into_iter().enumerate() {
            if i >= ROWS {
                return Err(io::Error::other("too many U32 operands"));
            }
            let p = u64::from(a) * u64::from(b);
            x.push(a as u128);
            y.push(b as u128);
            product.push(p as u128);
            let bits = a as u128 | ((b as u128) << 32) | ((p as u128) << 64);
            for slot in 0..128 {
                if bits >> slot & 1 != 0 {
                    let bit = ((i & 127) << 15) | (slot << 8) | (i >> 7);
                    let word = &mut packed[bit >> 7];
                    if bit & 127 < 64 {
                        word.lo |= 1 << (bit & 63);
                    } else {
                        word.hi |= 1 << (bit & 63);
                    }
                }
            }
        }
        if x.len() != ROWS {
            return Err(io::Error::other("expected 32768 U32 operand pairs"));
        }
        Ok(Self {
            x,
            y,
            product,
            packed,
        })
    }
    pub fn packed(&self) -> &[F128] {
        &self.packed
    }
    fn assignment(&self) -> Vec<u128> {
        let mut v = vec![0; 4 * ROWS];
        v[0] = 1;
        v[ROWS..2 * ROWS].copy_from_slice(&self.x);
        v[2 * ROWS..3 * ROWS].copy_from_slice(&self.y);
        v[3 * ROWS..].copy_from_slice(&self.product);
        v
    }
    /// Direct Boolean-hypercube evaluation, independent of sumcheck folding.
    pub fn check_claim(&self, f: &PrimeContext, c: &Claim) -> io::Result<()> {
        if f.modulus() <= u64::MAX as u128
            || c.point.len() != 17
            || c.point
                .iter()
                .chain([&c.scale, &c.value])
                .any(|&v| v >= f.modulus())
        {
            return Err(io::Error::other("invalid assignment claim"));
        }
        let mut value = 0;
        for index in 0..4 * ROWS {
            let a = match index / ROWS {
                0 => u128::from(index == 0),
                1 => self.x[index % ROWS],
                2 => self.y[index % ROWS],
                _ => self.product[index % ROWS],
            };
            if a == 0 {
                continue;
            }
            let mut weight = 1;
            for (bit, &r) in c.point.iter().enumerate() {
                weight = f.mul(
                    weight,
                    if index >> bit & 1 == 1 {
                        r
                    } else {
                        f.sub(1, r)
                    },
                );
            }
            value = f.add(value, f.mul(a, weight));
        }
        if f.mul(c.scale, value) != c.value {
            return Err(io::Error::other("terminal assignment claim failed"));
        }
        Ok(())
    }
}

pub fn draw_u128(t: &mut impl ByteTranscript) -> io::Result<u128> {
    let mut b = [0; 16];
    t.challenge_bytes(&mut b)?;
    Ok(u128::from_le_bytes(b))
}
pub fn uniform_below(t: &mut impl ByteTranscript, n: u128) -> io::Result<u128> {
    if n == 0 {
        return Err(io::Error::other("empty sampling interval"));
    }
    let cutoff = n.wrapping_neg() % n;
    for _ in 0..256 {
        let x = draw_u128(t)?;
        if x >= cutoff {
            return Ok(x % n);
        }
    }
    Err(io::Error::other("uniform rejection limit"))
}
pub fn sample_field(t: &mut Transcript, purpose: &str, ctx: Value, q: u128) -> io::Result<u128> {
    t.squeeze(purpose, ctx, |t| {
        let v = sample_field_value(t, q)?;
        Ok((
            v,
            json!({"field":"prime","modulus_hex":number(q),"value_hex":number(v)}),
        ))
    })
}
/// Samples and feeds back a canonical element using the protocol's exact framing.
pub fn sample_field_value(t: &mut impl ByteTranscript, q: u128) -> io::Result<u128> {
    if q < 3 || q >= 1 << 126 {
        return Err(io::Error::other("unsupported field modulus"));
    }
    let limit = u128::MAX - ((u128::MAX % q + 1) % q);
    for _ in 0..256 {
        let draw = draw_u128(t)?;
        if draw <= limit {
            let v = draw % q;
            t.absorb_object(&Fields {
                values: &[v],
                modulus: q,
            })?;
            return Ok(v);
        }
    }
    Err(io::Error::other("field rejection limit"))
}
pub fn matrix_digest(q: u128) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"f2z/spartan/constraint-matrices/v2");
    h.update(&16u64.to_le_bytes());
    h.update(&q.to_le_bytes());
    h.update(&(ROWS as u64).to_le_bytes());
    h.update(&((4 * ROWS) as u64).to_le_bytes());
    for (block, label) in [(1, b'A'), (2, b'B'), (3, b'C')] {
        h.update(&[label]);
        for row in 0..ROWS {
            h.update(&1u64.to_le_bytes());
            h.update(&((block * ROWS + row) as u64).to_le_bytes());
            h.update(&16u64.to_le_bytes());
            h.update(&1u128.to_le_bytes());
        }
    }
    *h.finalize().as_bytes()
}
struct PrimeParameters {
    min: u128,
    max: u128,
}
impl Absorbable for PrimeParameters {
    fn visit_chunks(&self, e: &mut dyn FnMut(&[u8])) {
        for (tag, payload) in [
            (
                b"prime-domain".as_slice(),
                b"f2z/spartan-u32-mul/runtime-prime/v2".to_vec(),
            ),
            (b"prime-min", self.min.to_le_bytes().to_vec()),
            (b"prime-max", self.max.to_le_bytes().to_vec()),
        ] {
            Message {
                tag,
                payload: &payload,
            }
            .visit_chunks(e);
        }
    }
    fn value(&self) -> Value {
        json!({"domain":"f2z/spartan-u32-mul/runtime-prime/v2","min_hex":number(self.min),"max_hex":number(self.max)})
    }
}
pub fn sample_prime(t: &mut Transcript, min: u128, max: u128) -> io::Result<PrimeContext> {
    if min < 3 || min > max || max >= 1 << 126 {
        return t.fail(io::Error::other("invalid prime interval"));
    }
    t.absorb(
        "projection_prime.parameters",
        json!({}),
        &PrimeParameters { min, max },
    )?;
    let q=t.squeeze("projection_prime",json!({}),|t| {
        let first=min|1;let last=(max-1)|1;
        if first>last {return Err(io::Error::other("no odd candidate"));}
        let count=(last-first)/2+1;
        let small=[3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251];
        for attempt in 1..=if count==1{1}else{64*(128-max.leading_zeros())} {
            let q=first+2*uniform_below(t,count)?;
            if small.iter().any(|&p|q!=p&&q%p==0)||!is_prime(Flavor::Any,&U128::from_u128(q)){continue;}
            let mut prime=true;
            if q>3 {let mr=MillerRabin::new(Odd::new(U128::from_u128(q)).unwrap());
                for _ in 0..72 {let base=2+uniform_below(t,q-3)?;if !mr.test(&U128::from_u128(base)).is_probably_prime(){prime=false;break;}}
            }
            if prime{return Ok((q,json!({"value_hex":number(q),"min_hex":format!("0x{min:x}"),"max_hex":format!("0x{max:x}"),"attempts":attempt})));}
        } Err(io::Error::other("prime sampling exhausted"))
    })?;
    t.absorb(
        "spartan.message",
        json!({}),
        &Message {
            tag: b"prime-q",
            payload: &q.to_le_bytes(),
        },
    )?;
    PrimeContext::new(q).map_err(io::Error::other)
}
pub struct Statement<'a> {
    pub q: u128,
    pub matrix: &'a [u8; 32],
    pub assignment: &'a [u8; 32],
}
impl Absorbable for Statement<'_> {
    fn visit_chunks(&self, e: &mut dyn FnMut(&[u8])) {
        for (tag, payload) in [
            (b"protocol".as_slice(), b"f2z/spartan/piop/v2".as_slice()),
            (b"field-modulus", &self.q.to_le_bytes()),
            (b"matrix-statement", self.matrix),
            (b"f2z/spartan/assignment-oracle/v1", self.assignment),
        ] {
            Message { tag, payload }.visit_chunks(e);
        }
    }
    fn value(&self) -> Value {
        json!({"protocol":"f2z/spartan/piop/v2","modulus_hex":number(self.q),"matrix_digest_hex":hex(self.matrix),"assignment_binding_hex":hex(self.assignment)})
    }
}
#[derive(Clone, Debug)]
pub struct Proof {
    pub outer: Vec<Vec<u128>>,
    pub terminal: [u128; 3],
    pub inner: Vec<Vec<u128>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub point: Vec<u128>,
    pub scale: u128,
    pub value: u128,
}
fn fold(f: &PrimeContext, v: &mut Vec<u128>, r: u128) {
    for i in 0..v.len() / 2 {
        v[i] = f.add(v[2 * i], f.mul(r, f.sub(v[2 * i + 1], v[2 * i])));
    }
    v.truncate(v.len() / 2);
}
fn polynomial(f: &PrimeContext, tables: &[Vec<u128>]) -> Vec<u128> {
    let mut coefficients = vec![0; tables.len() + 1];
    for i in 0..tables[0].len() / 2 {
        let mut p = vec![1];
        for v in tables {
            let a = v[2 * i];
            let b = f.sub(v[2 * i + 1], a);
            let mut next = vec![0; p.len() + 1];
            for j in 0..p.len() {
                next[j] = f.add(next[j], f.mul(p[j], a));
                next[j + 1] = f.add(next[j + 1], f.mul(p[j], b));
            }
            p = next;
        }
        for (c, p) in coefficients.iter_mut().zip(p) {
            *c = f.add(*c, p);
        }
    }
    coefficients
}
fn evaluate(f: &PrimeContext, p: &[u128], r: u128) -> u128 {
    p.iter().rev().fold(0, |a, &c| f.add(f.mul(a, r), c))
}
fn round(
    t: &mut Transcript,
    f: &PrimeContext,
    p: &[u128],
    index: usize,
    inner: bool,
) -> io::Result<u128> {
    let ctx = if inner && t.role() == "prover" {
        json!({"round":index,"stage":"spartan.inner"})
    } else {
        json!({"round":index})
    };
    t.absorb("sumcheck.round_polynomial",ctx.clone(),&Described{encoding:Fields{values:p,modulus:f.modulus()},display:&json!({"field":"prime","modulus_hex":number(f.modulus()),"basis":"monomial","coefficient_order":"constant_first","degree_bound":p.len()-1,"coefficients_hex":p.iter().copied().map(number).collect::<Vec<_>>()})})?;
    sample_field(t, "sumcheck.round_challenge", ctx, f.modulus())
}
fn start(t: &mut Transcript, f: &PrimeContext, binding: &[u8; 32]) -> io::Result<Vec<u128>> {
    t.absorb(
        "spartan.statement",
        json!({}),
        &Statement {
            q: f.modulus(),
            matrix: &matrix_digest(f.modulus()),
            assignment: binding,
        },
    )?;
    (0..15)
        .map(|coordinate| {
            sample_field(
                t,
                "spartan.zerocheck_equality_challenge",
                json!({"coordinate":coordinate}),
                f.modulus(),
            )
        })
        .collect()
}
fn terminal(t: &mut Transcript, f: &PrimeContext, v: &[u128; 3]) -> io::Result<u128> {
    t.absorb("spartan.outer_terminal_evaluations",json!({}),&Described{encoding:Fields{values:v,modulus:f.modulus()},display:&json!({"field":"prime","modulus_hex":number(f.modulus()),"az_hex":number(v[0]),"bz_hex":number(v[1]),"cz_hex":number(v[2])})})?;
    sample_field(
        t,
        "spartan.matrix_batching_challenge",
        json!({}),
        f.modulus(),
    )
}
fn matrix_weights(f: &PrimeContext, point: &[u128], rho: u128) -> Vec<u128> {
    let e = f.eq_table(point);
    let mut v = vec![0; 4 * ROWS];
    for (block, scale) in [(1, 1), (2, rho), (3, f.mul(rho, rho))] {
        for i in 0..ROWS {
            v[block * ROWS + i] = f.mul(scale, e[i]);
        }
    }
    v
}
pub fn prove(
    w: &Witness,
    f: &PrimeContext,
    binding: &[u8; 32],
    t: &mut Transcript,
) -> io::Result<(Proof, Claim)> {
    if f.modulus() <= u64::MAX as u128 {
        return Err(io::Error::other("U32 products require modulus above 2^64"));
    }
    let tau = start(t, f, binding)?;
    let mut tables = vec![f.eq_table(&tau), w.x.clone(), w.y.clone()];
    let mut c = w.product.clone();
    let mut outer = vec![];
    let mut outer_point = vec![];
    for i in 0..15 {
        let mut p = polynomial(f, &tables);
        let minus = polynomial(f, &[tables[0].clone(), c.clone()]);
        for (p, m) in p.iter_mut().zip(minus) {
            *p = f.sub(*p, m);
        }
        let r = round(t, f, &p, i, false)?;
        outer.push(p);
        outer_point.push(r);
        for v in &mut tables {
            fold(f, v, r);
        }
        fold(f, &mut c, r);
    }
    let values = [tables[1][0], tables[2][0], c[0]];
    let rho = terminal(t, f, &values)?;
    let mut tables = vec![matrix_weights(f, &outer_point, rho), w.assignment()];
    let mut inner = vec![];
    let mut point = vec![];
    for i in 0..17 {
        let p = polynomial(f, &tables);
        let r = round(t, f, &p, i, true)?;
        inner.push(p);
        point.push(r);
        for v in &mut tables {
            fold(f, v, r);
        }
    }
    Ok((
        Proof {
            outer,
            terminal: values,
            inner,
        },
        Claim {
            point,
            scale: tables[0][0],
            value: f.mul(tables[0][0], tables[1][0]),
        },
    ))
}
pub fn verify(
    proof: &Proof,
    f: &PrimeContext,
    binding: &[u8; 32],
    t: &mut Transcript,
) -> io::Result<Claim> {
    if f.modulus() <= u64::MAX as u128 {
        return Err(io::Error::other("U32 products require modulus above 2^64"));
    }
    if proof.outer.len() != 15 || proof.inner.len() != 17 {
        return Err(io::Error::other("Spartan proof dimensions"));
    }
    let tau = start(t, f, binding)?;
    let mut claim = 0;
    let mut outer_point = vec![];
    for (i, p) in proof.outer.iter().enumerate() {
        check_round(f, p, 4, claim)?;
        let r = round(t, f, p, i, false)?;
        claim = evaluate(f, p, r);
        outer_point.push(r);
    }
    for &v in &proof.terminal {
        f.canonical(v).map_err(io::Error::other)?;
    }
    let [a, b, c] = proof.terminal;
    if claim != f.mul(f.eq(&tau, &outer_point), f.sub(f.mul(a, b), c)) {
        return Err(io::Error::other("outer terminal claim"));
    }
    let rho = terminal(t, f, &proof.terminal)?;
    claim = f.add(a, f.add(f.mul(rho, b), f.mul(f.mul(rho, rho), c)));
    let mut point = vec![];
    for (i, p) in proof.inner.iter().enumerate() {
        check_round(f, p, 3, claim)?;
        let r = round(t, f, p, i, true)?;
        claim = evaluate(f, p, r);
        point.push(r);
    }
    let sel = f.eq_table(&point[15..]);
    let scale = f.mul(
        f.eq(&outer_point, &point[..15]),
        f.add(
            sel[1],
            f.add(f.mul(rho, sel[2]), f.mul(f.mul(rho, rho), sel[3])),
        ),
    );
    Ok(Claim {
        point,
        scale,
        value: claim,
    })
}
fn check_round(f: &PrimeContext, p: &[u128], len: usize, claim: u128) -> io::Result<()> {
    if p.len() != len
        || p.iter().any(|&x| x >= f.modulus())
        || f.add(p[0], p.iter().fold(0, |a, &x| f.add(a, x))) != claim
    {
        return Err(io::Error::other("invalid sumcheck round"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    struct Scripted {
        draws: VecDeque<u128>,
        chunks: Vec<Vec<u8>>,
        count: usize,
    }
    impl ByteTranscript for Scripted {
        fn absorb_bytes(&mut self, b: &[u8]) -> io::Result<()> {
            self.chunks.push(b.to_vec());
            Ok(())
        }
        fn challenge_bytes(&mut self, b: &mut [u8]) -> io::Result<()> {
            self.count += 1;
            b.copy_from_slice(
                &self
                    .draws
                    .pop_front()
                    .ok_or_else(|| io::Error::other("no scripted draw"))?
                    .to_le_bytes(),
            );
            Ok(())
        }
    }
    fn scripted(values: impl IntoIterator<Item = u128>) -> Scripted {
        Scripted {
            draws: values.into_iter().collect(),
            chunks: vec![],
            count: 0,
        }
    }
    #[test]
    fn rejection_edges_and_feedback() {
        let q = 65537;
        let limit = u128::MAX - ((u128::MAX % q + 1) % q);
        let mut t = scripted([limit + 1, limit]);
        let v = sample_field_value(&mut t, q).unwrap();
        assert_eq!(v, limit % q);
        assert_eq!(t.count, 2);
        // The accepted element has one length-delimited 16-byte value.
        let mut payload = 1u64.to_le_bytes().to_vec();
        payload.extend_from_slice(&16u64.to_le_bytes());
        payload.extend_from_slice(&v.to_le_bytes());
        let mut frame = b"f2z/spartan/transcript-frame/v1".to_vec();
        frame.extend_from_slice(&14u64.to_le_bytes());
        frame.extend_from_slice(b"field-elements");
        frame.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        frame.extend_from_slice(&payload);
        assert_eq!(t.chunks, vec![vec![6], frame, vec![7]]);
        let mut t = scripted([0, 1]);
        assert_eq!(uniform_below(&mut t, 3).unwrap(), 1);
        assert_eq!(t.count, 2);
        let mut t = scripted(std::iter::repeat_n(u128::MAX, 256));
        assert!(sample_field_value(&mut t, q).is_err());
        assert_eq!(t.count, 256);
        assert!(t.chunks.is_empty());
        let mut t = scripted(std::iter::repeat_n(0, 256));
        assert!(uniform_below(&mut t, 3).is_err());
        assert_eq!(t.count, 256);
    }
    #[test]
    fn prime_errors_poison_both_recording_modes() {
        for recording in [false, true] {
            let sink = recording.then(|| Box::new(Vec::<u8>::new()) as Box<dyn std::io::Write>);
            let mut t = Transcript::with_writer("prover", sink).unwrap();
            assert!(sample_prime(&mut t, 9, 9).is_err());
            assert!(sample_prime(&mut t, 3, 3).is_err());
            assert!(t.finish().is_err());
        }
    }
    #[test]
    fn witness_packing_comes_from_operands() {
        assert!(Witness::from_operands([(1, 2)]).is_err());
        assert!(Witness::from_operands(std::iter::repeat_n((0, 0), ROWS + 1)).is_err());
        let w = Witness::from_operands((0..ROWS).map(|i| (i as u32, u32::MAX))).unwrap();
        for i in [0, 1, 127, 128, ROWS - 1] {
            let mut bits = 0u128;
            for slot in 0..128 {
                let bit = ((i & 127) << 15) | (slot << 8) | (i >> 7);
                let word = &w.packed[bit >> 7];
                let limb = if bit & 127 < 64 { word.lo } else { word.hi };
                bits |= u128::from((limb >> (bit & 63)) & 1) << slot;
            }
            assert_eq!(
                bits,
                (i as u128)
                    | ((u32::MAX as u128) << 32)
                    | (((i as u64 * u32::MAX as u64) as u128) << 64)
            );
        }
    }
    #[test]
    fn spartan_discharge_and_malformed_proofs() {
        let w = Witness::deterministic();
        let f = PrimeContext::new(0x7e491c075b45202661dd17daefef).unwrap();
        let binding = [17; 32];
        let mut p = Transcript::new("prover", None).unwrap();
        let (proof, claim) = prove(&w, &f, &binding, &mut p).unwrap();
        w.check_claim(&f, &claim).unwrap();
        let verify_fresh = |proof: &Proof, binding: &[u8; 32]| {
            let mut t = Transcript::new("verifier", None).unwrap();
            verify(proof, &f, binding, &mut t)
        };
        assert_eq!(verify_fresh(&proof, &binding).unwrap(), claim);
        let mut bad = claim.clone();
        bad.value = f.add(bad.value, 1);
        assert!(w.check_claim(&f, &bad).is_err());
        for outer in [false, true] {
            for round in 0..if outer { 15 } else { 17 } {
                let mut bad = proof.clone();
                let p = if outer {
                    &mut bad.outer[round]
                } else {
                    &mut bad.inner[round]
                };
                p[0] = f.add(p[0], 1);
                assert!(verify_fresh(&bad, &binding).is_err());
            }
        }
        for value in 0..3 {
            let mut bad = proof.clone();
            bad.terminal[value] = f.add(bad.terminal[value], 1);
            assert!(verify_fresh(&bad, &binding).is_err());
        }
        let mut bad = proof.clone();
        bad.inner[16].push(0);
        assert!(verify_fresh(&bad, &binding).is_err());
        let mut bad = proof.clone();
        bad.outer[0][0] = f.modulus();
        assert!(verify_fresh(&bad, &binding).is_err());
        let mut bad = proof.clone();
        bad.inner.pop();
        assert!(verify_fresh(&bad, &binding).is_err());
        assert!(verify_fresh(&proof, &[18; 32]).is_err());
    }
}
