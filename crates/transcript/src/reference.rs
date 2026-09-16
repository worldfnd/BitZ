//! Opt-in F2Z byte profile. Proof transport is deliberately outside this API.
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn number(v: u128) -> String {
    format!("0x{v:032x}")
}
pub fn unhex(s: &str) -> io::Result<Vec<u8>> {
    if !s.is_ascii() || s.len() % 2 != 0 {
        return Err(io::Error::other("invalid hexadecimal bytes"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(io::Error::other))
        .collect()
}
pub trait Absorbable {
    fn visit_chunks(&self, emit: &mut dyn FnMut(&[u8]));
    fn value(&self) -> Value;
}
pub struct Bytes<'a>(pub &'a [u8]);
impl Absorbable for Bytes<'_> {
    fn visit_chunks(&self, emit: &mut dyn FnMut(&[u8])) {
        emit(&[6]);
        emit(self.0);
        emit(&[7]);
    }
    fn value(&self) -> Value {
        json!({"bytes_hex":hex(self.0)})
    }
}
pub struct Described<'a, T: Absorbable> {
    pub encoding: T,
    pub display: &'a Value,
}
impl<T: Absorbable> Absorbable for Described<'_, T> {
    fn visit_chunks(&self, e: &mut dyn FnMut(&[u8])) {
        self.encoding.visit_chunks(e)
    }
    fn value(&self) -> Value {
        self.display.clone()
    }
}

/// Field-independent byte interface. Protocol samplers own interpretation and rejection.
pub trait ByteTranscript {
    fn absorb_bytes(&mut self, bytes: &[u8]) -> io::Result<()>;
    fn challenge_bytes(&mut self, bytes: &mut [u8]) -> io::Result<()>;
    fn absorb_object(&mut self, object: &impl Absorbable) -> io::Result<()> {
        let mut result = Ok(());
        object.visit_chunks(&mut |bytes| {
            if result.is_ok() {
                result = self.absorb_bytes(bytes);
            }
        });
        result
    }
}

/// Optional recording observes the same byte engine used with logging disabled.
/// Any operation or sink error permanently invalidates this run. A caller must
/// publish its completion manifest only after `finish` succeeds.
pub struct Transcript {
    hasher: blake3::Hasher,
    role: String,
    sequence: u64,
    writer: Option<Box<dyn Write>>,
    wire: Vec<Value>,
    draws: usize,
    active: bool,
    failed: bool,
    finished: bool,
}
impl Transcript {
    pub fn new(role: &str, path: Option<&Path>) -> io::Result<Self> {
        let writer = path
            .map(File::create_new)
            .transpose()?
            .map(|f| Box::new(BufWriter::new(f)) as Box<dyn Write>);
        Self::with_writer(role, writer)
    }
    pub fn with_writer(role: &str, writer: Option<Box<dyn Write>>) -> io::Result<Self> {
        if !matches!(role, "prover" | "verifier") {
            return Err(io::Error::other("invalid transcript role"));
        }
        Ok(Self {
            hasher: blake3::Hasher::new(),
            role: role.into(),
            sequence: 0,
            writer,
            wire: vec![],
            draws: 0,
            active: false,
            failed: false,
            finished: false,
        })
    }
    pub fn role(&self) -> &str {
        &self.role
    }
    pub fn digest(&self) -> String {
        self.hasher.finalize().to_hex().to_string()
    }
    pub fn event_count(&self) -> u64 {
        self.sequence
    }
    fn ready(&self) -> io::Result<()> {
        if self.failed || self.finished {
            Err(io::Error::other("transcript run is closed or failed"))
        } else {
            Ok(())
        }
    }
    pub fn finish(&mut self) -> io::Result<()> {
        self.ready()?;
        if self.active {
            self.failed = true;
            return Err(io::Error::other("unfinished operation"));
        }
        if let Some(w) = &mut self.writer {
            if let Err(e) = w.flush() {
                self.failed = true;
                return Err(e);
            }
        }
        self.finished = true;
        Ok(())
    }
    fn update(&mut self, part: &str, bytes: &[u8]) {
        self.hasher.update(bytes);
        self.record("absorb", part, bytes);
    }
    fn record(&mut self, operation: &str, part: &str, bytes: &[u8]) {
        if self.writer.is_some() {
            self.wire.push(json!({"target":"f2z::transcript","fields":{"operation":operation,"part":part,"byte_len":bytes.len(),"bytes_hex":hex(bytes)},"spans":[]}));
        }
    }
    pub fn fail<T>(&mut self, error: io::Error) -> io::Result<T> {
        self.failed = true;
        Err(error)
    }
    fn start(&mut self) -> io::Result<()> {
        self.ready()?;
        if self.active {
            self.failed = true;
            return Err(io::Error::other("nested logical operation"));
        }
        self.active = true;
        self.draws = 0;
        self.wire.clear();
        Ok(())
    }
    fn end(
        &mut self,
        operation: &str,
        purpose: &str,
        context: Value,
        value: Value,
    ) -> io::Result<()> {
        self.ready()?;
        let result = (|| {
            if let Some(w) = &mut self.writer {
                serde_json::to_writer(
                    &mut *w,
                    &json!({"schema_version":1,"sequence":self.sequence,"role":self.role,"operation":operation,"purpose":purpose,"context":context,"complete":true,"value":value,"draw_count":self.draws,"spans":[],"wire":self.wire}),
                )?;
                w.write_all(b"\n")?;
            }
            Ok(())
        })();
        self.active = false;
        if result.is_err() {
            self.failed = true;
        } else {
            self.sequence += 1;
            self.wire.clear();
        }
        result
    }
    pub fn absorb(
        &mut self,
        purpose: &str,
        ctx: Value,
        object: &impl Absorbable,
    ) -> io::Result<()> {
        self.start()?;
        self.absorb_object(object)?;
        let value = if self.writer.is_some() {
            object.value()
        } else {
            Value::Null
        };
        self.end("absorb", purpose, ctx, value)
    }
    pub fn squeeze<T>(
        &mut self,
        purpose: &str,
        ctx: Value,
        f: impl FnOnce(&mut Self) -> io::Result<(T, Value)>,
    ) -> io::Result<T> {
        self.start()?;
        let (result, value) = match f(self) {
            Ok(v) => v,
            Err(e) => {
                self.failed = true;
                return Err(e);
            }
        };
        self.end("squeeze", purpose, ctx, value)?;
        Ok(result)
    }
}
impl ByteTranscript for Transcript {
    fn absorb_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.ready()?;
        if !self.active {
            self.failed = true;
            return Err(io::Error::other("bytes outside logical operation"));
        }
        self.update("input", bytes);
        Ok(())
    }
    fn challenge_bytes(&mut self, bytes: &mut [u8]) -> io::Result<()> {
        self.ready()?;
        if !self.active {
            self.failed = true;
            return Err(io::Error::other("draw outside logical operation"));
        }
        self.hasher.finalize_xof().fill(bytes);
        self.record("squeeze", "random_bytes", bytes);
        self.draws += 1;
        self.update("challenge_frame_start", &[0x12]);
        self.update("challenge_feedback", bytes);
        self.update("challenge_frame_end", &[0x34]);
        Ok(())
    }
}

pub fn validate_event(e: &Value) -> io::Result<()> {
    let bad = || io::Error::other("invalid or incomplete logical transcript event");
    if e["schema_version"] != 1
        || e["complete"] != true
        || !matches!(e["role"].as_str(), Some("prover" | "verifier"))
        || !matches!(e["operation"].as_str(), Some("absorb" | "squeeze"))
        || e["sequence"].as_u64().is_none()
        || e["purpose"].as_str().is_none()
        || !e["context"].is_object()
        || e.get("value").is_none()
    {
        return Err(bad());
    }
    let wire = e["wire"].as_array().ok_or_else(bad)?;
    let mut draws = 0;
    for step in wire {
        if step["target"] != "f2z::transcript" {
            return Err(bad());
        }
        let f = &step["fields"];
        let bytes = unhex(f["bytes_hex"].as_str().ok_or_else(bad)?)?;
        if f["byte_len"].as_u64() != Some(bytes.len() as u64) || f["part"].as_str().is_none() {
            return Err(bad());
        }
        match f["operation"].as_str() {
            Some("absorb") => (),
            Some("squeeze") => draws += 1,
            _ => return Err(bad()),
        }
    }
    if e["draw_count"].as_u64() != Some(draws) || (e["operation"] == "absorb" && draws != 0) {
        return Err(bad());
    }
    Ok(())
}
pub fn read_events(path: &Path) -> io::Result<impl Iterator<Item = io::Result<Value>>> {
    Ok(BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let parse = || {
                let e: Value = serde_json::from_str(&line?)?;
                validate_event(&e)?;
                Ok(e)
            };
            parse()
                .map_err(|e: io::Error| io::Error::other(format!("transcript line {}: {e}", i + 1)))
        }))
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fail(bool);
    impl Write for Fail {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if self.0 {
                Ok(b.len())
            } else {
                Err(io::Error::other("write failed"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }
    #[test]
    fn errors_are_terminal_with_or_without_recording() {
        for recording in [false, true] {
            let sink = recording.then(|| Box::new(Vec::<u8>::new()) as Box<dyn Write>);
            let mut t = Transcript::with_writer("prover", sink).unwrap();
            assert!(
                t.squeeze::<()>("failed", json!({}), |t| {
                    t.challenge_bytes(&mut [0; 16])?;
                    Err(io::Error::other("sampling exhausted"))
                })
                .is_err()
            );
            assert!(t.absorb("next", json!({}), &Bytes(b"x")).is_err());
            assert!(t.finish().is_err());
        }
    }
    #[test]
    fn writer_errors_are_terminal() {
        for flush in [false, true] {
            let mut t = Transcript::with_writer("prover", Some(Box::new(Fail(flush)))).unwrap();
            let r = t.absorb("data", json!({}), &Bytes(b"x"));
            assert_eq!(r.is_ok(), flush);
            assert!(t.finish().is_err());
            assert!(t.absorb("next", json!({}), &Bytes(b"x")).is_err());
        }
    }
    #[test]
    fn recording_does_not_change_bytes() {
        let mut a = Transcript::new("prover", None).unwrap();
        let mut b = Transcript::with_writer("prover", Some(Box::new(Vec::<u8>::new()))).unwrap();
        for t in [&mut a, &mut b] {
            t.absorb("bytes", json!({}), &Bytes(b"fixture")).unwrap();
            t.squeeze("raw", json!({}), |t| {
                let mut bytes = [0; 16];
                t.challenge_bytes(&mut bytes)?;
                Ok(((), json!(hex(&bytes))))
            })
            .unwrap();
            t.finish().unwrap();
        }
        assert_eq!(a.digest(), b.digest());
    }
}
