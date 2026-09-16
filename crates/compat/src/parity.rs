//! Strict role-local comparison plus independent replay of every squeeze.
use serde_json::{Value, json};
use std::{fs::File, io, path::Path};
use transcript::reference::{hex, read_events, unhex};

fn error(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}
fn load(path: &Path) -> io::Result<Value> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}
fn expected(index: usize) -> (&'static str, &'static str, Option<usize>) {
    match index {
        0 => ("absorb", "statement.assignment_binding_digest", None),
        1 | 2 => ("absorb", "statement.ligerito_policy_digest", None),
        3 => ("absorb", "projection_prime.parameters", None),
        4 => ("squeeze", "projection_prime", None),
        5 => ("absorb", "spartan.message", None),
        6 => ("absorb", "spartan.statement", None),
        7..=21 => (
            "squeeze",
            "spartan.zerocheck_equality_challenge",
            Some(index - 7),
        ),
        22..=51 => (
            if index % 2 == 0 { "absorb" } else { "squeeze" },
            if index % 2 == 0 {
                "sumcheck.round_polynomial"
            } else {
                "sumcheck.round_challenge"
            },
            Some((index - 22) / 2),
        ),
        52 => ("absorb", "spartan.outer_terminal_evaluations", None),
        53 => ("squeeze", "spartan.matrix_batching_challenge", None),
        54..=87 => (
            if index % 2 == 0 { "absorb" } else { "squeeze" },
            if index % 2 == 0 {
                "sumcheck.round_polynomial"
            } else {
                "sumcheck.round_challenge"
            },
            Some((index - 54) / 2),
        ),
        _ => unreachable!(),
    }
}
fn validate_role(events: &[Value], role: &str) -> io::Result<String> {
    if events.len() != 88 {
        return Err(error(format!(
            "{role}: expected exactly 88 events, got {}",
            events.len()
        )));
    }
    let base = events[0]["sequence"]
        .as_u64()
        .ok_or_else(|| error("sequence"))?;
    let mut h = blake3::Hasher::new();
    for (i, e) in events.iter().enumerate() {
        transcript::reference::validate_event(e)?;
        let (op, purpose, coordinate) = expected(i);
        if e["role"] != role
            || e["sequence"].as_u64() != base.checked_add(i as u64)
            || e["operation"] != op
            || e["purpose"] != purpose
        {
            return Err(error(format!(
                "{role} event {i}: invalid schedule or sequence"
            )));
        }
        let mut ctx = json!({});
        if let Some(n) = coordinate {
            ctx[if i < 22 { "coordinate" } else { "round" }] = json!(n);
        }
        if i >= 54 && role == "prover" {
            ctx["stage"] = json!("spartan.inner");
        }
        if e["context"] != ctx {
            return Err(error(format!("{role} event {i}: invalid semantic context")));
        }
        let wire = e["wire"].as_array().ok_or_else(|| error("wire"))?;
        if wire.is_empty() {
            return Err(error("empty wire event"));
        }
        let mut j = 0;
        while j < wire.len() {
            let f = &wire[j]["fields"];
            let bytes = unhex(f["bytes_hex"].as_str().ok_or_else(|| error("bytes"))?)?;
            if f["operation"] == "squeeze" {
                if bytes.len() != 16 || f["part"] != "random_bytes" {
                    return Err(error("unsupported raw draw"));
                }
                let mut expected_bytes = vec![0; bytes.len()];
                h.finalize_xof().fill(&mut expected_bytes);
                if bytes != expected_bytes {
                    return Err(error(format!(
                        "{role} event {i} wire {j}: squeeze replay mismatch"
                    )));
                }
                for (offset, part, data) in [
                    (1, "challenge_frame_start", vec![0x12]),
                    (2, "challenge_feedback", bytes),
                    (3, "challenge_frame_end", vec![0x34]),
                ] {
                    let next = wire
                        .get(j + offset)
                        .ok_or_else(|| error("missing feedback"))?;
                    if next["fields"]
                        != json!({"operation":"absorb","part":part,"byte_len":data.len(),"bytes_hex":hex(&data)})
                    {
                        return Err(error(format!(
                            "{role} event {i}: invalid challenge feedback"
                        )));
                    }
                    h.update(&data);
                }
                j += 4;
            } else {
                if f["part"] != "input" {
                    return Err(error("unexpected standalone feedback"));
                }
                h.update(&bytes);
                j += 1;
            }
        }
    }
    Ok(h.finalize().to_hex().to_string())
}
fn capture(path: &Path) -> io::Result<[Vec<Value>; 2]> {
    let mut roles = [vec![], vec![]];
    for e in read_events(path)? {
        let e = e?;
        let role = match e["role"].as_str() {
            Some("prover") => 0,
            Some("verifier") => 1,
            _ => return Err(error("unknown role")),
        };
        roles[role].push(e);
    }
    Ok(roles)
}
fn manifest(m: &Value) -> io::Result<()> {
    if m["schema"] != "f2z.spartan-prefix-run/v1"
        || m["complete"] != true
        || m["boundary"] != "end-spartan"
        || m["events_per_role"] != 88
        || m["verification"] != json!({"prefix":true,"opening":false})
        || m["fiat_shamir_transcript_logs"] != true
    {
        return Err(error("capture lacks a complete Spartan-only manifest"));
    }
    Ok(())
}

/// Reading reference captures happens only after the independent run completes.
pub fn compare(reference: &Path, benchmark: &Path) -> io::Result<Value> {
    let a = load(&reference.join("run.json"))?;
    let b = load(&benchmark.join("run.json"))?;
    manifest(&a)?;
    manifest(&b)?;
    for key in [
        "fixture",
        "root_hex",
        "assignment_binding_hex",
        "policy_digest_hex",
        "matrix_digest_hex",
        "prime_hex",
        "terminal_claims",
        "final_transcript_digests",
    ] {
        if a.get(key).is_none() || a[key] != b[key] {
            return Err(error(format!("checkpoint mismatch: {key}")));
        }
    }
    if b["direct_assignment_check"] != true {
        return Err(error(
            "benchmark did not discharge the terminal assignment claim",
        ));
    }
    let statement = load(&reference.join("statement.json"))?;
    let local_statement = load(&benchmark.join("statement.json"))?;
    let cfg = crate::statement::Configuration::from_public_statement(&statement)?;
    if statement != local_statement {
        return Err(error("public statements differ"));
    }
    let root: [u8; 32] = unhex(a["root_hex"].as_str().ok_or_else(|| error("root"))?)?
        .try_into()
        .map_err(|_| error("root length"))?;
    if statement != cfg.public_statement(&root)?
        || a["assignment_binding_hex"] != hex(&cfg.assignment_binding(&root))
        || a["policy_digest_hex"] != hex(&cfg.policy_digest())
    {
        return Err(error("public statement checkpoint mismatch"));
    }
    let x = capture(&reference.join("transcript.jsonl"))?;
    let y = capture(&benchmark.join("transcript.jsonl"))?;
    let mut roles = json!({});
    for (k, role) in ["prover", "verifier"].into_iter().enumerate() {
        validate_checkpoint(&x[k], &a, role)?;
        validate_checkpoint(&y[k], &b, role)?;
        roles[role] = compare_role(
            &x[k],
            &y[k],
            role,
            &a["final_transcript_digests"][role],
            &b["final_transcript_digests"][role],
        )?;
    }
    Ok(
        json!({"schema":"f2z.spartan-parity/v1","equivalent":true,"boundary":"end-spartan","roles":roles,"terminal_assignment_checked":true,"opening_verified":false}),
    )
}

fn parse_number(v: &Value) -> io::Result<u128> {
    let s = v
        .as_str()
        .ok_or_else(|| error("field value must be hexadecimal"))?;
    if s.len() != 34 || !s.starts_with("0x") {
        return Err(error("field value must use canonical 16-byte encoding"));
    }
    u128::from_str_radix(&s[2..], 16).map_err(io::Error::other)
}
fn validate_checkpoint(events: &[Value], m: &Value, role: &str) -> io::Result<()> {
    if events.len() != 88 {
        return Err(error("incomplete Spartan checkpoint"));
    }
    let q = parse_number(&m["prime_hex"])?;
    if !(1u128 << 110..1u128 << 111).contains(&q) {
        return Err(error("prime outside supported interval"));
    }
    let f = field::runtime::PrimeContext::new(q).map_err(io::Error::other)?;
    let matrix = hex(&spartan::reference::matrix_digest(q));
    if events[0]["value"]["bytes_hex"] != m["assignment_binding_hex"]
        || events[2]["value"]["bytes_hex"] != m["policy_digest_hex"]
        || events[4]["value"]["value_hex"] != m["prime_hex"]
        || events[6]["value"]["matrix_digest_hex"] != matrix
        || m["matrix_digest_hex"] != matrix
    {
        return Err(error(format!("{role}: transcript checkpoint mismatch")));
    }
    let canonical = |v: &Value| {
        let n = parse_number(v)?;
        f.canonical(n).map_err(io::Error::other)
    };
    let outer = (0..15)
        .map(|i| canonical(&events[23 + 2 * i]["value"]["value_hex"]))
        .collect::<io::Result<Vec<_>>>()?;
    let point = (0..17)
        .map(|i| canonical(&events[55 + 2 * i]["value"]["value_hex"]))
        .collect::<io::Result<Vec<_>>>()?;
    let rho = canonical(&events[53]["value"]["value_hex"])?;
    let selectors = f.eq_table(&point[15..]);
    let scale = f.mul(
        f.eq(&outer, &point[..15]),
        f.add(
            selectors[1],
            f.add(
                f.mul(rho, selectors[2]),
                f.mul(f.mul(rho, rho), selectors[3]),
            ),
        ),
    );
    let polynomial = events[86]["value"]["coefficients_hex"]
        .as_array()
        .ok_or_else(|| error("terminal polynomial"))?;
    if polynomial.len() != 3 {
        return Err(error("terminal polynomial degree"));
    }
    let mut value = 0;
    for coefficient in polynomial.iter().rev() {
        value = f.add(f.mul(value, point[16]), canonical(coefficient)?);
    }
    let expected = json!({"point":point.into_iter().map(transcript::reference::number).collect::<Vec<_>>(),"scale":transcript::reference::number(scale),"value":transcript::reference::number(value)});
    if m["terminal_claims"][role] != expected {
        return Err(error(format!(
            "{role}: terminal claim differs from transcript"
        )));
    }
    Ok(())
}

fn compare_role(
    x: &[Value],
    y: &[Value],
    role: &str,
    expected_left: &Value,
    expected_right: &Value,
) -> io::Result<Value> {
    let dx = validate_role(x, role)?;
    let dy = validate_role(y, role)?;
    if expected_left.as_str() != Some(dx.as_str()) || expected_right.as_str() != Some(dy.as_str()) {
        return Err(error(format!(
            "{role}: replay digest differs from manifest"
        )));
    }
    for (i, (left, right)) in x.iter().zip(y).enumerate() {
        for field in ["operation", "purpose", "context", "value", "draw_count"] {
            if left[field] != right[field] {
                return Err(error(format!(
                    "{role} event {i} {}: {field} mismatch",
                    left["purpose"]
                )));
            }
        }
        let lw = left["wire"].as_array().unwrap();
        let rw = right["wire"].as_array().unwrap();
        if lw.len() != rw.len() {
            return Err(error(format!(
                "{role} event {i}: wire boundary count mismatch"
            )));
        }
        for (j, (l, r)) in lw.iter().zip(rw).enumerate() {
            if l["fields"] != r["fields"] {
                return Err(error(format!(
                    "{role} event {i} wire {j}: bytes or framing mismatch"
                )));
            }
        }
    }
    Ok(
        json!({"events":88,"raw_draws":x.iter().map(|e|e["draw_count"].as_u64().unwrap()).sum::<u64>(),"replayed_digest":dx}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_records_are_rejected() {
        for e in [
            json!({"schema_version":999,"complete":false}),
            json!({"schema_version":1,"complete":true}),
        ] {
            assert!(transcript::reference::validate_event(&e).is_err());
        }
        assert!(validate_role(&[], "prover").is_err());
        assert!(manifest(&json!({})).is_err());
    }
    fn golden() -> ([Vec<Value>; 2], Value) {
        let mut roles = [vec![], vec![]];
        for line in include_str!("../tests/fixtures/spartan-canonical/transcript.jsonl").lines() {
            let e: Value = serde_json::from_str(line).unwrap();
            let k = usize::from(e["role"] == "verifier");
            roles[k].push(e);
        }
        (
            roles,
            serde_json::from_str(include_str!("../tests/fixtures/spartan-canonical/run.json"))
                .unwrap(),
        )
    }
    #[test]
    fn reference_capture_replays_and_rejects_mutations() {
        let (roles, run) = golden();
        manifest(&run).unwrap();
        for (k, role) in ["prover", "verifier"].into_iter().enumerate() {
            let good = &roles[k];
            let digest = &run["final_transcript_digests"][role];
            compare_role(good, good, role, digest, digest).unwrap();
            validate_checkpoint(good, &run, role).unwrap();
            let mut wrong = run.clone();
            wrong["terminal_claims"][role]["value"] = json!("0x00000000000000000000000000000000");
            assert!(validate_checkpoint(good, &wrong, role).is_err());
            let mut harmless = good.clone();
            for e in &mut harmless {
                e["sequence"] = json!(e["sequence"].as_u64().unwrap() + 1000);
                e["spans"] = json!([{"path":"elsewhere"}]);
            }
            compare_role(good, &harmless, role, digest, digest).unwrap();
            for mutation in 0..13 {
                let mut bad = good.clone();
                match mutation {
                    0 => {
                        bad.pop();
                    }
                    1 => bad.push(good[87].clone()),
                    2 => bad[5]["value"]["bytes_hex"] = json!("00"),
                    3 => bad[54]["context"] = json!({"round":9}),
                    4 => bad[4]["draw_count"] = json!(0),
                    5 => bad[7]["complete"] = json!(false),
                    6 => bad[8]["schema_version"] = json!(2),
                    7 => bad[3]["purpose"] = json!("other"),
                    8 => bad[10]["sequence"] = json!(999),
                    9 => bad[4]["wire"][0]["fields"]["bytes_hex"] = json!("00".repeat(16)),
                    10 => bad[0]["wire"][0]["fields"]["byte_len"] = json!(5),
                    11 => {
                        bad[4]["wire"].as_array_mut().unwrap().remove(2);
                    }
                    12 => bad[0]["wire"][0]["fields"]["bytes_hex"] = json!("zz"),
                    _ => unreachable!(),
                }
                assert!(
                    compare_role(good, &bad, role, digest, digest).is_err(),
                    "{role}, mutation {mutation}"
                );
            }
        }
    }
}
