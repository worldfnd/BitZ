//! Independent execution through the last inner sumcheck challenge.
use crate::{ROWS, Witness, statement::Configuration};
use serde_json::{Value, json};
use spartan::reference::{self, Claim};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
};
use transcript::reference::{Transcript, hex, number};

pub fn fixture(name: &str) -> io::Result<Witness> {
    let mut seed = if matches!(name, "canonical" | "edges") {
        0
    } else {
        name.parse::<u64>().map_err(io::Error::other)?
    };
    Witness::from_operands((0..ROWS).map(|i| match name {
        "canonical" => (
            (i as u32).wrapping_mul(0x9e3779b9) | 1,
            (i as u32).wrapping_mul(0x85ebca6b) | 1,
        ),
        "edges" => [
            (0, 0),
            (0, u32::MAX),
            (1, u32::MAX),
            (u32::MAX, u32::MAX),
            (1 << 31, 1 << 31),
        ][i % 5],
        _ => {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (seed >> 32) as u32;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x, (seed >> 32) as u32)
        }
    }))
}
fn claim_json(c: &Claim) -> Value {
    json!({"point":c.point.iter().copied().map(number).collect::<Vec<_>>(),"scale":number(c.scale),"value":number(c.value)})
}
pub fn save(path: &Path, value: &Value) -> io::Result<()> {
    let mut w = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer_pretty(&mut w, value)?;
    w.write_all(b"\n")?;
    w.flush()
}

/// Only public configuration and locally constructed operands enter execution.
/// No expected challenge, message, or claim is an input to this function.
pub fn run(
    configuration: &Configuration,
    witness: &Witness,
    out: &Path,
    record: bool,
    fixture_name: &str,
) -> io::Result<Value> {
    fs::create_dir(out)?;
    let pcs = pcs::Pcs::from_security_config(
        &common::Shape::new(15, 7).map_err(|e| io::Error::other(format!("{e:?}")))?,
        pcs::LigeritoProfile::Fast,
        configuration.policy(),
    )
    .map_err(|e| io::Error::other(format!("{e:?}")))?;
    let (root, _) = pcs
        .commit(witness.packed())
        .map_err(|e| io::Error::other(format!("{e:?}")))?;
    let file = record
        .then(|| File::create_new(out.join("transcript.jsonl")))
        .transpose()?;
    let sink = |f: Option<File>| f.map(|f| Box::new(BufWriter::new(f)) as Box<dyn Write>);
    let mut p = Transcript::with_writer(
        "prover",
        sink(file.as_ref().map(File::try_clone).transpose()?),
    )?;
    let binding = configuration.bind(&root.0, &mut p)?;
    let field = reference::sample_prime(
        &mut p,
        configuration.prime_bounds().0,
        configuration.prime_bounds().1,
    )?;
    let (proof, claim) = reference::prove(witness, &field, &binding, &mut p)?;
    witness.check_claim(&field, &claim)?;
    let prover_digest = p.digest();
    let prover_count = p.event_count();
    p.finish()?;
    let mut v = Transcript::with_writer("verifier", sink(file))?;
    let vb = configuration.bind(&root.0, &mut v)?;
    let vf = reference::sample_prime(
        &mut v,
        configuration.prime_bounds().0,
        configuration.prime_bounds().1,
    )?;
    let verified = reference::verify(&proof, &vf, &vb, &mut v)?;
    witness.check_claim(&vf, &verified)?;
    if field.modulus() != vf.modulus() || binding != vb || claim != verified {
        return Err(io::Error::other("local prover/verifier mismatch"));
    }
    let verifier_digest = v.digest();
    let verifier_count = v.event_count();
    v.finish()?;
    if prover_count != 88 || verifier_count != 88 {
        return Err(io::Error::other("unexpected Spartan boundary"));
    }
    let report = json!({"schema":"f2z.spartan-prefix-run/v1","boundary":"end-spartan","complete":true,
        "fixture":fixture_name,"events_per_role":88,"root_hex":hex(&root.0),"assignment_binding_hex":hex(&binding),
        "policy_digest_hex":hex(&configuration.policy_digest()),"matrix_digest_hex":hex(&reference::matrix_digest(field.modulus())),"prime_hex":number(field.modulus()),
        "terminal_claims":{"prover":claim_json(&claim),"verifier":claim_json(&verified)},
        "final_transcript_digests":{"prover":prover_digest,"verifier":verifier_digest},
        "verification":{"prefix":true,"opening":false},"direct_assignment_check":true,"fiat_shamir_transcript_logs":record});
    save(
        &out.join("statement.json"),
        &configuration.public_statement(&root.0)?,
    )?;
    save(&out.join("run.json"), &report)?;
    Ok(report)
}
