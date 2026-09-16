//! Capture locally, then compare with a completed f2z-pcs Spartan-only export.
use f2z_compat::{capture, parity, statement::Configuration};
use std::{fs::File, io, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut reference = None;
    let mut out = None;
    let mut fixture = "canonical".to_owned();
    let mut record = true;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--reference" => reference = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--fixture" => fixture = args.next().ok_or("missing fixture")?,
            "--no-logs" => record = false,
            _ => return Err(format!(
                "unknown argument {arg}; usage: --reference DIR --out NEW_DIR [--fixture canonical|edges|SEED] [--no-logs]"
            ).into()),
        }
    }
    let reference = reference.ok_or("missing --reference")?;
    let out = out.ok_or("missing --out")?;
    let statement: serde_json::Value =
        serde_json::from_reader(File::open(reference.join("statement.json"))?)?;
    let cfg = Configuration::from_public_statement(&statement)?;
    let witness = capture::fixture(&fixture)?;
    let run = capture::run(&cfg, &witness, &out, record, &fixture)?;
    if run["root_hex"] != statement["public_instance"]["root_hex"]
        || run["assignment_binding_hex"] != statement["assignment_binding_hex"]
    {
        return Err(io::Error::other(
            "independently derived commitment/binding does not match public statement",
        )
        .into());
    }
    if record {
        let report = parity::compare(&reference, &out)?;
        capture::save(&out.join("parity.json"), &report)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Spartan verified with recording disabled: {}",
            out.display()
        );
    }
    Ok(())
}
