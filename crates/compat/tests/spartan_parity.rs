use f2z_compat::{capture, parity, statement::Configuration};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "spartan-parity-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn independent_execution_matches_reference_with_and_without_logging() {
    let reference = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/spartan-canonical");
    let expected: Value =
        serde_json::from_str(include_str!("fixtures/spartan-canonical/run.json")).unwrap();
    let mut statement: Value =
        serde_json::from_str(include_str!("../src/supported_statement.json")).unwrap();
    statement["public_instance"]["root_hex"] = expected["root_hex"].clone();
    statement["assignment_binding_hex"] = expected["assignment_binding_hex"].clone();
    let cfg = Configuration::from_public_statement(&statement).unwrap();
    let w = capture::fixture("canonical").unwrap();
    let scratch = Scratch::new();
    let logged = scratch.0.join("logged");
    let silent = scratch.0.join("silent");
    let a = capture::run(&cfg, &w, &logged, true, "canonical").unwrap();
    assert_eq!(
        parity::compare(&reference, &logged).unwrap()["equivalent"],
        true
    );
    let b = capture::run(&cfg, &w, &silent, false, "canonical").unwrap();
    for field in [
        "root_hex",
        "assignment_binding_hex",
        "prime_hex",
        "matrix_digest_hex",
        "terminal_claims",
        "final_transcript_digests",
    ] {
        assert_eq!(a[field], b[field], "{field}");
    }
    assert!(parity::compare(&reference, &silent).is_err());
    let mut invalid = a;
    invalid["complete"] = json!(false);
    fs::write(
        logged.join("run.json"),
        serde_json::to_vec(&invalid).unwrap(),
    )
    .unwrap();
    assert!(parity::compare(&reference, &logged).is_err());
}
#[test]
fn reader_rejects_malformed_or_incomplete_files() {
    let scratch = Scratch::new();
    let path = scratch.0.join("bad.jsonl");
    for bytes in [
        b"\n".as_slice(),
        b"{\"schema_version\":1",
        b"{\"schema_version\":999,\"complete\":false}\n",
        b"{}\n",
    ] {
        fs::write(&path, bytes).unwrap();
        let events = transcript::reference::read_events(&path)
            .unwrap()
            .collect::<std::io::Result<Vec<_>>>();
        assert!(events.is_err());
    }
}
