use f2z_compat::{Witness, statement::*};
use transcript::reference::{Transcript, hex};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let s: serde_json::Value = serde_json::from_reader(std::fs::File::open(
        "/private/tmp/f2z-u32-plain-udr-reference/statement.json",
    )?)?;
    let cfg = Configuration::from_public_statement(&s)?;
    let w = Witness::deterministic();
    let pcs = pcs::Pcs::from_security_config(
        &common::Shape::new(15, 7).unwrap(),
        pcs::LigeritoProfile::Fast,
        cfg.policy(),
    )
    .map_err(|e| format!("{e:?}"))?;
    let (root, _data) = pcs.commit(w.packed()).map_err(|e| format!("{e:?}"))?;
    assert_eq!(hex(&root.0), s["public_instance"]["root_hex"]);
    assert_eq!(
        hex(&cfg.assignment_binding(&root.0)),
        s["assignment_binding_hex"]
    );
    assert_eq!(
        hex(&cfg.policy_digest()),
        s["configuration"]["ligerito"]["configuration_fingerprint"]
    );
    let mut t = Transcript::new(
        "prover",
        Some(std::path::Path::new("/private/tmp/benchmark-forest2.jsonl")),
    )?;
    let binding = cfg.bind(&root.0, &mut t)?;
    let f = spartan::reference::sample_prime(&mut t, cfg.prime_bounds().0, cfg.prime_bounds().1)?;
    let (proof, claim) = spartan::reference::prove(&w, &f, &binding, &mut t)?;
    let bitified = bitify(&f, &binding, &claim)?;
    t.absorb(
        "opening.statement",
        serde_json::json!({}),
        &OpeningStatement {
            root: &root.0,
            config: cfg.prover(),
            digest: &bitified.digest,
            bits: f.bits() as usize,
        },
    )?;
    let (forest, point, mu) = f2z_compat::forest::prove(&mut t, w.packed(), &bitified)?;
    t.finish()?;
    let mut v = Transcript::new("verifier", None)?;
    cfg.bind(&root.0, &mut v)?;
    let fv = spartan::reference::sample_prime(&mut v, cfg.prime_bounds().0, cfg.prime_bounds().1)?;
    let verified = spartan::reference::verify(&proof, &fv, &binding, &mut v)?;
    assert_eq!(verified.point, claim.point);
    assert_eq!(verified.scale, claim.scale);
    assert_eq!(verified.value, claim.value);
    let vb = bitify(&fv, &binding, &verified)?;
    v.absorb(
        "opening.statement",
        serde_json::json!({}),
        &OpeningStatement {
            root: &root.0,
            config: cfg.prover(),
            digest: &vb.digest,
            bits: fv.bits() as usize,
        },
    )?;
    let (vp, vm) = f2z_compat::forest::verify(&mut v, &forest, &vb)?;
    assert_eq!(vp, point);
    assert_eq!(vm, mu);
    println!(
        "root={} q={:x} bridge={}",
        hex(&root.0),
        f.modulus(),
        hex(&bitified.digest)
    );
    Ok(())
}
