//! Their verifier on a proof produced elsewhere (f2z-pcs stage B):
//! `verify_e2e <dump-dir> [<prefix>] [--sampled]` rebuilds the statement from the dump's
//! `public.bin` and `meta.txt`, reads `<prefix>spartan.bin`,
//! `<prefix>narg.bin` and `<prefix>hints.bin` (default prefix `ours.`) plus
//! the root from `meta.txt`, and runs `Prepared::verify`. Exit status 0 on
//! acceptance.
use bitz_cli::end_to_end::{Prepared, PreparedSampled, Proof, SampledProof};
use common::Root;
use field::{Fq, FqDefault, FqRuntime};
use poly::ScaledMleEvaluationClaim;
use spartan::{OuterSumcheckProof, SpartanPiopProof, SumcheckProof};

#[path = "common/sha256.rs"]
mod sha256;
use sha256::{Sha256Circuit, Sha256Statement};

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sampled = args.iter().any(|a| a == "--sampled");
    let args: Vec<String> = args.into_iter().filter(|a| a != "--sampled").collect();
    let dir = std::path::PathBuf::from(&args[0]);
    let prefix = args.get(1).cloned().unwrap_or_else(|| "ours.".to_string());
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    rayon::broadcast(|_| {});
    let meta: std::collections::HashMap<String, String> = std::fs::read_to_string(dir.join("meta.txt"))
        .expect("meta.txt")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let read = |name: String| std::fs::read(dir.join(&name)).unwrap_or_else(|e| panic!("{name}: {e}"));

    // The statement from its public bytes: u64 block count, the initial
    // state, the blocks, the digest, all u32 LE.
    let public = read("public.bin".into());
    let count = u64::from_le_bytes(public[..8].try_into().unwrap()) as usize;
    let words: Vec<u32> = public[8..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(words.len(), 8 + 16 * count + 8);
    let statement = Sha256Statement {
        circuit: match meta["circuit"].as_str() {
            "sha256-compression" => Sha256Circuit::Compression,
            "sha256-chain" => Sha256Circuit::Chain,
            other => panic!("unknown circuit {other}"),
        },
        blocks: (0..count).map(|b| words[8 + 16 * b..8 + 16 * (b + 1)].try_into().unwrap()).collect(),
        initial_state: words[..8].try_into().unwrap(),
        digest: words[8 + 16 * count..].try_into().unwrap(),
    };

    // The proof from the three files.
    let nr: usize = meta["num_row_vars"].parse().unwrap();
    let nc: usize = meta["num_column_vars"].parse().unwrap();
    let spartan = read(format!("{prefix}spartan.bin"));
    let root = Root(unhex(&meta["root"]).try_into().unwrap());
    let opening = transcript::Proof {
        narg_string: read(format!("{prefix}narg.bin")),
        hints: read(format!("{prefix}hints.bin")),
    };
    // `decode::<Q>` reads the canonical residues; the sampled scheme installs
    // its prime first, so `Fq<RUNTIME>::from` reduces by the right modulus.
    fn decode<const Q: u128>(
        spartan: &[u8],
        nr: usize,
        nc: usize,
    ) -> (SpartanPiopProof<Fq<Q>>, ScaledMleEvaluationClaim<Fq<Q>>) {
        let mut at = 0usize;
        let mut next = || {
            let v = Fq::<Q>::from(u128::from_le_bytes(spartan[at..at + 16].try_into().unwrap()));
            at += 16;
            v
        };
        let outer_rounds: Vec<[Fq<Q>; 4]> = (0..nr).map(|_| [next(), next(), next(), next()]).collect();
        let (az, bz, cz) = (next(), next(), next());
        let inner_rounds: Vec<[Fq<Q>; 3]> = (0..nc).map(|_| [next(), next(), next()]).collect();
        let point: Vec<Fq<Q>> = (0..nc).map(|_| next()).collect();
        let scale = next();
        let value = next();
        (
            SpartanPiopProof {
                outer: OuterSumcheckProof {
                    sumcheck: SumcheckProof { round_polynomials: outer_rounds },
                    az_mle_claim: az,
                    bz_mle_claim: bz,
                    cz_mle_claim: cz,
                },
                inner: SumcheckProof { round_polynomials: inner_rounds },
            },
            ScaledMleEvaluationClaim::new(point.into_boxed_slice(), scale, value),
        )
    }

    let (setup, result, verify) = if sampled {
        let prime_bits: u32 = meta["prime_bits"].parse().unwrap();
        let started = std::time::Instant::now();
        let prepared = PreparedSampled::new(statement, prime_bits).expect("prepared");
        let setup = started.elapsed();
        let prime = prepared.prime_for(root);
        field::set_modulus(prime).expect("admissible prime");
        let (spartan, terminal) = decode::<{ field::RUNTIME }>(&spartan, nr, nc);
        let proof = SampledProof {
            root,
            prime,
            spartan,
            terminal,
            opening,
        };
        let started = std::time::Instant::now();
        let result = prepared.verify(&proof);
        (setup, result, started.elapsed())
    } else {
        let (spartan, _) = decode::<{ field::Q100 }>(&spartan, nr, nc);
        let proof = Proof {
            root,
            spartan,
            opening,
        };
        let started = std::time::Instant::now();
        let prepared = Prepared::new(statement).expect("prepared");
        let setup = started.elapsed();
        let started = std::time::Instant::now();
        let result = prepared.verify(&proof);
        (setup, result, started.elapsed())
    };
    let _ = FqDefault::from(0u128);
    let _ = FqRuntime::from(0u128);
    println!(
        "verify_e2e {}: {} (setup {:.1?}, verify {:.1?})",
        prefix,
        match &result {
            Ok(()) => "ACCEPTED".to_string(),
            Err(e) => format!("REJECTED: {e:?}"),
        },
        setup,
        verify
    );
    if result.is_err() {
        std::process::exit(1);
    }
}
