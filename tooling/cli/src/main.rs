use bitz_cli::{
    end_to_end::Prepared,
    sha256::{Sha256Circuit, Sha256Statement},
};
use circuit::sha256::INITIAL_STATE;
use std::{error::Error, path::PathBuf, time::Instant};

const USAGE: &str = "circuit-e2e --circuit sha256-compression|sha256-chain --blocks FILE --digest HEX [--initial-state HEX] [--threads N]\nFILE contains raw 64-byte blocks in stream order; no padding is added. HEX is eight big-endian u32 words (64 hex digits). Chain requires the standard SHA-256 IV.";

struct Options {
    circuit: Sha256Circuit,
    blocks: PathBuf,
    digest: [u32; 8],
    initial_state: [u32; 8],
    threads: Option<usize>,
}

fn words(hex: &str) -> Result<[u32; 8], Box<dyn Error>> {
    if hex.len() != 64 || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("expected 64 hexadecimal digits".into());
    }
    let mut words = [0; 8];
    for (i, word) in words.iter_mut().enumerate() {
        *word = u32::from_str_radix(&hex[i * 8..i * 8 + 8], 16)?;
    }
    Ok(words)
}

fn options() -> Result<Option<Options>, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (mut circuit, mut blocks, mut digest, mut threads) = (None, None, None, None);
    let mut initial_state = INITIAL_STATE;
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!("{USAGE}");
            return Ok(None);
        }
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--circuit" => {
                circuit = Some(match value.as_str() {
                    "sha256-compression" => Sha256Circuit::Compression,
                    "sha256-chain" => Sha256Circuit::Chain,
                    _ => return Err("unknown circuit".into()),
                })
            }
            "--blocks" => blocks = Some(PathBuf::from(value)),
            "--digest" => digest = Some(words(&value)?),
            "--initial-state" => initial_state = words(&value)?,
            "--threads" => {
                let count: usize = value.parse()?;
                if count == 0 {
                    return Err("thread count must be positive".into());
                }
                threads = Some(count);
            }
            _ => return Err(format!("unknown option: {flag}").into()),
        }
    }
    Ok(Some(Options {
        circuit: circuit.ok_or("--circuit is required")?,
        blocks: blocks.ok_or("--blocks is required")?,
        digest: digest.ok_or("--digest is required")?,
        initial_state,
        threads,
    }))
}

fn run(options: Options) -> Result<(), Box<dyn Error>> {
    let bytes = std::fs::read(options.blocks)?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(64) {
        return Err("block file must contain a positive multiple of 64 bytes".into());
    }
    let blocks: Vec<[u32; 16]> = bytes
        .chunks_exact(64)
        .map(|block| {
            std::array::from_fn(|i| {
                u32::from_be_bytes([
                    block[4 * i],
                    block[4 * i + 1],
                    block[4 * i + 2],
                    block[4 * i + 3],
                ])
            })
        })
        .collect();
    let count = blocks.len();
    let statement = Sha256Statement {
        circuit: options.circuit,
        blocks,
        initial_state: options.initial_state,
        digest: options.digest,
    };
    let inputs = statement.input();
    let started = Instant::now();
    let prepared = Prepared::new(statement)?;
    let setup = started.elapsed();
    let started = Instant::now();
    let witness = prepared.witness(&inputs)?;
    let witness_time = started.elapsed();
    let started = Instant::now();
    let data = prepared.commit(&witness)?;
    let commit = started.elapsed();
    let started = Instant::now();
    let proof = prepared.prove(witness, &data)?;
    let prove = started.elapsed();
    let started = Instant::now();
    prepared.verify(&proof)?;
    let verify = started.elapsed();
    println!(
        "circuit={:?} compressions={count} threads={} field=Q100 pcs=Fast hash=Blake3 verified=true",
        options.circuit,
        rayon::current_num_threads()
    );
    println!(
        "setup_ms={:.3} witness_ms={:.3} commit_ms={:.3} prove_ms={:.3} total_prove_ms={:.3} verify_ms={:.3}",
        setup.as_secs_f64() * 1000.,
        witness_time.as_secs_f64() * 1000.,
        commit.as_secs_f64() * 1000.,
        prove.as_secs_f64() * 1000.,
        (commit + prove).as_secs_f64() * 1000.,
        verify.as_secs_f64() * 1000.
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(options) = options()? else {
        return Ok(());
    };
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(threads) = options.threads {
        builder = builder.num_threads(threads);
    }
    builder.build_global()?;
    rayon::broadcast(|_| {});
    run(options)
}
