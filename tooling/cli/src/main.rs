use bitz_cli::{
    end_to_end::Prepared,
    sha256::{Sha256Circuit, Sha256Statement},
};
use circuit::sha256::INITIAL_STATE;
use rand::Rng;
use std::{error::Error, time::Instant};

const USAGE: &str = "circuit-e2e --circuit sha256-compression|sha256-chain [--num-blocks N] [--initial-state HEX] [--threads N]\nGenerates fresh random 64-byte blocks and computes the expected output before timing. No padding is added. N defaults to 1; compression requires exactly 1 block. HEX is eight big-endian u32 words (64 hex digits). Chain requires the standard SHA-256 IV.";

struct Options {
    circuit: Sha256Circuit,
    num_blocks: usize,
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
    let (mut circuit, mut threads) = (None, None);
    let mut num_blocks = 1;
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
            "--num-blocks" => num_blocks = value.parse()?,
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
    let circuit = circuit.ok_or("--circuit is required")?;
    if num_blocks == 0 {
        return Err("block count must be positive".into());
    }
    if circuit == Sha256Circuit::Compression && num_blocks != 1 {
        return Err("compression requires one block".into());
    }
    if circuit == Sha256Circuit::Chain && initial_state != INITIAL_STATE {
        return Err("chain requires the standard IV".into());
    }
    Ok(Some(Options {
        circuit,
        num_blocks,
        initial_state,
        threads,
    }))
}

fn run(options: Options) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::rng();
    let count = options.num_blocks;
    let mut digest = options.initial_state;
    let blocks = (0..count)
        .map(|_| {
            let bytes: [u8; 64] = std::array::from_fn(|_| rng.random());
            sha2::compress256(&mut digest, &[bytes.into()]);
            std::array::from_fn(|i| u32::from_be_bytes(std::array::from_fn(|j| bytes[4 * i + j])))
        })
        .collect();
    let statement = Sha256Statement {
        circuit: options.circuit,
        blocks,
        initial_state: options.initial_state,
        digest,
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
