use {
    super::Command,
    anyhow::{Result, ensure},
    argh::FromArgs,
    bitz_cli::{
        benchmark,
        circuits::{BuiltinCircuit, CircuitInstance},
    },
};

/// Prove generated circuit constraints over Q100 and verify the proof.
#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "circuit-e2e")]
pub struct Args {
    /// sha256-compression, sha256-chain, sha256-block-aligned, or sha256-2kb
    #[argh(option)]
    circuit: BuiltinCircuit,

    /// random 64-byte input blocks (default 1, or 32 for 2kb circuits).
    /// Block-aligned SHA adds padding; compression and chain do not.
    #[argh(option)]
    num_blocks: Option<usize>,

    /// compression initial state as 64 hex digits (default standard SHA-256 IV)
    #[argh(option, from_str_fn(parse_state))]
    initial_state: Option<[u32; 8]>,

    /// positive Rayon worker count (default available parallelism)
    #[argh(option)]
    threads: Option<usize>,
}

impl Command for Args {
    fn run(&self) -> Result<()> {
        ensure!(self.threads != Some(0), "thread count must be positive");
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(threads) = self.threads {
            builder = builder.num_threads(threads);
        }
        let pool = builder.build()?;
        pool.broadcast(|_| {});
        pool.install(|| {
            // Enter on the worker: Rayon does not inherit the caller's current span.
            let _span = tracing::info_span!(
                "circuit_e2e",
                circuit = %self.circuit,
                threads = rayon::current_num_threads(),
                field = "Q100",
                pcs = "Fast",
                hash = "Blake3",
            ).entered();
            let statement = tracing::info_span!("generate_inputs").in_scope(|| {
                CircuitInstance::random(self.circuit, self.num_blocks, self.initial_state)
            })?;
            let inputs = statement.inputs.clone();
            let timings = benchmark::run(statement, &inputs)?;
            tracing::info!("Proof verified successfully");
            println!("circuit={} threads={} field=Q100 pcs=Fast hash=Blake3 relation=Q100-r1cs constraints_verified=true", self.circuit, rayon::current_num_threads());
            println!("{timings}");
            Ok(())
        })
    }
}

fn parse_state(hex: &str) -> Result<[u32; 8], String> {
    if hex.len() != 64 || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("expected 64 hexadecimal digits".into());
    }
    let mut words = [0; 8];
    for (i, word) in words.iter_mut().enumerate() {
        *word = u32::from_str_radix(&hex[i * 8..i * 8 + 8], 16).map_err(|err| err.to_string())?;
    }
    Ok(words)
}
