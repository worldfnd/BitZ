use {
    super::Command,
    anyhow::{Result, ensure},
    argh::FromArgs,
    bitz_cli::{
        benchmark,
        circuits::{BuiltinCircuit, CircuitInstance},
    },
    pcs::SecurityLevel,
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

    /// classical PCS round budget: 100 (default) or 128; Spartan remains over Q100
    #[argh(
        option,
        default = "SecurityLevel::Bits100",
        from_str_fn(parse_pcs_security)
    )]
    pcs_security_bits: SecurityLevel,
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
                pcs_round_target_bits = self.pcs_security_bits.bits(),
                hash = "Blake3",
            ).entered();
            let statement = tracing::info_span!("generate_inputs").in_scope(|| {
                CircuitInstance::random(self.circuit, self.num_blocks, self.initial_state)
            })?;
            let inputs = statement.inputs.clone();
            let timings = benchmark::run(statement, &inputs, self.pcs_security_bits)?;
            tracing::info!("Proof verified successfully");
            println!("circuit={} threads={} field=Q100 pcs_round_target_bits={} security_model=classical-pcs-round-budget relation=Q100-r1cs constraints_verified=true", self.circuit, rayon::current_num_threads(), self.pcs_security_bits.bits());
            println!("{timings}");
            Ok(())
        })
    }
}

fn parse_pcs_security(value: &str) -> Result<SecurityLevel, String> {
    match value {
        "100" => Ok(SecurityLevel::Bits100),
        "128" => Ok(SecurityLevel::Bits128),
        _ => Err("expected a PCS round budget of 100 or 128 bits".into()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcs_budget_defaults_to_100_and_accepts_only_supported_values() {
        let args = Args::from_args(&["circuit-e2e"], &["--circuit", "sha256-compression"]).unwrap();
        assert_eq!(args.pcs_security_bits, SecurityLevel::Bits100);
        for (value, expected) in [
            ("100", SecurityLevel::Bits100),
            ("128", SecurityLevel::Bits128),
        ] {
            let args = Args::from_args(
                &["circuit-e2e"],
                &[
                    "--circuit",
                    "sha256-compression",
                    "--pcs-security-bits",
                    value,
                ],
            )
            .unwrap();
            assert_eq!(args.pcs_security_bits, expected);
        }
        for value in ["0", "99", "120", "129", "invalid"] {
            assert!(parse_pcs_security(value).is_err());
        }
    }
}
