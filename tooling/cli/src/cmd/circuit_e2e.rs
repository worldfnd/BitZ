use {
    super::Command,
    anyhow::{Result, bail},
    argh::FromArgs,
};

/// Run a circuit proof benchmark.
#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "circuit-e2e")]
pub struct Args {}

impl Command for Args {
    fn run(&self) -> Result<()> {
        bail!("no circuit adapters are registered")
    }
}
