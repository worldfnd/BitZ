use std::error::Error;

const USAGE: &str = "circuit-e2e --help\nCircuit proving CLI. No circuit adapters are registered.";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    Err("no circuit adapters are registered; use --help".into())
}
