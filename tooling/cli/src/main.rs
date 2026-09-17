mod cmd;
mod span_stats;

use {
    self::cmd::Command,
    anyhow::{Context, Result},
    std::io::IsTerminal,
    tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt},
};

fn main() -> Result<()> {
    let args = argh::from_env::<cmd::Args>();
    if !args.quiet {
        let filter = EnvFilter::builder()
            .with_default_directive(tracing::Level::INFO.into())
            .from_env()
            .context("invalid RUST_LOG filter")?;
        let ansi = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        tracing_subscriber::registry()
            .with(filter)
            .with(span_stats::SpanStats::new(std::io::stderr, ansi))
            .try_init()?;
    }
    args.run()
}
