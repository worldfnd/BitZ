mod cmd;

use {self::cmd::Command, anyhow::Result};

fn main() -> Result<()> {
    let args = argh::from_env::<cmd::Args>();
    args.run()
}
