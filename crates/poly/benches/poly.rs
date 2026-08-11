#[path = "poly/common.rs"]
mod common;
#[path = "poly/eq.rs"]
mod eq;
#[path = "poly/mle.rs"]
mod mle;
#[path = "poly/natural_eval.rs"]
mod natural_eval;

fn main() {
    divan::main();
}
