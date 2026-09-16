//! Shared execution and timing for circuit proof benchmarks.

use crate::end_to_end::{CircuitProofSystem, CircuitStatement, CircuitStats, Error};
use std::{
    fmt,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
pub struct Timings {
    pub circuit: CircuitStats,
    pub setup: Duration,
    pub witness: Duration,
    pub commit: Duration,
    pub prove: Duration,
    pub verify: Duration,
}

/// Runs setup, witness generation, commitment, proving, and verification.
/// Callers generate inputs and initialize worker threads before calling this.
pub fn run<S: CircuitStatement>(statement: S, inputs: &[bool]) -> Result<Timings, Error> {
    let started = Instant::now();
    let prepared = CircuitProofSystem::new(statement)?;
    let setup = started.elapsed();
    let circuit = prepared.stats();
    let started = Instant::now();
    let witness = prepared.witness(inputs)?;
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
    Ok(Timings {
        circuit,
        setup,
        witness: witness_time,
        commit,
        prove,
        verify,
    })
}

impl fmt::Display for Timings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "opening_path={:?} constraints={} assignment_bits={} committed_bits={} padded_committed_bits={} setup_ms={:.3} witness_ms={:.3} commit_ms={:.3} prove_ms={:.3} total_prove_ms={:.3} verify_ms={:.3}",
            self.circuit.opening_path,
            self.circuit.constraints,
            self.circuit.assignment_bits,
            self.circuit.committed_bits,
            self.circuit.padded_committed_bits,
            self.setup.as_secs_f64() * 1000.,
            self.witness.as_secs_f64() * 1000.,
            self.commit.as_secs_f64() * 1000.,
            self.prove.as_secs_f64() * 1000.,
            (self.commit + self.prove).as_secs_f64() * 1000.,
            self.verify.as_secs_f64() * 1000.
        )
    }
}
