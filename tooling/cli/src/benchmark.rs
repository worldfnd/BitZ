//! Shared execution and timing for circuit proof benchmarks.

use crate::ProjectConstraint;
use crate::end_to_end::{CircuitProofSystem, CircuitStatement, CircuitStats, Error};
use circuit::matrix_products::StoredInteger;
use circuit::{BitWidth, IntoWords};
use common::{BitzClaimField, BitzConstraintRing};
use field::FieldWithDynamicModulus;
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
pub fn run<S, F, R, Proj>(statement: S, inputs: &[bool]) -> Result<Timings, Error>
where
    S: CircuitStatement,
    F: BitzClaimField + FieldWithDynamicModulus,
    F::Integer: BitWidth + IntoWords,
    R: BitzConstraintRing + for<'a> From<&'a StoredInteger>,
    for<'a> StoredInteger: From<&'a R>,
    Proj: ProjectConstraint<R, F>,
{
    let started = Instant::now();
    let prepared = CircuitProofSystem::<_, _, _, Proj>::new(statement)?;
    let setup = started.elapsed();
    let circuit = prepared.stats();
    tracing::info!(
        opening_path = ?circuit.opening_path,
        constraints = circuit.constraints,
        assignment_bits = circuit.assignment_bits,
        committed_bits = circuit.committed_bits,
        padded_committed_bits = circuit.padded_committed_bits,
        prime_bits = circuit.prime_bits,
        "Circuit prepared",
    );
    let started = Instant::now();
    let witness = prepared.witness(inputs)?;
    let witness_time = started.elapsed();
    let started = Instant::now();
    let data = prepared.commit(&witness)?;
    let commit = started.elapsed();
    let started = Instant::now();
    let proof = prepared.prove(witness, data)?;
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
            "opening_path={:?} constraints={} assignment_bits={} committed_bits={} padded_committed_bits={} prime_bits={} setup_ms={:.3} witness_ms={:.3} commit_ms={:.3} prove_ms={:.3} total_prove_ms={:.3} verify_ms={:.3}",
            self.circuit.opening_path,
            self.circuit.constraints,
            self.circuit.assignment_bits,
            self.circuit.committed_bits,
            self.circuit.padded_committed_bits,
            self.circuit.prime_bits,
            self.setup.as_secs_f64() * 1000.,
            self.witness.as_secs_f64() * 1000.,
            self.commit.as_secs_f64() * 1000.,
            self.prove.as_secs_f64() * 1000.,
            (self.commit + self.prove).as_secs_f64() * 1000.,
            self.verify.as_secs_f64() * 1000.
        )
    }
}
