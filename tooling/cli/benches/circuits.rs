//! End-to-end and per-stage benchmarks; instance generation is outside timing.

use bitz_cli::{
    ProjectBigIntToField, benchmark,
    circuits::{BuiltinCircuit, CircuitInstance},
    end_to_end::CircuitProofSystem,
};
use divan::Bencher;

type R = num_bigint::BigInt;
type F = field::DynField;
type Proj = ProjectBigIntToField;

fn main() {
    divan::main();
}

fn instance(circuit: BuiltinCircuit) -> (CircuitInstance, Vec<bool>) {
    let statement = CircuitInstance::random(circuit, None, None).unwrap();
    let inputs = statement.inputs.clone();
    (statement, inputs)
}

fn setup(circuit: BuiltinCircuit) -> (CircuitProofSystem<CircuitInstance, F, R, Proj>, Vec<bool>) {
    let (statement, inputs) = instance(circuit);
    (
        CircuitProofSystem::<_, F, _, Proj>::new(statement).unwrap(),
        inputs,
    )
}

#[divan::bench(args = BuiltinCircuit::ALL)]
fn end_to_end(bencher: Bencher, circuit: BuiltinCircuit) {
    bencher
        .with_inputs(|| instance(circuit))
        .bench_local_values(|(statement, inputs)| {
            benchmark::run::<_, F, R, Proj>(statement, &inputs).unwrap()
        });
}

#[divan::bench(args = BuiltinCircuit::ALL)]
fn circuit_setup(bencher: Bencher, circuit: BuiltinCircuit) {
    bencher
        .with_inputs(|| CircuitInstance::random(circuit, None, None).unwrap())
        .bench_local_values(|statement| {
            CircuitProofSystem::<_, F, _, Proj>::new(statement).unwrap()
        });
}

#[divan::bench(args = BuiltinCircuit::ALL)]
fn witness(bencher: Bencher, circuit: BuiltinCircuit) {
    let (system, inputs) = setup(circuit);
    bencher.bench_local(|| system.witness(&inputs).unwrap());
}

#[divan::bench(args = BuiltinCircuit::ALL)]
fn commit(bencher: Bencher, circuit: BuiltinCircuit) {
    let (system, inputs) = setup(circuit);
    let witness = system.witness(&inputs).unwrap();
    bencher.bench_local(|| system.commit(&witness).unwrap());
}

#[divan::bench(args = BuiltinCircuit::ALL)]
fn prove(bencher: Bencher, circuit: BuiltinCircuit) {
    let (system, inputs) = setup(circuit);
    bencher
        .with_inputs(|| {
            let witness = system.witness(&inputs).unwrap();
            let data = system.commit(&witness).unwrap();
            (witness, data)
        })
        .bench_local_values(|(witness, data)| system.prove(witness, data).unwrap());
}

#[divan::bench(args = BuiltinCircuit::ALL)]
fn verify(bencher: Bencher, circuit: BuiltinCircuit) {
    let (system, inputs) = setup(circuit);
    let witness = system.witness(&inputs).unwrap();
    let data = system.commit(&witness).unwrap();
    let proof = system.prove(witness, data).unwrap();
    bencher.bench_local(|| system.verify(&proof).unwrap());
}
