use bitz_cli::{
    circuits::{BuiltinCircuit, CircuitInstance},
    end_to_end::{CircuitProofSystem, CircuitStatement, Error},
};
use circuit::{
    constraints::ConstraintGenerator,
    sha256::{ABC_BLOCK, ABC_DIGEST, INITIAL_STATE},
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};

#[test]
fn sha_constraint_residuals_cannot_wrap_modulo_q100() {
    for circuit in BuiltinCircuit::ALL {
        let statement = CircuitInstance::random(circuit, None, None).unwrap();
        let mut generator = ConstraintGenerator::<BigInt>::new(statement.input_bits());
        let inputs: Vec<_> = (0..statement.input_bits())
            .map(|i| generator.input(i))
            .collect();
        statement.synthesize(&mut generator, &inputs).unwrap();
        let matrices = generator
            .into_matrices()
            .map_coefficients(|coefficient| coefficient.abs().to_u128().unwrap());
        for ((a, b), c) in matrices
            .a
            .rows()
            .iter()
            .zip(matrices.b.rows())
            .zip(matrices.c.rows())
        {
            // Each assignment entry is a bit. Absolute coefficient sums bound
            // every row evaluation, including malicious Boolean assignments.
            let bound = |row: &circuit::constraints::SparseRow<u128>| {
                row.entries()
                    .iter()
                    .try_fold(0u128, |sum, (_, coefficient)| sum.checked_add(*coefficient))
                    .unwrap()
            };
            let residual_bound = bound(a)
                .checked_mul(bound(b))
                .unwrap()
                .checked_add(bound(c))
                .unwrap();
            assert!(residual_bound < field::Q100, "{circuit}: residual may wrap");
        }
    }
}

#[test]
fn supported_sha_circuits_prove_and_verify() {
    for circuit in BuiltinCircuit::ALL {
        let statement = CircuitInstance::random(circuit, None, None).unwrap();
        let inputs = statement.inputs.clone();
        let system = CircuitProofSystem::new(statement).unwrap();
        let witness = system.witness(&inputs).unwrap();
        let data = system.commit(&witness).unwrap();
        let proof = system.prove(witness, &data).unwrap();
        system.verify(&proof).unwrap();
    }
}

#[test]
fn sha_compression_matches_abc_and_binds_public_values() {
    let bits = |words: &[u32]| {
        words
            .iter()
            .flat_map(|word| (0..32).map(move |bit| word >> bit & 1 != 0))
            .collect::<Vec<_>>()
    };
    let mut inputs = bits(&ABC_BLOCK);
    inputs.extend(bits(&INITIAL_STATE));
    let statement = CircuitInstance {
        circuit: BuiltinCircuit::Sha256Compression,
        inputs: inputs.clone(),
        output: bits(&ABC_DIGEST),
    };
    let system = CircuitProofSystem::new(statement.clone()).unwrap();
    let witness = system.witness(&inputs).unwrap();
    let data = system.commit(&witness).unwrap();
    let proof = system.prove(witness, &data).unwrap();
    CircuitProofSystem::new(statement.clone())
        .unwrap()
        .verify(&proof)
        .unwrap();
    let mut changed = statement.clone();
    changed.output[0] ^= true;
    let wrong_output = CircuitProofSystem::new(changed).unwrap();
    assert!(wrong_output.verify(&proof).is_err());
    assert!(matches!(
        wrong_output.witness(&inputs),
        Err(Error::Unsatisfied)
    ));
    let mut changed = statement;
    changed.inputs[0] ^= true;
    assert!(
        CircuitProofSystem::new(changed)
            .unwrap()
            .verify(&proof)
            .is_err()
    );
}

#[test]
fn variable_lengths_custom_state_and_invalid_dimensions() {
    for (circuit, count, state) in [
        (BuiltinCircuit::Sha256Chain, Some(3), None),
        (BuiltinCircuit::Sha256BlockAligned, Some(0), None),
        (BuiltinCircuit::Sha256BlockAligned, Some(2), None),
        (BuiltinCircuit::Sha256Compression, None, Some([0; 8])),
    ] {
        let statement = CircuitInstance::random(circuit, count, state).unwrap();
        let inputs = statement.inputs.clone();
        let system = CircuitProofSystem::new(statement).unwrap();
        system.witness(&inputs).unwrap();
    }
    for (circuit, count) in [
        (BuiltinCircuit::Sha256Compression, Some(2)),
        (BuiltinCircuit::Sha256Chain, Some(0)),
        (BuiltinCircuit::Sha2562kb, Some(1)),
        (BuiltinCircuit::Sha256Chain, Some(usize::MAX)),
    ] {
        assert!(CircuitInstance::random(circuit, count, None).is_err());
    }
    assert!(
        CircuitInstance::random(BuiltinCircuit::Sha256Chain, None, Some(INITIAL_STATE)).is_err()
    );
    let mut malformed =
        CircuitInstance::random(BuiltinCircuit::Sha256Compression, None, None).unwrap();
    malformed.inputs.pop();
    assert!(CircuitProofSystem::new(malformed).is_err());
}
