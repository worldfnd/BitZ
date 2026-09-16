use bitz_cli::end_to_end::{CircuitProofSystem, CircuitStatement, Error, OpeningPath};
use circuit::Circuit;

struct PublicBit;

impl CircuitStatement for PublicBit {
    fn domain(&self) -> &'static [u8] {
        b"test/public-bit/v1"
    }
    fn public_bytes(&self) -> Vec<u8> {
        vec![1]
    }
    fn input_bits(&self) -> usize {
        1
    }
    fn synthesize<CS: Circuit>(&self, cs: &mut CS, inputs: &[CS::Bool]) -> Result<(), Error> {
        let bit = cs.bitz::<1>(inputs[0].clone());
        let one = CS::Z::<1>::from(CS::Coefficient::<1>::from(1u64));
        cs.assert_r1c::<1>(one.clone(), bit, one);
        Ok(())
    }
}

#[test]
fn generic_driver_accepts_a_non_sha_circuit() {
    let prepared = CircuitProofSystem::new(PublicBit).unwrap();
    assert_eq!(prepared.stats().opening_path, OpeningPath::Direct);
    assert_eq!(prepared.stats().committed_bits, 2);
    let witness = prepared.witness(&[true]).unwrap();
    let data = prepared.commit(&witness).unwrap();
    let proof = prepared.prove(witness, &data).unwrap();
    CircuitProofSystem::new(PublicBit)
        .unwrap()
        .verify(&proof)
        .unwrap();

    let mut changed = proof.clone();
    changed.root.0[0] ^= 1;
    assert!(prepared.verify(&changed).is_err());
    let mut changed = proof.clone();
    changed.spartan.inner.round_polynomials[0][0] += field::FqDefault::from(1u128);
    assert!(prepared.verify(&changed).is_err());
    for hints in [false, true] {
        let mut changed = proof.clone();
        let bytes = if hints {
            &mut changed.opening.hints
        } else {
            &mut changed.opening.narg_string
        };
        bytes.push(0);
        assert!(prepared.verify(&changed).is_err());
    }
    assert!(prepared.witness(&[]).is_err());
    assert!(matches!(
        prepared.witness(&[false]),
        Err(Error::Unsatisfied)
    ));
}

#[test]
fn benchmark_runs_a_generic_circuit_and_propagates_failure() {
    let timings = bitz_cli::benchmark::run(PublicBit, &[true]).unwrap();
    let output = timings.to_string();
    assert!(output.contains("total_prove_ms="));
    assert!(output.contains("verify_ms="));
    assert!(matches!(
        bitz_cli::benchmark::run(PublicBit, &[false]),
        Err(Error::Unsatisfied)
    ));
}

struct PublicXor;

impl CircuitStatement for PublicXor {
    fn domain(&self) -> &'static [u8] {
        b"test/public-xor/v1"
    }
    fn public_bytes(&self) -> Vec<u8> {
        vec![1]
    }
    fn input_bits(&self) -> usize {
        2
    }
    fn synthesize<C: Circuit>(&self, cs: &mut C, inputs: &[C::Bool]) -> Result<(), Error> {
        let first = cs.bitz::<1>(inputs[0].clone());
        let xor = cs.xor(inputs[0].clone(), inputs[1].clone());
        let output = cs.bitz::<1>(xor);
        let one = C::Z::<1>::from(C::Coefficient::<1>::from(1u64));
        cs.assert_r1c::<1>(one.clone(), first.clone(), first);
        cs.assert_r1c::<1>(one.clone(), output, one);
        Ok(())
    }
}

#[test]
fn nonidentity_map_uses_virtual_opening_and_checks_xor_relation() {
    let system = CircuitProofSystem::new(PublicXor).unwrap();
    assert_eq!(system.stats().opening_path, OpeningPath::Virtual);
    assert_eq!(system.stats().assignment_bits, 3);
    assert_eq!(system.stats().committed_bits, 2);
    let witness = system.witness(&[true, false]).unwrap();
    let data = system.commit(&witness).unwrap();
    let proof = system.prove(witness, &data).unwrap();
    CircuitProofSystem::new(PublicXor)
        .unwrap()
        .verify(&proof)
        .unwrap();
    assert!(matches!(
        system.witness(&[true, true]),
        Err(Error::Unsatisfied)
    ));
    let mut changed = proof.clone();
    changed.root.0[0] ^= 1;
    assert!(system.verify(&changed).is_err());
    let mut changed = proof;
    changed.opening.narg_string.push(0);
    assert!(system.verify(&changed).is_err());
    let timings = bitz_cli::benchmark::run(PublicXor, &[true, false]).unwrap();
    assert_eq!(timings.circuit.opening_path, OpeningPath::Virtual);
}
