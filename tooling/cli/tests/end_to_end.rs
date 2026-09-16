use bitz_cli::end_to_end::{CircuitStatement, Error, Prepared};
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
    let prepared = Prepared::new(PublicBit).unwrap();
    let witness = prepared.witness(&[true]).unwrap();
    let data = prepared.commit(&witness).unwrap();
    let proof = prepared.prove(witness, &data).unwrap();
    Prepared::new(PublicBit).unwrap().verify(&proof).unwrap();

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
