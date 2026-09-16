use bitz_cli::{
    end_to_end::{CircuitStatement, Error, Prepared},
    sha256::{Sha256Circuit, Sha256Statement},
};
use circuit::{
    Circuit,
    sha256::{ABC_BLOCK, ABC_DIGEST, INITIAL_STATE},
};
use sha2::{Digest, Sha256};

fn compression() -> Sha256Statement {
    Sha256Statement {
        circuit: Sha256Circuit::Compression,
        blocks: vec![ABC_BLOCK],
        initial_state: INITIAL_STATE,
        digest: ABC_DIGEST,
    }
}

#[test]
fn compression_proof_binds_public_inputs_circuit_and_commitment() {
    let statement = compression();
    let inputs = statement.input();
    let prepared = Prepared::new(statement.clone()).unwrap();
    let witness = prepared.witness(&inputs).unwrap();
    let data = prepared.commit(&witness).unwrap();
    let proof = prepared.prove(witness, &data).unwrap();
    // A fresh verifier setup receives only the statement and proof.
    Prepared::new(statement.clone())
        .unwrap()
        .verify(&proof)
        .unwrap();

    let mut changed = statement.clone();
    changed.blocks[0][0] ^= 1;
    assert!(Prepared::new(changed).unwrap().verify(&proof).is_err());
    let mut changed = statement.clone();
    changed.digest[0] ^= 1;
    let wrong_digest = Prepared::new(changed).unwrap();
    assert!(wrong_digest.verify(&proof).is_err());
    assert!(matches!(
        wrong_digest.witness(&inputs),
        Err(Error::Unsatisfied)
    ));
    let mut changed = statement.clone();
    changed.initial_state[0] ^= 1;
    assert!(Prepared::new(changed).unwrap().verify(&proof).is_err());
    let mut changed = statement;
    changed.circuit = Sha256Circuit::Chain;
    assert!(Prepared::new(changed).unwrap().verify(&proof).is_err());

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
        assert!(!bytes.is_empty());
        bytes[0] ^= 1;
        assert!(prepared.verify(&changed).is_err());
        let mut changed = proof.clone();
        let bytes = if hints {
            &mut changed.opening.hints
        } else {
            &mut changed.opening.narg_string
        };
        bytes.push(0);
        assert!(prepared.verify(&changed).is_err());
    }
    let mut wrong_inputs = inputs;
    wrong_inputs[0] = !wrong_inputs[0];
    assert!(matches!(
        prepared.witness(&wrong_inputs),
        Err(Error::Unsatisfied)
    ));
    assert!(prepared.witness(&[]).is_err());
}

#[test]
fn two_block_chain_proves_the_digest_of_a_full_message() {
    let message: [u8; 64] = std::array::from_fn(|i| i as u8);
    let expected = Sha256::digest(message);
    let block =
        std::array::from_fn(|i| u32::from_be_bytes(message[4 * i..4 * i + 4].try_into().unwrap()));
    let mut padding = [0; 16];
    padding[0] = 0x80000000;
    padding[15] = 512;
    let statement = Sha256Statement {
        circuit: Sha256Circuit::Chain,
        blocks: vec![block, padding],
        initial_state: INITIAL_STATE,
        digest: std::array::from_fn(|i| {
            u32::from_be_bytes(expected[4 * i..4 * i + 4].try_into().unwrap())
        }),
    };
    let inputs = statement.input();
    let prepared = Prepared::new(statement.clone()).unwrap();
    let witness = prepared.witness(&inputs).unwrap();
    let data = prepared.commit(&witness).unwrap();
    let proof = prepared.prove(witness, &data).unwrap();
    prepared.verify(&proof).unwrap();
    let mut reversed = statement.clone();
    reversed.blocks.reverse();
    assert!(Prepared::new(reversed).unwrap().verify(&proof).is_err());
    let mut shortened = statement;
    shortened.blocks.pop();
    assert!(Prepared::new(shortened).unwrap().verify(&proof).is_err());
}

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
    prepared.verify(&proof).unwrap();
    assert!(matches!(
        prepared.witness(&[false]),
        Err(Error::Unsatisfied)
    ));
}

#[test]
fn invalid_sha_shapes_are_rejected() {
    let mut statement = compression();
    statement.blocks.clear();
    assert!(Prepared::new(statement).is_err());
    let mut statement = compression();
    statement.blocks.push(ABC_BLOCK);
    assert!(Prepared::new(statement).is_err());
    let mut statement = compression();
    statement.circuit = Sha256Circuit::Chain;
    statement.initial_state[0] ^= 1;
    assert!(Prepared::new(statement).is_err());
}
