//! Public SHA-256 compression and raw compression-chain statements.

use bitz_cli::end_to_end::{CircuitStatement, Error};
use circuit::{
    Circuit,
    sha256::{INITIAL_STATE, Word, compress},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sha256Circuit {
    Compression,
    Chain,
}

/// Every block and the final chaining value are public. No padding is added.
/// Compression accepts a public initial state; Chain starts at the standard IV.
#[derive(Clone, Debug)]
pub struct Sha256Statement {
    pub circuit: Sha256Circuit,
    pub blocks: Vec<[u32; 16]>,
    pub initial_state: [u32; 8],
    pub digest: [u32; 8],
}

impl Sha256Statement {
    pub fn input(&self) -> Vec<bool> {
        self.blocks
            .iter()
            .flatten()
            .flat_map(|word| (0..32).map(move |bit| word >> bit & 1 != 0))
            .collect()
    }

    fn validate(&self) -> Result<(), Error> {
        if self.blocks.is_empty() {
            return Err(Error::Input("SHA chain must contain at least one block"));
        }
        match self.circuit {
            Sha256Circuit::Compression if self.blocks.len() != 1 => {
                Err(Error::Input("compression requires one block"))
            }
            Sha256Circuit::Chain if self.initial_state != INITIAL_STATE => {
                Err(Error::Input("chain requires the standard IV"))
            }
            _ => Ok(()),
        }
    }
}

impl CircuitStatement for Sha256Statement {
    fn domain(&self) -> &'static [u8] {
        match self.circuit {
            Sha256Circuit::Compression => b"sha256-compression/v1",
            Sha256Circuit::Chain => b"sha256-chain/v1",
        }
    }

    fn public_bytes(&self) -> Vec<u8> {
        let mut bytes = (self.blocks.len() as u64).to_le_bytes().to_vec();
        bytes.extend(
            self.initial_state
                .iter()
                .chain(self.blocks.iter().flatten())
                .chain(self.digest.iter())
                .flat_map(|word| word.to_le_bytes()),
        );
        bytes
    }

    fn input_bits(&self) -> usize {
        self.blocks.len() * 512
    }

    fn synthesize<CS: Circuit>(&self, cs: &mut CS, inputs: &[CS::Bool]) -> Result<(), Error> {
        self.validate()?;
        if inputs.len() != self.input_bits() {
            return Err(Error::Input("wrong SHA input length"));
        }
        for (bit, expected) in inputs.iter().zip(self.input()) {
            constrain_bit(cs, bit.clone(), expected);
        }
        let mut state = self
            .initial_state
            .map(|word| Word::constant(u64::from(word)));
        for bits in inputs.chunks_exact(512) {
            let block = std::array::from_fn(|word| {
                Word::new(std::array::from_fn(|bit| bits[word * 32 + bit].clone()))
            });
            state = compress(cs, block, state).map(|value| value.word);
        }
        for (word, expected) in state.iter().zip(self.digest) {
            for bit in 0..32 {
                constrain_bit(cs, word.bit(bit), expected >> bit & 1 != 0);
            }
        }
        Ok(())
    }
}

fn constrain_bit<CS: Circuit>(cs: &mut CS, bit: CS::Bool, expected: bool) {
    let value = cs.bitz::<1>(bit);
    let expected = CS::Z::<1>::from(CS::Coefficient::<1>::from(u64::from(expected)));
    cs.assert_r1c::<1>(
        CS::Z::<1>::from(CS::Coefficient::<1>::from(1u64)),
        value,
        expected,
    );
}
