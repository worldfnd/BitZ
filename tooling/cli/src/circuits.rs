//! Compiled-in circuit adapters and random benchmark instances.

use crate::end_to_end::{CircuitStatement, Error};
use anyhow::{Context, Result, ensure};
use circuit::{
    Circuit,
    sha256::{
        COMPRESSION_INPUT_BITS, INITIAL_STATE, SHA256_2KB_MESSAGE_BITS, Word, compress,
        compression_circuit, sha256_2kb_circuit, sha256_block_aligned_circuit,
    },
};
use rand::RngExt;
use sha2::{Digest, Sha256};
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltinCircuit {
    Sha256Compression,
    Sha256Chain,
    Sha256BlockAligned,
    Sha2562kb,
}

impl BuiltinCircuit {
    pub const ALL: [Self; 4] = [
        Self::Sha256Compression,
        Self::Sha256Chain,
        Self::Sha256BlockAligned,
        Self::Sha2562kb,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Sha256Compression => "sha256-compression",
            Self::Sha256Chain => "sha256-chain",
            Self::Sha256BlockAligned => "sha256-block-aligned",
            Self::Sha2562kb => "sha256-2kb",
        }
    }
}

impl fmt::Display for BuiltinCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for BuiltinCircuit {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|circuit| circuit.name() == value)
            .ok_or_else(|| format!("unknown circuit: {value}"))
    }
}

/// Public input bits and expected output bits in the selected circuit's ordering.
#[derive(Clone, Debug)]
pub struct CircuitInstance {
    pub circuit: BuiltinCircuit,
    pub inputs: Vec<bool>,
    pub output: Vec<bool>,
}

impl CircuitInstance {
    pub fn random(
        circuit: BuiltinCircuit,
        blocks: Option<usize>,
        initial_state: Option<[u32; 8]>,
    ) -> Result<Self> {
        let count = match circuit {
            BuiltinCircuit::Sha2562kb => {
                ensure!(
                    blocks.is_none() || blocks == Some(32),
                    "this circuit requires 32 message blocks"
                );
                32
            }
            _ => blocks.unwrap_or(1),
        };
        ensure!(
            circuit != BuiltinCircuit::Sha256Compression || count == 1,
            "compression requires one block"
        );
        ensure!(
            circuit != BuiltinCircuit::Sha256Chain || count > 0,
            "chain requires at least one block"
        );
        ensure!(
            initial_state.is_none() || circuit == BuiltinCircuit::Sha256Compression,
            "only compression accepts --initial-state"
        );
        let byte_count = count.checked_mul(64).context("message length overflow")?;
        let mut rng = rand::rng();
        let bytes: Vec<u8> = (0..byte_count).map(|_| rng.random()).collect();
        let (inputs, output) = match circuit {
            BuiltinCircuit::Sha256Compression | BuiltinCircuit::Sha256Chain => {
                let state = initial_state.unwrap_or(INITIAL_STATE);
                let mut digest = state;
                for block in bytes.chunks_exact(64) {
                    sha2::compress256(&mut digest, &[block.try_into().map(<[u8; 64]>::into)?]);
                }
                let mut inputs = word_bits(&bytes);
                if circuit == BuiltinCircuit::Sha256Compression {
                    inputs.extend(
                        state
                            .into_iter()
                            .flat_map(|word| (0..32).map(move |bit| word >> bit & 1 != 0)),
                    );
                }
                let output = digest
                    .into_iter()
                    .flat_map(|word| (0..32).map(move |bit| word >> bit & 1 != 0))
                    .collect();
                (inputs, output)
            }
            BuiltinCircuit::Sha256BlockAligned | BuiltinCircuit::Sha2562kb => {
                (stream_bits(&bytes), stream_bits(&Sha256::digest(&bytes)))
            }
        };
        Ok(Self {
            circuit,
            inputs,
            output,
        })
    }

    fn validate(&self) -> Result<(), Error> {
        let valid = match self.circuit {
            BuiltinCircuit::Sha256Compression => self.inputs.len() == COMPRESSION_INPUT_BITS,
            BuiltinCircuit::Sha256Chain => {
                !self.inputs.is_empty() && self.inputs.len().is_multiple_of(512)
            }
            BuiltinCircuit::Sha256BlockAligned => self.inputs.len().is_multiple_of(512),
            BuiltinCircuit::Sha2562kb => self.inputs.len() == SHA256_2KB_MESSAGE_BITS,
        };
        if !valid || self.output.len() != 256 {
            return Err(Error::Input("invalid circuit input/output dimensions"));
        }
        Ok(())
    }
}

impl CircuitStatement for CircuitInstance {
    fn domain(&self) -> &'static [u8] {
        self.circuit.name().as_bytes()
    }
    fn public_bytes(&self) -> Vec<u8> {
        let mut bytes = (self.inputs.len() as u64).to_le_bytes().to_vec();
        bytes.extend(
            self.inputs
                .iter()
                .chain(&self.output)
                .map(|bit| u8::from(*bit)),
        );
        bytes
    }
    fn input_bits(&self) -> usize {
        self.inputs.len()
    }
    fn synthesize<C: Circuit>(&self, cs: &mut C, inputs: &[C::Bool]) -> Result<(), Error> {
        self.validate()?;
        if inputs.len() != self.inputs.len() {
            return Err(Error::Input("wrong witness input length"));
        }
        for (bit, expected) in inputs.iter().zip(&self.inputs) {
            constrain_bit(cs, bit.clone(), *expected);
        }
        let output = match self.circuit {
            BuiltinCircuit::Sha256Compression => compression_circuit(
                cs,
                inputs
                    .try_into()
                    .map_err(|_| Error::Input("compression input length"))?,
            ),
            BuiltinCircuit::Sha256Chain => {
                let mut state = circuit::sha256::initial_state();
                for bits in inputs.chunks_exact(512) {
                    let block = std::array::from_fn(|word| {
                        Word::new(std::array::from_fn(|bit| bits[word * 32 + bit].clone()))
                    });
                    state = compress(cs, block, state).map(|value| value.word);
                }
                std::array::from_fn(|bit| state[bit / 32].bit(bit % 32))
            }
            BuiltinCircuit::Sha256BlockAligned => {
                sha256_block_aligned_circuit(cs, inputs.len(), |bit| inputs[bit].clone())
            }
            BuiltinCircuit::Sha2562kb => sha256_2kb_circuit(
                cs,
                inputs
                    .try_into()
                    .map_err(|_| Error::Input("2 KiB message length"))?,
            ),
        };
        for (bit, expected) in output.into_iter().zip(&self.output) {
            constrain_bit(cs, bit, *expected);
        }
        Ok(())
    }
}

fn constrain_bit<C: Circuit>(cs: &mut C, bit: C::Bool, expected: bool) {
    let value = cs.bitz::<1>(bit);
    let one = C::Z::<1>::from(C::Coefficient::<1>::from(1u64));
    let expected = C::Z::<1>::from(C::Coefficient::<1>::from(u64::from(expected)));
    cs.assert_r1c::<1>(one, value, expected);
}

fn stream_bits(bytes: &[u8]) -> Vec<bool> {
    bytes
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |bit| byte >> bit & 1 != 0))
        .collect()
}

fn word_bits(bytes: &[u8]) -> Vec<bool> {
    bytes
        .chunks_exact(4)
        .flat_map(|word| {
            word.iter()
                .rev()
                .flat_map(|byte| (0..8).map(move |bit| byte >> bit & 1 != 0))
        })
        .collect()
}
