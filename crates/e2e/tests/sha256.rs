//! SHA-256 proved through Spartan and F2Z.

use e2e::{Sha256Statement, shape_for, size_of};

/// A deterministic message of `blocks` 512-bit blocks.
fn message(blocks: usize) -> Vec<bool> {
    (0..blocks * 512)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) & 1 == 1)
        .collect()
}

#[test]
fn one_block_proves_and_verifies() {
    let statement = Sha256Statement::build(&message(1)).unwrap();
    let (proof, piop) = statement.prove().unwrap();
    statement.verify(&proof, &piop).unwrap();
}

#[test]
fn a_multi_block_message_proves_and_verifies() {
    let statement = Sha256Statement::build(&message(4)).unwrap();
    let (proof, piop) = statement.prove().unwrap();
    statement.verify(&proof, &piop).unwrap();
}

/// The shapes have to hold both witnesses, and the floor dominates until the
/// circuit is large enough to fill it.
#[test]
fn shapes_cover_the_circuit() {
    let size = size_of(512);
    let committed = shape_for(size.witness_bits).unwrap();
    let claim = shape_for(size.assignment_bits).unwrap();

    assert!(1usize << committed.log_bits() >= size.witness_bits);
    assert!(1usize << claim.log_bits() >= size.assignment_bits);
}

/// A proof of one message must not verify as a proof of another.
#[test]
fn a_proof_does_not_transfer_between_messages() {
    let proved = Sha256Statement::build(&message(1)).unwrap();
    let mut other_message = message(1);
    other_message[17] = !other_message[17];
    let other = Sha256Statement::build(&other_message).unwrap();

    let (proof, piop) = proved.prove().unwrap();
    assert!(other.verify(&proof, &piop).is_err());
}

/// The commitment binds the witness: verifying against a statement built from
/// a different witness must fail even though the circuit is identical.
#[test]
fn a_tampered_piop_proof_is_refused() {
    let statement = Sha256Statement::build(&message(1)).unwrap();
    let (proof, mut piop) = statement.prove().unwrap();
    piop.outer.az_mle_claim += field::FqDefault::from(1u128);

    assert!(statement.verify(&proof, &piop).is_err());
}
