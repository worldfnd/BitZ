//! SHA-256 compressions through the virtual pipeline: the mocked PIOP's claim
//! on the assignment, the fold, the grand product, the transposition onto the
//! committed bits, the sumcheck and the opening.

use field::Q100;
use pcs::LigeritoProfile;
use tests::Sha256Instance;

#[test]
fn a_batch_of_compressions_proves_and_verifies_against_the_committed_bits() {
    // `2^9` compressions: the smallest batch whose source table clears the
    // commitment floor.
    let instance = Sha256Instance::<Q100>::new(9, LigeritoProfile::Fast, 61);
    let batch = &instance.batch;
    assert_eq!(
        batch.source.len(),
        1 << (batch.source_shape().log_bits() - 7)
    );
    assert_eq!(
        batch.assignment.len(),
        1 << (batch.assignment_shape().log_bits() - 7)
    );

    let proof = instance.prove();
    instance.verify(&proof).expect("honest proof");
}
