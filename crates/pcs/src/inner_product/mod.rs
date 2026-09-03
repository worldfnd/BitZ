//! Arbitrary `F128` inner products over the original committed bits.
//!
//! [`open`] and [`verify`] define the protocol roles. [`ring_switch`] contains
//! the pure reduction from original-bit weights to one packed Ligerito claim.

mod open;
mod ring_switch;
mod verify;

pub(crate) use open::open;
pub(crate) use verify::verify;

use crate::{CommitError, LigeritoProfile, Pcs};

fn validate_profile(pcs: &Pcs) -> Result<(), CommitError> {
    if pcs.params().profile != LigeritoProfile::Secure {
        return Err(CommitError::UnsupportedInnerProductProfile);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_inner_products_require_the_secure_profile() {
        let shape = common::Shape::new(7, 15).unwrap();
        let pcs = Pcs::new(&shape, LigeritoProfile::Fast, crate::HashKind::Blake3).unwrap();
        assert_eq!(
            validate_profile(&pcs),
            Err(CommitError::UnsupportedInnerProductProfile),
        );
    }
}
