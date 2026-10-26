//! Shared protocol security targets and grinding requirements.

/// The target for each classical PCS challenge block over `F128`.
///
/// This target does not certify the complete protocol or quantum security.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityLevel {
    Bits100,
    Bits128,
}

impl SecurityLevel {
    /// Returns the classical security target in bits.
    pub const fn bits(self) -> u32 {
        match self {
            Self::Bits100 => 100,
            Self::Bits128 => 128,
        }
    }

    /// Returns `ceil(log2(coefficient)) + target - 128`, bounded below by zero.
    ///
    /// This covers a challenge block with error at most `coefficient / 2^128`.
    /// A zero coefficient needs no grinding.
    pub const fn grinding_bits(self, coefficient: usize) -> u32 {
        if coefficient == 0 {
            return 0;
        }
        let log_coefficient = usize::BITS - (coefficient - 1).leading_zeros();
        (self.bits() + log_coefficient).saturating_sub(128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grinding_covers_the_integer_error_coefficient() {
        for (coefficient, expected) in [(0, 0), (1, 0), (2, 1), (3, 2), (7, 3), (8, 3), (15, 4)] {
            assert_eq!(SecurityLevel::Bits128.grinding_bits(coefficient), expected);
            assert_eq!(SecurityLevel::Bits100.grinding_bits(coefficient), 0);
        }
        assert_eq!(SecurityLevel::Bits100.grinding_bits(1 << 28), 0);
        assert_eq!(SecurityLevel::Bits100.grinding_bits((1 << 28) + 1), 1);
    }
}
