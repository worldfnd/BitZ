//! The public instance shape and the gates that run before any proof work.

/// The seven low bits of a row index select the basis coefficients that one
/// packed field element carries.
pub const PACK_BITS: u32 = 7;

/// The commitment size window the opening parameters are fixed for.
pub const MIN_M: usize = 22;
/// The upper end of that window.
pub const MAX_M: usize = 35;

/// A shape one of the admissibility constraints rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    /// `t < 7`: a packed row would not fill one codeword position.
    RowIndexTooNarrow,
    /// `m` falls outside `22..=35`.
    CommitmentSizeOutOfRange,
}

/// The instance shape `(t, s)`.
///
/// `t` indexes the bits of a column and `s` the columns. Geometry only: the
/// modulus sits in [`crate::F2ZConfig`], with the one gate that couples the
/// two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    t: usize,
    s: usize,
}

impl Shape {
    /// Checks the geometric admissibility constraints and returns the shape.
    pub fn new(t: usize, s: usize) -> Result<Self, ShapeError> {
        if t < PACK_BITS as usize {
            return Err(ShapeError::RowIndexTooNarrow);
        }
        // `t` alone is bounded first: `m >= t`, so an out-of-window `t` is
        // already a rejection, and the later `MAX_M - t` cannot underflow.
        if t > MAX_M || s > MAX_M - t || t + s < MIN_M {
            return Err(ShapeError::CommitmentSizeOutOfRange);
        }

        Ok(Self { t, s })
    }

    /// The number of row-index bits, `t`.
    pub fn t(&self) -> usize {
        self.t
    }

    /// The number of column-index bits, `s`.
    pub fn s(&self) -> usize {
        self.s
    }

    /// `m = t + s`: the bits indexing the whole committed bit table.
    pub fn m(&self) -> usize {
        self.t + self.s
    }

    /// `m_P = m - 7`: the bits indexing the packed field elements.
    pub fn packed_m(&self) -> usize {
        self.m() - PACK_BITS as usize
    }

    /// The number of bits in one column, `2^t`, which is also the number of
    /// grand-product factors per column.
    pub fn rows(&self) -> usize {
        1 << self.t
    }

    /// The number of columns, `2^s`.
    pub fn columns(&self) -> usize {
        1 << self.s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_dimensions_from_the_pair() {
        let shape = Shape::new(14, 21).unwrap();
        assert_eq!(shape.m(), 35);
        assert_eq!(shape.rows(), 1 << 14);
        assert_eq!(shape.columns(), 1 << 21);
    }

    #[test]
    fn rejects_a_row_index_narrower_than_the_pack_width() {
        assert_eq!(Shape::new(6, 16), Err(ShapeError::RowIndexTooNarrow));
        assert!(Shape::new(7, 15).is_ok());
    }

    #[test]
    fn rejects_a_commitment_size_outside_the_window() {
        // m = 21, then m = 36.
        assert_eq!(Shape::new(7, 14), Err(ShapeError::CommitmentSizeOutOfRange));
        assert_eq!(
            Shape::new(13, 23),
            Err(ShapeError::CommitmentSizeOutOfRange)
        );
        // `t` alone outside the window, which would underflow `MAX_M - t`.
        assert_eq!(Shape::new(36, 0), Err(ShapeError::CommitmentSizeOutOfRange));
    }

    #[test]
    fn accepts_the_window_boundaries() {
        assert_eq!(Shape::new(7, 15).unwrap().m(), MIN_M);
        assert_eq!(Shape::new(14, 21).unwrap().m(), MAX_M);
    }
}
