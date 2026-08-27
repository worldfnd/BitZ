//! The public instance shape and the gates that run before any proof work.

/// The seven low bits of a row index select the basis coefficients that one
/// packed field element carries.
pub const PACK_BITS: u32 = 7;

/// The commitment size window the opening parameters are fixed for.
pub const MIN_LOG_BITS: usize = 22;
/// The upper end of that window.
pub const MAX_LOG_BITS: usize = 35;

/// A shape one of the admissibility constraints rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    /// Fewer than seven row-index bits: a packed row would not fill one
    /// codeword position. The paper writes this count `t`.
    RowIndexTooNarrow,
    /// The total bit count falls outside `2^22..=2^35`.
    CommitmentSizeOutOfRange,
}

/// How the committed bits are laid out, as the two index widths.
///
/// One counts the bits of a column, the other the columns; the paper writes
/// them `t` and `s`. Geometry only: the modulus sits in [`crate::F2ZParams`],
/// with the one gate that couples the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    log_rows: usize,
    log_columns: usize,
}

impl Shape {
    /// Checks the geometric admissibility constraints and returns the shape.
    pub fn new(log_rows: usize, log_columns: usize) -> Result<Self, ShapeError> {
        if log_rows < PACK_BITS as usize {
            return Err(ShapeError::RowIndexTooNarrow);
        }
        // The row width alone is bounded first: the total is at least as
        // large, so an out-of-window row width is already a rejection, and the
        // later `MAX_LOG_BITS - log_rows` cannot underflow.
        if log_rows > MAX_LOG_BITS
            || log_columns > MAX_LOG_BITS - log_rows
            || log_rows + log_columns < MIN_LOG_BITS
        {
            return Err(ShapeError::CommitmentSizeOutOfRange);
        }

        Ok(Self {
            log_rows,
            log_columns,
        })
    }

    /// The number of row-index bits, the paper's `t`.
    pub fn log_rows(&self) -> usize {
        self.log_rows
    }

    /// The number of column-index bits, the paper's `s`.
    pub fn log_columns(&self) -> usize {
        self.log_columns
    }

    /// The bits indexing the whole committed bit table, the paper's `m = t + s`.
    pub fn log_bits(&self) -> usize {
        self.log_rows + self.log_columns
    }

    /// The bits indexing the packed field elements, the paper's `m_P = m - 7`.
    pub fn log_packed_len(&self) -> usize {
        self.log_bits() - PACK_BITS as usize
    }

    /// The number of bits in one column, which is also the number of
    /// grand-product factors per column.
    pub fn rows(&self) -> usize {
        1 << self.log_rows
    }

    /// The number of columns.
    pub fn columns(&self) -> usize {
        1 << self.log_columns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_dimensions_from_the_pair() {
        let shape = Shape::new(14, 21).unwrap();
        assert_eq!(shape.log_bits(), 35);
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
        // A row width outside the window, which would underflow the subtraction.
        assert_eq!(Shape::new(36, 0), Err(ShapeError::CommitmentSizeOutOfRange));
    }

    #[test]
    fn accepts_the_window_boundaries() {
        assert_eq!(Shape::new(7, 15).unwrap().log_bits(), MIN_LOG_BITS);
        assert_eq!(Shape::new(14, 21).unwrap().log_bits(), MAX_LOG_BITS);
    }
}
