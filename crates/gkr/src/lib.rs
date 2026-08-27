pub struct Mle {
    table: Vec<i32>,
}

impl Mle {
    pub fn new(eval: &[i32]) -> Self {
        Mle {
            table: eval.to_owned(),
        }
    }

    // g(0), g(1) for the round polynomial: sum of each half.
    pub fn round_sums(&self) -> (i32, i32) {
        let n = self.table.len();
        let (lower, upper) = self.table.split_at(n / 2);
        (lower.iter().sum(), upper.iter().sum())
    }

    pub fn fix_variable(&mut self, r: i32) {
        let n = self.table.len();
        let (lower, upper) = self.table.split_at_mut(n / 2);
        for i in 0..lower.len() {
            lower[i] = lower[i] + r * (upper[i] - lower[i]);
        }
        self.table.truncate(n / 2);
    }

    pub fn scalar(&self) -> i32 {
        assert_eq!(self.table.len(), 1);
        self.table[0]
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }
}

impl std::ops::Index<usize> for Mle {
    type Output = i32;
    fn index(&self, i: usize) -> &i32 {
        &self.table[i]
    }
}

pub fn mle(eval: &[i32], rs: &[i32]) -> i32 {
    assert_eq!(eval.len(), 1 << rs.len());
    let mut m = Mle::new(eval);
    for r in rs {
        m.fix_variable(*r);
    }
    m.scalar()
}

use std::cell::{Cell, UnsafeCell};
pub type Field = i32;

pub struct Challenge {
    frame_pointer: Cell<usize>,
    len: Cell<usize>,
    all: UnsafeCell<Box<[Field]>>,
}

impl Challenge {
    pub fn with_capacity(cap: usize) -> Self {
        Challenge {
            frame_pointer: Cell::new(0),
            len: Cell::new(0),
            all: UnsafeCell::new(vec![0; cap].into_boxed_slice()),
        }
    }

    pub fn get_challenge(&self) -> Field {
        let i = self.len.get();
        let val = i as Field;
        // SAFETY: `all` is fixed-size and never reallocated, so a raw
        // write can't invalidate a slice returned by `new_frame` — those
        // only ever cover `0..len` as of when they were taken, and `len`
        // only grows, so this write (at index `i == len`) never lands
        // inside a range any live slice covers. The pointer is derived
        // from a shared reference to the box so this never claims
        // exclusive access to the whole allocation, only to slot `i`.
        unsafe {
            let boxed: &Box<[Field]> = &*self.all.get();
            assert!(i < boxed.len(), "exceeded preallocated challenge capacity");
            let slot = boxed.as_ptr().add(i) as *mut Field;
            slot.write(val);
        }
        self.len.set(i + 1);
        val
    }

    pub fn new_frame(&self) -> &[Field] {
        let start = self.frame_pointer.get();
        let end = self.len.get();
        self.frame_pointer.set(end);
        // SAFETY: see `get_challenge` — this range is never written again.
        unsafe {
            let boxed: &Box<[Field]> = &*self.all.get();
            &boxed[start..end]
        }
    }
}
