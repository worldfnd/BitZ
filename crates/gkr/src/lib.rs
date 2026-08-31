pub struct Mle {
    table: Vec<Field>,
}

impl Mle {
    pub fn new(eval: Vec<Field>) -> Self {
        Mle { table: eval }
    }

    // g(0), g(1) for the round polynomial: sum of each half.
    pub fn round_sums(&self) -> (Field, Field) {
        let n = self.table.len();
        let (lower, upper) = self.table.split_at(n / 2);
        (lower.iter().sum(), upper.iter().sum())
    }

    pub fn fix_variable(&mut self, r: Field) {
        let n = self.table.len();
        let (lower, upper) = self.table.split_at_mut(n / 2);
        for i in 0..lower.len() {
            lower[i] = lower[i] + r * (upper[i] - lower[i]);
        }
        self.table.truncate(n / 2);
    }

    pub fn scalar(&self) -> Field {
        assert_eq!(self.table.len(), 1);
        self.table[0]
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }
}

impl std::ops::Index<usize> for Mle {
    type Output = Field;
    fn index(&self, i: usize) -> &Field {
        &self.table[i]
    }
}

pub fn mle(eval: Vec<Field>, rs: impl Iterator<Item = Field> + ExactSizeIterator) -> Field {
    assert_eq!(eval.len(), 1 << rs.len());
    let mut m = Mle::new(eval);
    for r in rs {
        m.fix_variable(r);
    }
    m.scalar()
}

use std::cell::UnsafeCell;
pub type Field = i128;

pub struct Challenge {
    data: UnsafeCell<TriangularArray<Field>>,
}

/// Triangular arrays with a fixed extra data per round.
pub struct TriangularArray<T> {
    data: Box<[T]>,
    len: usize,
    lgroups: usize,
}

impl<T: Default + Copy> TriangularArray<T> {
    pub fn with_capacity(rounds: usize, groups: usize) -> Self {
        let lg = groups.ilog2() as usize;
        let capacity = rounds * (rounds + 1) / 2 + rounds * lg;
        Self {
            data: vec![T::default(); capacity].into_boxed_slice(),
            len: 0,
            lgroups: lg,
        }
    }

    pub fn push(&mut self, el: T) {
        // Preincrement because we don't want to have 0 as a possible challenge due to division.
        let index = self.len;
        self.data[index] = el;
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn round(&self, r: usize) -> &[T] {
        let start = r * (r + 1) / 2 + r * self.lgroups;
        let end_inclusive = start + r + self.lgroups;
        &self.data[start..=end_inclusive]
    }
}

pub struct Point<'a, T: Copy> {
    backend: &'a [T],
    state: PointState,
}

enum PointState {
    Start,
    Next(usize),
    Stop,
}

impl<'a, T: Copy> Point<'a, T> {
    pub fn new(backend: &'a [T]) -> Self {
        Self {
            backend: backend,
            state: if backend.len() != 0 {
                PointState::Start
            } else {
                PointState::Stop
            },
        }
    }

    // Will crash if called on empty
    pub fn sc(&self) -> (T, &[T]) {
        let selector_idx = self.backend.len() - 1;
        (self.backend[selector_idx], &self.backend[..selector_idx])
    }

    pub fn c(&self) -> &[T] {
        let selector_idx = self.backend.len().saturating_sub(1);
        &self.backend[..selector_idx]
    }
}

impl<'a, T: Copy> Iterator for Point<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self.state {
            PointState::Start => {
                let idx = self.backend.len() - 1;
                let val = self.backend[idx];
                self.state = if idx != 0 {
                    PointState::Next(0)
                } else {
                    PointState::Stop
                };
                Some(val)
            }
            PointState::Next(i) => {
                let val = self.backend[i];
                self.state = if i < self.backend.len() - 2 {
                    PointState::Next(i + 1)
                } else {
                    PointState::Stop
                };
                Some(val)
            }
            PointState::Stop => None,
        }
    }
}

impl<'a, T: Copy> ExactSizeIterator for Point<'a, T> {
    fn len(&self) -> usize {
        self.backend.len()
    }
}

impl Challenge {
    pub fn with_capacity(rounds: usize, groups: usize) -> Self {
        Self {
            data: UnsafeCell::new(TriangularArray::with_capacity(rounds, groups)),
        }
    }
    pub fn get_challenge(&self) -> Field {
        unsafe {
            let val = (*self.data.get()).len() as Field + 1;
            (*self.data.get()).push(val);
            val
        }
    }

    // new frame breaks when
    pub fn build_point<'a>(&'a self, round: usize) -> Point<'a, Field> {
        let backend = unsafe { (*self.data.get()).round(round as usize) };
        Point::new(backend)
    }

    // pub fn into_inner(self) -> TriangularArray<Field> {
    //     self.data.into_inner()
    // }
}

// pub struct BTreeArray<T> {
//     data: Box<[T]>,
//     len: usize,
// }

// impl<T: Default + Copy> BTreeArray<T> {
//     pub fn with_capacity(capacity: usize) -> Self {
//         BTreeArray {
//             data: vec![T::default(); capacity].into_boxed_slice(),
//             len: 0,
//         }
//     }

//     pub fn len(&self) -> usize {
//         self.len
//     }

//     // TODO capacity check?
//     pub fn push(&mut self, el: T) {
//         let index = self.len;
//         self.data[index] = el;
//         self.len += 1;
//     }

//     pub fn layer(&self, depth: u32) -> &[T] {
//         &self.data[2_usize.pow(depth) - 1..2_usize.pow(depth + 1) - 1]
//     }
// }
