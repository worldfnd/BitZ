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

use std::cell::UnsafeCell;
pub type Field = i32;

pub struct Challenge {
    data: UnsafeCell<TriangularArray<Field>>,
}

pub struct TriangularArray<T> {
    data: Box<[T]>,
    len: usize,
}

impl<T: Default + Copy> TriangularArray<T> {
    pub fn with_capacity(rounds: usize) -> Self {
        let capacity = rounds * (rounds + 1) / 2;
        Self {
            data: vec![T::default(); capacity].into_boxed_slice(),
            len: 0,
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
        let start = r * (r + 1) / 2;
        let end = start + r + 1;
        &self.data[start..end]
    }
}

impl Challenge {
    pub fn with_capacity(rounds: usize) -> Self {
        Self {
            data: UnsafeCell::new(TriangularArray::with_capacity(rounds)),
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
    pub fn build_point(&self, round: usize) -> &[Field] {
        unsafe { (*self.data.get()).round(round as usize) }
    }

    pub fn into_inner(self) -> TriangularArray<Field> {
        self.data.into_inner()
    }
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
