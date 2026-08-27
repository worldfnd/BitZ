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
