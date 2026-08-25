fn main() {}

type Field = i32;

// A circuit is defined by it's leaf value only because it is a balanced tree
struct Circuit {
    leafs: Vec<Field>,
}

impl Circuit {
    fn new(mut leafs: Vec<Field>) -> Self {
        leafs.resize(leafs.len().next_power_of_two(), 1);

        Circuit { leafs: leafs }
    }

    // Should an evaluation consume?
    fn eval(&self) -> CircuitEval {
        // Size can be precalculated
        let mut witnesses = Vec::with_capacity(self.leafs.len().ilog2() as usize);

        let mut prev_eval = self.leafs.clone();

        while prev_eval.len() > 1 {
            let mut eval = Vec::with_capacity(prev_eval.len() >> 1);
            for pair in prev_eval.chunks_exact(2) {
                eval.push(pair[0] * pair[1])
            }
            witnesses.push(prev_eval);
            prev_eval = eval;
        }

        witnesses.push(prev_eval);

        CircuitEval(witnesses)
    }
}

// Main purpose of the wrapper is to have the order denoted
struct CircuitEval(Vec<Vec<Field>>);

impl CircuitEval {
    // Return from the final layer back to the front
    // If circuit evaluation needs to stay an alternative is to wrap it in an iterator to keep track of the location
    fn pop(&mut self) -> Option<Vec<Field>> {
        self.0.pop()
    }
}

mod tests {
    
}
