use crate::Matrix;

use super::Optimizer;

#[derive(Default)]
pub struct DefaultOptimizer;

impl Optimizer for DefaultOptimizer {
    fn add_variables(&mut self, _shape: (usize, usize)) -> usize {
        0
    }

    fn step(
        &mut self,
        learning_rate: f32,
        gradient: &Matrix,
        _step: usize,
        _index: usize,
        variables: &mut crate::Matrix,
    ) {
        *variables -= learning_rate * gradient;
    }
}
