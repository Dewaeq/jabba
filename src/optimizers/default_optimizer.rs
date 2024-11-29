use crate::Matrix;

use super::Optimizer;

#[derive(Default)]
pub struct DefaultOptimizer;

impl Optimizer for DefaultOptimizer {
    fn step(
        &mut self,
        learning_rate: f32,
        gradient: &Matrix,
        iteration: usize,
        index: usize,
        variables: &mut crate::Matrix,
    ) {
        *variables -= learning_rate * gradient;
    }
}
