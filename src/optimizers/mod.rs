use default_optimizer::DefaultOptimizer;

use crate::Matrix;

pub mod adam_optimizer;
pub mod default_optimizer;

pub trait Optimizer {
    fn boxed() -> Box<Self>
    where
        Self: Default,
    {
        Box::new(Self::default())
    }

    fn add_variables(&mut self, shape: (usize, usize)) -> usize {
        0
    }

    fn step(
        &mut self,
        learning_rate: f32,
        gradient: &Matrix,
        iteration: usize,
        index: usize,
        variables: &mut Matrix,
    );
}

impl Default for Box<dyn Optimizer> {
    fn default() -> Self {
        DefaultOptimizer::boxed()
    }
}
