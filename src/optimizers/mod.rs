use adam_optimizer::AdamOptimizer;
use default_optimizer::DefaultOptimizer;

use crate::Matrix;

pub mod adam_optimizer;
pub mod default_optimizer;

#[derive(Default)]
pub enum OptimizerType {
    Adam,
    #[default]
    Default,
}

impl OptimizerType {
    pub fn optimizer(&self) -> Box<dyn Optimizer> {
        match self {
            Self::Adam => AdamOptimizer::boxed(),
            Self::Default => DefaultOptimizer::boxed(),
        }
    }
}

pub trait Optimizer {
    fn boxed() -> Box<Self>
    where
        Self: Default,
    {
        Box::new(Self::default())
    }

    fn add_variables(&mut self, shape: (usize, usize)) -> usize;

    fn step(
        &mut self,
        learning_rate: f32,
        gradient: &Matrix,
        step: usize,
        index: usize,
        variables: &mut Matrix,
    );
}

impl Default for Box<dyn Optimizer> {
    fn default() -> Self {
        DefaultOptimizer::boxed()
    }
}

pub struct ThreadSafeOptimizer(pub Box<dyn Optimizer>);
unsafe impl Sync for ThreadSafeOptimizer {}
