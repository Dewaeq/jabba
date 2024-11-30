use crate::empty_like;
use crate::utils::{pow_to, sqrt, sqrt_to};
use crate::{utils::pow, Matrix};

use crate::optimizers::Optimizer;

#[derive(Default)]
pub struct AdamOptimizer {
    momentum: Vec<Matrix>,
    velocity: Vec<Matrix>,
}

impl Optimizer for AdamOptimizer {
    fn add_variables(&mut self, shape: (usize, usize)) -> usize {
        self.momentum.push(Matrix::zeros(shape.0, shape.1));
        self.velocity.push(Matrix::zeros(shape.0, shape.1));

        self.momentum.len() - 1
    }

    fn step(
        &mut self,
        learning_rate: f32,
        gradient: &Matrix,
        iteration: usize,
        index: usize,
        variables: &mut Matrix,
    ) {
        let beta_1 = 0.9f32;
        let beta_2 = 0.999f32;
        let epsilon = 1e-7f32;

        let step = (iteration + 1) as i32;
        let beta_1_power = beta_1.powi(step);
        let beta_2_power = beta_2.powi(step);

        let alpha = learning_rate * (1. - beta_2_power).sqrt() / (1. - beta_1_power);

        let m = &mut self.momentum[index];
        let v = &mut self.velocity[index];

        // use buffer to reduce allocations
        let mut buffer = unsafe { empty_like(gradient.shape()) };

        // calcutate new momentum
        gradient.sub_to(&*m, &mut buffer);
        buffer *= 1. - beta_1;
        *m += &buffer;

        // calculate new velocity
        pow_to(&gradient, 2, &mut buffer);
        buffer -= &*v;
        buffer *= 1. - beta_2;
        *v += &buffer;

        sqrt_to(&v, &mut buffer);
        buffer.add_scalar_mut(epsilon);

        *variables -= alpha * m.component_div(&buffer);
    }
}
