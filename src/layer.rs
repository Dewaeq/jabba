use nalgebra::DMatrix;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};

use crate::{activation::Activation, empty_like, Matrix};

pub(crate) struct Layer {
    pub bias: Matrix,
    pub weights: Matrix,
    pub activation: Activation,

    pub a: Matrix,
    pub z: Matrix,
}

impl Layer {
    pub(crate) fn new(num_inputs: usize, num_neurons: usize, activation: Activation) -> Self {
        Layer {
            bias: random_bias(num_neurons),
            weights: random_weights(num_neurons, num_inputs),
            activation,
            a: Matrix::zeros(num_neurons, 1),
            z: Matrix::zeros(num_neurons, 1),
        }
    }

    pub(crate) fn step(&mut self, data: &Matrix) -> Matrix {
        self.z = &self.weights * data;

        self.z
            .column_iter_mut()
            .for_each(|mut col| col += &self.bias);

        // TODO: find a nicer way to do this
        if self.a.ncols() != self.z.ncols() {
            self.a = unsafe { empty_like(&self.z) };
        }

        (self.activation.func)(&self.z, &mut self.a);

        self.a.clone()
    }

    pub(crate) fn back_propagate(
        &mut self,
        mut delta: Matrix,
        prev_a: &Matrix,
        learning_rate: f32,
    ) -> Matrix {
        let mut buffer = unsafe { empty_like(&self.z) };
        (self.activation.derv)(&self.z, &mut buffer);

        delta.component_mul_assign(&buffer);

        let dw = learning_rate * &delta * prev_a.transpose();
        let db = learning_rate * delta.column_sum();

        self.weights -= dw;
        self.bias -= db;

        let next_delta = self.weights.transpose() * delta;
        next_delta
    }
}

fn random_weights(num_neurons: usize, num_inputs: usize) -> Matrix {
    //let normal = Normal::new(0., (1. / (num_inputs as f32)).sqrt()).unwrap();
    let distr = Uniform::new(-0.02, 0.02);

    let mut rng = thread_rng();
    let weights = Matrix::from_fn(num_neurons, num_inputs, |_, _| distr.sample(&mut rng));

    weights
}

fn random_bias(num_neurons: usize) -> Matrix {
    Matrix::zeros(num_neurons, 1)
}
