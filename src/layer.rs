use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::{activation::Activation, Matrix};

pub struct Layer {
    pub bias: Matrix,
    pub weights: Matrix,
    pub activation: Activation,

    pub a: Matrix,
    pub z: Matrix,
}

impl Layer {
    pub fn new(num_inputs: usize, num_neurons: usize, activation: Activation) -> Self {
        Layer {
            bias: random_bias(num_neurons),
            weights: random_weights(num_neurons, num_inputs),
            activation,
            a: Matrix::zeros(num_neurons, 0),
            z: Matrix::zeros(num_neurons, 0),
        }
    }

    pub fn step(&mut self, data: &Matrix) -> Matrix {
        self.z = &self.weights * data;
        self.z
            .column_iter_mut()
            .for_each(|mut col| col += &self.bias);
        self.a = (self.activation.func)(&self.z);

        self.a.clone()
    }
}

fn random_weights(num_neurons: usize, num_inputs: usize) -> Matrix {
    let normal = Normal::new(0., (1. / (num_inputs as f32)).sqrt()).unwrap();

    let mut rng = thread_rng();
    let weights = Matrix::from_fn(num_neurons, num_inputs, |_, _| normal.sample(&mut rng));

    weights
}

fn random_bias(num_neurons: usize) -> Matrix {
    let normal = Normal::new(0., 1.).unwrap();

    let mut rng = thread_rng();
    let bias = Matrix::from_fn(num_neurons, 1, |_, _| normal.sample(&mut rng));

    bias
}
