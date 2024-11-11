use nalgebra::DMatrix;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::{activation::Activation, Column};

pub struct Layer {
    pub bias: Column,
    pub weights: DMatrix<f32>,
    pub activation: Activation,

    pub a: DMatrix<f32>,
    pub z: DMatrix<f32>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_neurons: usize, activation: Activation) -> Self {
        Layer {
            bias: random_bias(num_neurons),
            weights: random_weights(num_neurons, num_inputs),
            activation,
            a: DMatrix::<f32>::zeros(num_neurons, 0),
            z: DMatrix::<f32>::zeros(num_neurons, 0),
        }
    }

    pub fn step(&mut self, data: &DMatrix<f32>) -> DMatrix<f32> {
        self.z = &self.weights * data;
        self.z
            .column_iter_mut()
            .for_each(|mut col| col += &self.bias);
        self.a = (self.activation.func)(&self.z);

        self.a.clone()
    }
}

fn random_weights(num_neurons: usize, num_inputs: usize) -> DMatrix<f32> {
    let normal = Normal::new(0., (1. / (num_inputs as f32)).sqrt()).unwrap();

    let mut rng = thread_rng();
    let weights = DMatrix::<f32>::from_fn(num_neurons, num_inputs, |_, _| normal.sample(&mut rng));

    weights
}
fn random_bias(num_neurons: usize) -> Column {
    let normal = Normal::new(0., 1.).unwrap();

    let mut rng = thread_rng();
    let bias = Column::from_fn(num_neurons, |_, _| normal.sample(&mut rng));

    bias
}
