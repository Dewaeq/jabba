use rand::thread_rng;
use rand_distr::{Distribution, Uniform};

use crate::{activation::Activation, empty_like, Matrix};

pub(crate) struct Layer {
    pub(crate) bias: Matrix,
    pub(crate) weights: Matrix,
    pub(crate) activation: Activation,

    pub(crate) a: Matrix,
    pub(crate) z: Matrix,
    pub(crate) vw: Matrix,
}

impl Layer {
    pub(crate) fn new(
        num_inputs: usize,
        num_neurons: usize,
        activation: Activation,
        batch_size: usize,
    ) -> Self {
        Layer {
            bias: random_bias(num_neurons),
            weights: random_weights(num_neurons, num_inputs),
            activation,
            a: Matrix::zeros(num_neurons, batch_size),
            z: Matrix::zeros(num_neurons, batch_size),
            vw: Matrix::zeros(num_neurons, num_inputs),
        }
    }

    pub(crate) fn step(&mut self, data: &Matrix) -> Matrix {
        // TODO: find a nicer way to do this
        if self.a.ncols() != data.ncols() {
            let shape = (self.z.nrows(), data.ncols());
            self.a = unsafe { empty_like(shape) };
            self.z = unsafe { empty_like(shape) };
        }

        self.weights.mul_to(data, &mut self.z);

        self.z
            .column_iter_mut()
            .for_each(|mut col| col += &self.bias);

        self.activation.func(&self.z, &mut self.a);

        self.a.clone()
    }

    pub(crate) fn back_propagate(
        &mut self,
        mut delta: Matrix,
        prev_a: &Matrix,
        learning_rate: f32,
    ) -> Matrix {
        let mut buffer = unsafe { empty_like(self.z.shape()) };
        self.activation.derv(&self.z, &mut buffer);

        delta.component_mul_assign(&buffer);

        let dw = &delta * prev_a.transpose();
        let db = delta.column_sum();

        // Momentum
        let beta = 0.9;
        self.vw = beta * &self.vw + learning_rate * &dw;

        self.weights -= &self.vw;
        self.bias -= learning_rate * db;

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
