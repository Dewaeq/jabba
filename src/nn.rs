use crate::{activation::Activation, layer::Layer, Column, ALPHA};

pub struct NN {
    layers: Vec<Layer>,
}

impl NN {
    pub fn feed_forward(&mut self, mut data: Column) -> Column {
        for layer in &mut self.layers {
            data = layer.step(&data);
        }

        data.clone()
    }

    pub fn back_propagate(&mut self, x: Column, label: Column, predicted: Column) {
        let cost = predicted - label;
        let mut delta = cost;

        let n = self.layers.len();

        for i in (0..n).rev() {
            let layer = &self.layers[i];
            delta.component_mul_assign(&(layer.activation.derv)(&layer.z));

            let dw = &delta * (if i == 0 { &x } else { &self.layers[i - 1].a }).transpose();
            let db = &delta;

            self.layers[i].bias -= ALPHA * db;
            self.layers[i].weights -= ALPHA * dw;

            delta = self.layers[i].weights.transpose() * delta;
        }
    }
}

pub struct NNBuilder {
    layers: Vec<Layer>,
    num_inputs: usize,
}

impl NNBuilder {
    pub fn new(num_inputs: usize) -> Self {
        NNBuilder {
            layers: vec![],
            num_inputs,
        }
    }

    pub fn add_layer(mut self, num_neurons: usize, activation: Activation) -> Self {
        let num_inputs = if let Some(layer) = self.layers.last() {
            layer.bias.shape().0
        } else {
            self.num_inputs
        };

        self.layers
            .push(Layer::new(num_inputs, num_neurons, activation));

        self
    }

    pub fn build(self) -> NN {
        NN {
            layers: self.layers,
        }
    }
}
