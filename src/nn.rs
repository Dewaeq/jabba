use crate::{activation::Activation, layer::Layer, Matrix};

pub struct NN {
    layers: Vec<Layer>,
}

impl NN {
    pub fn feed_forward(&mut self, data: &Matrix) -> Matrix {
        let mut data = data.clone_owned();
        for layer in &mut self.layers {
            data = layer.step(&data);
        }

        data.clone()
    }

    pub fn back_propagate(
        &mut self,
        x: &Matrix,
        label: &Matrix,
        predicted: &Matrix,
        learning_rate: f32,
    ) {
        let cost = predicted - label;
        let mut delta = cost;

        let n = self.layers.len();

        for i in (0..n).rev() {
            let layer = &self.layers[i];
            delta.component_mul_assign(&(layer.activation.derv)(&layer.z));

            let dw = &delta * (if i == 0 { x } else { &self.layers[i - 1].a }).transpose();
            let db = delta.column_sum();

            if i > 0 {
                // we have to use the old weights
                delta = self.layers[i].weights.transpose() * delta;
            }

            self.layers[i].bias -= learning_rate * db;
            self.layers[i].weights -= learning_rate * dw;
        }
    }

    pub fn train(
        &mut self,
        x_train: &Matrix,
        y_train: &Matrix,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
    ) -> f32 {
        let num_samples = x_train.shape().1;
        let mut loss = 0.;

        for epoch in 0..epochs {
            for i in (0..num_samples).step_by(batch_size) {
                let batch_x = x_train.columns_range(i..(i + batch_size).min(num_samples));
                let batch_y = y_train.columns_range(i..(i + batch_size).min(num_samples));
                let predicted = self.feed_forward(&batch_x.into());

                self.back_propagate(&batch_x.into(), &batch_y.into(), &predicted, learning_rate);
                loss += (predicted - batch_y).norm_squared();
            }

            //if epoch % 10_000 == 0 {
            //print!("\x1B[2J\x1B[1;1H");
            //println!("avg loss: {}", loss / epoch as f32);
            //println!("epoch {epoch}");
            //}
        }

        loss / epochs as f32
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
