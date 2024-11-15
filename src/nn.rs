use crate::{activation::Activation, layer::Layer, Matrix};

pub struct NN {
    pub(crate) layers: Vec<Layer>,
    pub(crate) options: NNOptions,
}

impl NN {
    pub fn feed_forward(&mut self, data: &Matrix) -> Matrix {
        let mut data = data.clone_owned();
        for layer in &mut self.layers {
            data = layer.step(&data);
        }

        data.clone()
    }

    fn back_propagate(
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
            let (layer, prev_a) = if i == 0 {
                (&mut self.layers[i], x)
            } else {
                let (left, right) = self.layers.split_at_mut(i);
                (&mut right[0], &left.last().unwrap().a)
            };

            delta = layer.back_propagate(delta, prev_a, learning_rate);
        }
    }

    pub fn train(&mut self, x_train: &Matrix, y_train: &Matrix, epochs: usize) -> f32 {
        let num_samples = x_train.ncols();
        let batch_size = self.options.batch_size;

        let mut total_loss = 0.;
        let mut learning_rate = self.options.learning_rate;

        for epoch in 0..epochs {
            let mut current_loss = 0.;
            for i in (0..num_samples).step_by(batch_size) {
                let batch_x = x_train.columns_range(i..(i + batch_size).min(num_samples));
                let batch_y = y_train.columns_range(i..(i + batch_size).min(num_samples));
                let predicted = self.feed_forward(&batch_x.into());

                self.back_propagate(&batch_x.into(), &batch_y.into(), &predicted, learning_rate);
                current_loss += (predicted - batch_y).norm_squared();
            }

            current_loss /= num_samples as f32;
            total_loss += current_loss;

            if self.options.log_interval.is_some_and(|x| epoch % x == 0) {
                print!("\x1B[2J\x1B[1;1H");
                println!("avg loss:\t{}", total_loss / epoch as f32);
                println!("current loss:\t{current_loss}");
                println!("epoch:\t\t{epoch}");
                println!("learning rate:\t{learning_rate}");
            }

            learning_rate = self.update_learning_rate(learning_rate, epoch, current_loss);
        }

        total_loss / epochs as f32
    }

    fn update_learning_rate(&self, learning_rate: f32, epoch: usize, loss: f32) -> f32 {
        learning_rate / (1. + self.options.decay_rate * epoch as f32)
    }
}

pub struct NNOptions {
    pub log_interval: Option<usize>,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub decay_rate: f32,
}

impl Default for NNOptions {
    fn default() -> Self {
        NNOptions {
            batch_size: 1,
            learning_rate: 0.1,
            log_interval: None,
            decay_rate: 0.,
        }
    }
}

#[derive(Default)]
pub struct NNBuilder {
    layers: Vec<Layer>,
    num_inputs: usize,
    options: NNOptions,
}

impl NNBuilder {
    pub fn new(num_inputs: usize) -> Self {
        NNBuilder {
            num_inputs,
            ..Default::default()
        }
    }

    pub fn options(mut self, options: NNOptions) -> Self {
        self.options = options;
        self
    }

    pub fn add_layer(mut self, num_neurons: usize, activation: Activation) -> Self {
        let num_inputs = if let Some(layer) = self.layers.last() {
            layer.bias.nrows()
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
            options: self.options,
        }
    }
}
