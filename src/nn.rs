use std::time::{Duration, Instant};

use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator,
};

use crate::{
    activation::ActivationType,
    layer::Layer,
    optimizers::{Optimizer, OptimizerType, ThreadSafeOptimizer},
    Matrix,
};

pub struct NN {
    pub(crate) layers: Vec<Layer>,
    pub(crate) options: NNOptions,
    // pub(crate) optimizer: Box<dyn Optimizer>,
    pub(crate) optimizer: ThreadSafeOptimizer,

    test_accuracy: f32,
}

impl NN {
    pub(crate) fn new(
        layers: Vec<Layer>,
        options: NNOptions,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        Self {
            layers,
            options,
            optimizer: ThreadSafeOptimizer(optimizer),
            test_accuracy: 0.,
        }
    }

    pub fn feed_forward(&mut self, data: &Matrix) -> Matrix {
        let mut data = data.clone_owned();
        for layer in &mut self.layers {
            data = layer.step(&data);
        }

        data.clone()
    }

    pub fn feed_hard(&self, data: &Matrix) -> Vec<(Matrix, Matrix)> {
        let mut res = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (a, z) = layer.step_hard(data);
            res.push((a, z));
        }

        res
    }

    fn back_propagate(
        &mut self,
        x: &Matrix,
        label: &Matrix,
        predicted: &Matrix,
        learning_rate: f32,
        step: usize,
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

            let optimizer = &mut self.optimizer.0;
            delta = layer.back_propagate(delta, prev_a, learning_rate, optimizer, step);
        }
    }

    pub fn train(&mut self, x_train: &Matrix, y_train: &Matrix, x_test: &Matrix, y_test: &Matrix) {
        let num_samples = x_train.ncols();
        let batch_size = self.options.batch_size;

        assert!(num_samples % batch_size == 0);

        let start = Instant::now();

        for epoch in 0.. {
            let (mut gradients, costs): (Vec<_>, Vec<_>) = (0..num_samples)
                .step_by(batch_size)
                .par_bridge()
                .into_par_iter()
                .map(|i| {
                    let batch_x: Matrix = x_train.columns_range(i..(i + batch_size)).into();
                    let batch_y: Matrix = y_train.columns_range(i..(i + batch_size)).into();

                    let feedback = self.feed_hard(&batch_x);
                    let (predicted, _) = feedback.last().unwrap();

                    let cost = predicted - &batch_y;
                    let mut delta = cost.clone();
                    let mut gradients = vec![];

                    for (i, (layer, (_, z))) in self.layers.iter().zip(&feedback).enumerate().rev()
                    {
                        let prev_a = if i != 0 { &feedback[i - 1].0 } else { &batch_x };
                        let (dw, db, new_delta) = layer.back_prop(delta, prev_a, z);
                        delta = new_delta;
                        gradients.push((dw, db));
                    }

                    (gradients, cost)
                })
                .unzip();

            let (a, b) = gradients.split_first_mut().unwrap();
            for batch_result in b {
                for (i, (dw, db)) in batch_result.iter().enumerate() {
                    a[i].0 += dw;
                    a[i].1 += db;
                }
            }

            let lr = self.options.learning_rate;
            let n = x_train.ncols() as f32;
            for (i, (dw, db)) in a.iter().enumerate() {
                self.layers[i].weights -= lr * dw / n;
                self.layers[i].bias -= lr * db / n;

                // let step = epoch * batch_size;
                // self.optimizer.0.step(
                //     lr,
                //     dw,
                //     step,
                //     self.layers[i].weights_index,
                //     &mut self.layers[i].weights,
                // );
                // self.optimizer.0.step(
                //     lr,
                //     db,
                //     step,
                //     self.layers[i].bias_index,
                //     &mut self.layers[i].bias,
                // );
            }

            let current_loss = costs.par_iter().map(|c| c.norm_squared()).sum::<f32>();
            if self.options.log_interval.is_some_and(|x| epoch % x == 0) {
                self.log(epoch, start, current_loss, x_test, y_test);
            }

            if self.options.stop_condition.must_stop(
                current_loss,
                epoch,
                start.elapsed(),
                self.test_accuracy,
            ) {
                break;
            }
        }
    }

    fn log(
        &mut self,
        epoch: usize,
        start: Instant,
        current_loss: f32,
        x_test: &Matrix,
        y_test: &Matrix,
    ) {
        print!("\x1B[2J\x1B[1;1H");

        println!("current loss:\t{current_loss}");
        println!("epoch:\t\t{epoch}");
        println!(
            "epochs/sec:\t{}",
            epoch as f32 / start.elapsed().as_secs_f32()
        );

        if self.options.test {
            self.test_accuracy = self.test(x_test, y_test);
            println!("test accuracy:\t{}", self.test_accuracy);
        }
    }

    fn test(&mut self, x_test: &Matrix, y_test: &Matrix) -> f32 {
        let predicted = self.feed_forward(x_test);

        let mut num_correct = 0;

        for (p, y) in predicted.column_iter().zip(y_test.column_iter()) {
            let p_max = p.max();
            let y_max = y.max();

            let p_val = p.iter().position(|&x| x == p_max).unwrap();
            let y_val = y.iter().position(|&x| x == y_max).unwrap();

            num_correct += (p_val == y_val) as i32;
        }

        num_correct as f32 / x_test.ncols() as f32
    }
}

pub enum StopCondition {
    Loss(f32),
    Epoch(usize),
    Time(Duration),
    TestAccuracy(f32),
}

impl StopCondition {
    fn must_stop(&self, loss: f32, epoch: usize, time: Duration, test_accuracy: f32) -> bool {
        match self {
            StopCondition::Loss(l) => loss <= *l,
            StopCondition::Time(d) => time >= *d,
            StopCondition::Epoch(e) => epoch >= *e,
            StopCondition::TestAccuracy(t) => test_accuracy >= *t,
        }
    }
}

pub struct NNOptions {
    pub log_interval: Option<usize>,
    pub log_batches: bool,
    pub test: bool,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub stop_condition: StopCondition,
}

impl Default for NNOptions {
    fn default() -> Self {
        NNOptions {
            batch_size: 1,
            learning_rate: 0.1,
            log_interval: None,
            log_batches: false,
            test: false,
            stop_condition: StopCondition::Epoch(200),
        }
    }
}

#[derive(Default)]
pub struct NNBuilder {
    layers: Vec<Layer>,
    num_inputs: usize,
    options: NNOptions,
    optimizer_type: OptimizerType,
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

    pub fn optimizer(mut self, optimizer_type: OptimizerType) -> Self {
        self.optimizer_type = optimizer_type;
        self
    }

    pub fn add_layer(mut self, num_neurons: usize, activation_type: ActivationType) -> Self {
        let num_inputs = if let Some(layer) = self.layers.last() {
            layer.bias.nrows()
        } else {
            self.num_inputs
        };

        self.layers.push(Layer::new(
            num_inputs,
            num_neurons,
            activation_type.activation(),
            self.options.batch_size,
        ));

        self
    }

    pub fn build(mut self) -> NN {
        let mut optimizer = self.optimizer_type.optimizer();

        for layer in &mut self.layers {
            layer.init(&mut optimizer);
        }

        NN::new(self.layers, self.options, optimizer)
    }
}
