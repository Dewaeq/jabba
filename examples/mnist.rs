use std::fs;

use jabba::{
    activation::Activation,
    nn::{NNBuilder, NNOptions, StopCondition},
    optimizers::{adam_optimizer::AdamOptimizer, Optimizer},
    storage,
    utils::one_hot,
    Matrix,
};

fn main() {
    let y_train = read_labels_file("C:/src/projects/jabba/examples/mnist/train-labels.idx1-ubyte");
    let x_train = read_images_file("C:/src/projects/jabba/examples/mnist/train-images.idx3-ubyte");

    let y_train = one_hot(&y_train);

    let y_test = read_labels_file("C:/src/projects/jabba/examples/mnist/t10k-labels.idx1-ubyte");
    let x_test = read_images_file("C:/src/projects/jabba/examples/mnist/t10k-images.idx3-ubyte");

    let y_test = one_hot(&y_test);

    let options = NNOptions {
        log_interval: Some(1),
        test: true,
        batch_size: 120,
        learning_rate: 0.001,
        decay_rate: 0.00006,
        stop_condition: StopCondition::TestAccuracy(0.98),
        ..Default::default()
    };

    let mut nn = NNBuilder::new(784)
        .options(options)
        .add_layer(200, Activation::ReLu_leaky())
        .add_layer(200, Activation::ReLu_leaky())
        .add_layer(10, Activation::ReLu_leaky())
        .optimizer(AdamOptimizer::boxed())
        .build();

    nn.train(&x_train, &y_train, &x_test, &y_test);

    storage::write_to("examples/mnist-nn.txt", &nn).unwrap()
}

fn read_labels_file(path: &str) -> Matrix {
    let bytes = fs::read(path).unwrap();
    let magic = to_u32(&bytes[0..4]);

    assert_eq!(magic, 2049);

    let num_items = to_u32(&bytes[4..8]) as usize;
    let m = Matrix::from_iterator(1, num_items, bytes[8..].iter().map(|&x| x as f32));

    m
}

fn read_images_file(path: &str) -> Matrix {
    let bytes = fs::read(path).unwrap();
    let magic = to_u32(&bytes[0..4]);

    assert_eq!(magic, 2051);

    let num_images = to_u32(&bytes[4..8]) as usize;
    let nrows = to_u32(&bytes[8..12]) as usize;
    let ncols = to_u32(&bytes[12..16]) as usize;

    assert_eq!(bytes.len(), 16 + num_images * nrows * ncols);

    let m = Matrix::from_iterator(
        nrows * ncols,
        num_images,
        bytes[16..].iter().map(|&x| x as f32 / 255.),
    );

    m
}

fn to_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().unwrap())
}
