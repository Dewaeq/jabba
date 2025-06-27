use jabba::{
    activation::ActivationType,
    nn::{NNBuilder, NNOptions, StopCondition},
    optimizers::OptimizerType,
};
use nalgebra::dmatrix;

fn main() {
    let x_train = dmatrix![
        0., 1., 1., 0.;
        1., 0., 1., 0.;
    ];

    let y_train = dmatrix![
        0., 0., 1., 1.;
        1., 1., 0., 0.;
    ];

    let options = NNOptions {
        test: true,
        log_interval: Some(5000),
        batch_size: 2,
        learning_rate: 0.01,
        stop_condition: StopCondition::Loss(0.002),
        ..Default::default()
    };

    let mut nn = NNBuilder::new(2)
        .options(options)
        .add_layer(2, ActivationType::Sigmoid)
        .add_layer(2, ActivationType::Sigmoid)
        .optimizer(OptimizerType::Adam)
        .build();

    nn.train(&x_train, &y_train, &x_train, &y_train);
    let prediction = nn.feed_forward(&x_train);

    println!("{x_train} {prediction}");
}
