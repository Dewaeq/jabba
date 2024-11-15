use jabba::{
    activation::Activation,
    nn::{NNBuilder, NNOptions},
};
use nalgebra::dmatrix;

fn main() {
    let x_train = dmatrix![
        0., 1., 1., 0.;
        1., 0., 1., 0.;
    ];

    let y_train = dmatrix![
        1., 1., 0., 0.;
    ];

    let options = NNOptions {
        log_interval: Some(5000),
        ..Default::default()
    };

    let mut nn = NNBuilder::new(2)
        .options(options)
        .add_layer(2, Activation::sigmoid())
        .add_layer(1, Activation::sigmoid())
        .build();

    let loss = nn.train(&x_train, &y_train, &x_train, &y_train, 100_000);
    let prediction = nn.feed_forward(&x_train);

    println!("loss: {loss}");
    println!("{x_train} {prediction}");
}
