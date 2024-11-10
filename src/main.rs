use activation::Activation;
use nalgebra::{dmatrix, DVector};
use nn::NNBuilder;

mod activation;
mod layer;
mod nn;

pub type Column = DVector<f32>;

pub const ALPHA: f32 = 0.1;

fn main() {
    let mut nn = NNBuilder::new(2)
        .add_layer(2, Activation::ReLu())
        .add_layer(1, Activation::ReLu())
        .build();

    let training_data = dmatrix![
        0., 1., 1.;
        1., 0., 1.;
        1., 1., 0.;
        0., 0., 0.;
    ];

    let x_train = training_data.columns_range(1..);
    let y_train = training_data.column(0);

    let mut epoch = 0;
    let mut loss = f32::MAX;

    while loss > 0.02 {
        let mut current_loss = 0.;
        for (i, x) in x_train.row_iter().enumerate() {
            let x = x.transpose();
            let label = y_train.row(i);
            let label = Column::from_element(1, label[0]);

            let prediction = nn.feed_forward(x.clone());

            current_loss += (&prediction - &label).norm_squared();

            nn.back_propagate(x, label, prediction);
        }
        loss = current_loss;

        if epoch % 1000 == 0 {
            println!("loss: {current_loss}");
        }

        epoch += 1;
    }

    //println!("///////////////////////////////");
    //for (i, x) in x_train.row_iter().enumerate() {
    //let x = x.transpose();
    //let label = y_train.row(i);
    //let label = Column::from_element(1, label[0]);

    //println!("x: {}", x);
    //println!("label: {}", label);

    //let prediction = nn.feed_forward(x);
    //println!("prediction: {}", prediction);
    //println!("///////////////////////////////");
    //}
}
