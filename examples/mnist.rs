use std::{
    env,
    fs::{self, File},
    io::Read,
    ops::Sub,
};

use image::{imageops::FilterType, open};
use jabba::{
    activation::Activation,
    nn::{NNBuilder, NNOptions},
    storage,
    utils::{one_hot, shuffle_rows},
    Matrix,
};

fn main() {
    let data_path = env::args().last().unwrap();
    let mut data = read_data(&data_path);
    shuffle_rows(&mut data);

    let testing_data = data.rows_range(0..100).transpose();
    let training_data = data.rows_range(100..).transpose();

    let x_train = training_data.rows_range(1..).clone_owned();
    let y_train = one_hot(&training_data.rows_range(0..1).clone_owned());

    let x_test = testing_data.rows_range(1..).clone_owned();
    let y_test = one_hot(&testing_data.rows_range(0..1).clone_owned());

    let options = NNOptions {
        log_interval: Some(1),
        test_interval: Some(1),
        batch_size: 64,
        learning_rate: 0.003,
        decay_rate: 0.000006,
        ..Default::default()
    };

    let mut nn = NNBuilder::new(784)
        .options(options)
        .add_layer(200, Activation::sigmoid())
        .add_layer(200, Activation::sigmoid())
        .add_layer(10, Activation::sigmoid())
        .build();

    nn.train(&x_train, &y_train, &x_test, &y_test, 1000);

    File::create("nn.txt").unwrap();
    storage::write_to("nn.txt", &nn).unwrap()
}

fn read_data(path: &str) -> Matrix {
    let mut data = vec![];
    let mut num_rows = 0;

    for x in fs::read_dir(path).unwrap() {
        if let Ok(entry) = x {
            let label = entry.file_name().to_str().unwrap()[3..6]
                .parse::<f32>()
                .unwrap()
                .sub(1.);
            let img = open(entry.path())
                .unwrap()
                .resize(28, 28, FilterType::Triangle)
                .into_luma8();
            let mut inputs = img
                .as_raw()
                .bytes()
                .map(|x| x.unwrap() as f32 / 255.)
                .collect::<Vec<_>>();

            inputs.insert(0, label);
            data.extend(inputs);
            num_rows += 1;
        }
    }

    Matrix::from_row_slice(num_rows, 28 * 28 + 1, &data)
}
