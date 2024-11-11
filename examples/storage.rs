use std::{env, io::Read};

use image::{imageops::FilterType, open};
use jabba::{storage::read_from, Matrix};

fn main() {
    let path = env::args().last().unwrap();

    let img = open(path)
        .unwrap()
        .resize(28, 28, FilterType::Triangle)
        .into_luma8();
    let inputs = img
        .as_raw()
        .bytes()
        .map(|x| x.unwrap() as f32 / 255.)
        .collect::<Vec<_>>();

    let x = Matrix::from_column_slice(784, 1, &inputs);
    let mut nn = read_from("examples/nn.txt").unwrap();

    let predicted = nn.feed_forward(&x);
    let v = predicted
        .as_slice()
        .iter()
        .position(|&x| x == predicted.max());

    println!("{predicted}");
    if let Some(value) = v {
        println!("{value}");
    }
}
