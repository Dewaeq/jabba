use nalgebra::DMatrix;

pub mod activation;
pub mod nn;
pub mod storage;
pub mod utils;

mod layer;

pub type Matrix = DMatrix<f32>;
