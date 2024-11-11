use nalgebra::DMatrix;

pub mod activation;
pub mod nn;
pub mod storage;

mod layer;

pub type Matrix = DMatrix<f32>;
