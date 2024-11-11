use nalgebra::{DMatrix, DVector};

pub mod activation;
mod layer;
pub mod nn;

pub type Column = DVector<f32>;
pub type Matrix = DMatrix<f32>;
