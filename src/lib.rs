use nalgebra::{DMatrix, Dyn};

pub mod activation;
pub mod nn;
pub mod optimizers;
pub mod storage;
pub mod utils;

mod layer;

pub type Matrix = DMatrix<f32>;

pub(crate) unsafe fn empty_like(shape: (usize, usize)) -> Matrix {
    let (nrows, ncols) = shape;
    DMatrix::uninit(Dyn(nrows), Dyn(ncols)).assume_init()
}
