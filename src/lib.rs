use nalgebra::DMatrix;

pub mod activation;
pub mod nn;
pub mod storage;
pub mod utils;

mod layer;

pub type Matrix = DMatrix<f32>;

pub(crate) unsafe fn empty_like(m: &Matrix) -> Matrix {
    let (nrows, ncols) = m.shape_generic();
    DMatrix::uninit(nrows, ncols).assume_init()
}
