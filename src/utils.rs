use rand::{thread_rng, Rng};

use crate::Matrix;

pub fn shuffle_rows(m: &mut Matrix) {
    let mut rng = thread_rng();

    for i in (1..m.nrows()).rev() {
        m.swap_rows(i, rng.gen_range(0..(i + 1)));
    }
}

/// One of m's axes should be of length one
/// eg. m is either of shape (n, 1) or (1, n)
pub fn one_hot(m: &Matrix) -> Matrix {
    if m.ncols() == 1 {
        let mut one_hot = Matrix::zeros(m.max() as usize + 1, m.nrows());

        one_hot
            .row_iter_mut()
            .enumerate()
            .for_each(|(i, mut row)| row[m[(i, 0)] as usize] = 1.);

        one_hot
    } else {
        let mut one_hot = Matrix::zeros(m.max() as usize + 1, m.ncols());

        one_hot
            .column_iter_mut()
            .enumerate()
            .for_each(|(i, mut col)| col[m[(0, i)] as usize] = 1.);

        one_hot
    }
}

#[allow(unused)]
pub(crate) fn pow(m: &Matrix, p: i32) -> Matrix {
    m.map(|x| x.powi(p))
}

pub(crate) fn pow_to(m: &Matrix, p: i32, buffer: &mut Matrix) {
    buffer
        .as_mut_slice()
        .iter_mut()
        .zip(m)
        .for_each(|(x, y)| *x = y.powi(p));
}

#[allow(unused)]
pub(crate) fn sqrt(m: &Matrix) -> Matrix {
    m.map(|x| x.sqrt())
}

pub(crate) fn sqrt_to(m: &Matrix, buffer: &mut Matrix) {
    buffer.as_mut_slice().iter_mut().zip(m).for_each(|(x, y)| {
        *x = y.sqrt();
    });
}
