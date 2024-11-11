use rand::{thread_rng, Rng};

use crate::Matrix;

pub fn shuffle_rows(m: &mut Matrix) {
    let mut rng = thread_rng();

    for i in (1..m.nrows()).rev() {
        m.swap_rows(i, rng.gen_range(0..(i + 1)));
    }
}

pub fn one_hot(m: &Matrix) -> Matrix {
    let mut one_hot = Matrix::zeros(m.max() as usize + 1, m.ncols());

    one_hot
        .column_iter_mut()
        .enumerate()
        .for_each(|(i, mut col)| col[m[(0, i)] as usize] = 1.);

    one_hot
}
