use crate::Matrix;

pub struct Activation {
    pub func: fn(&Matrix) -> Matrix,
    pub derv: fn(&Matrix) -> Matrix,
}

impl Activation {
    #[allow(non_snake_case)]
    pub fn ReLu() -> Self {
        Activation {
            func: |m| m.map(|x| x.max(0.)),
            derv: |m| m.map(|x| if x > 0. { 1. } else { 0. }),
        }
    }

    #[allow(non_snake_case)]
    pub fn ReLu_leaky() -> Self {
        Activation {
            func: |m| m.map(|x| if x > 0. { x } else { 0.01 * x }),
            derv: |m| m.map(|x| if x > 0. { 1. } else { 0.01 }),
        }
    }

    pub fn sigmoid() -> Self {
        Activation {
            func: |m| m.map(sigmoid),
            derv: |m| {
                m.map(|x| {
                    let s = sigmoid(x);
                    s * (1. - s)
                })
            },
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
