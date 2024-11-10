use crate::Column;

pub struct Activation {
    pub func: fn(&Column) -> Column,
    pub derv: fn(&Column) -> Column,
}

impl Activation {
    #[allow(non_snake_case)]
    pub fn ReLu() -> Self {
        Activation {
            func: |col| col.map(|x| x.max(0.)),
            derv: |col| col.map(|x| if x > 0. { 1. } else { 0. }),
        }
    }

    #[allow(non_snake_case)]
    pub fn ReLu_leaky() -> Self {
        Activation {
            func: |col| col.map(|x| if x > 0. { x } else { 0.01 * x }),
            derv: |col| col.map(|x| if x > 0. { 1. } else { 0.01 }),
        }
    }

    pub fn sigmoid() -> Self {
        Activation {
            func: |col| col.map(sigmoid),
            derv: |col| {
                col.map(|x| {
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
