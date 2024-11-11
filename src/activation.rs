use std::str::FromStr;

use crate::Matrix;

#[derive(Debug)]
pub(crate) enum ActivationType {
    ReLu,
    ReLuLeaky,
    Sigmoid,
}

impl FromStr for ActivationType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ReLu" => Ok(ActivationType::ReLu),
            "ReLuLeaky" => Ok(ActivationType::ReLuLeaky),
            "Sigmoid" => Ok(ActivationType::Sigmoid),
            _ => Err(()),
        }
    }
}

pub struct Activation {
    pub(crate) func: fn(&Matrix) -> Matrix,
    pub(crate) derv: fn(&Matrix) -> Matrix,
    pub(crate) activation_type: ActivationType,
}

impl Activation {
    #[allow(non_snake_case)]
    pub fn ReLu() -> Self {
        Activation {
            func: |m| m.map(|x| x.max(0.)),
            derv: |m| m.map(|x| if x > 0. { 1. } else { 0. }),
            activation_type: ActivationType::ReLu,
        }
    }

    #[allow(non_snake_case)]
    pub fn ReLu_leaky() -> Self {
        Activation {
            func: |m| m.map(|x| if x > 0. { x } else { 0.01 * x }),
            derv: |m| m.map(|x| if x > 0. { 1. } else { 0.01 }),
            activation_type: ActivationType::ReLuLeaky,
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
            activation_type: ActivationType::Sigmoid,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
