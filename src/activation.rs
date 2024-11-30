use std::str::FromStr;

use crate::Matrix;

#[derive(Debug)]
pub enum ActivationType {
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

impl ActivationType {
    pub(crate) fn activation(self) -> Activation {
        match self {
            Self::ReLu => Activation::ReLu(),
            Self::ReLuLeaky => Activation::ReLu_leaky(),
            Self::Sigmoid => Activation::sigmoid(),
        }
    }
}

pub(crate) struct Activation {
    pub(crate) activation_type: ActivationType,

    f: fn(f32) -> f32,
    df: fn(f32) -> f32,
}

impl Activation {
    pub fn func(&self, input: &Matrix, output: &mut Matrix) {
        apply(input, output, self.f);
    }

    pub fn derv(&self, input: &Matrix, output: &mut Matrix) {
        apply(input, output, self.df);
    }

    #[allow(non_snake_case)]
    pub fn ReLu() -> Self {
        Activation {
            f: |x| x.max(0.),
            df: |x| if x > 0. { 1. } else { 0. },
            activation_type: ActivationType::ReLu,
        }
    }

    #[allow(non_snake_case)]
    pub fn ReLu_leaky() -> Self {
        Activation {
            f: |x| if x > 0. { x } else { 0.01 * x },
            df: |x| if x > 0. { 1. } else { 0.01 },
            activation_type: ActivationType::ReLuLeaky,
        }
    }

    pub fn sigmoid() -> Self {
        Activation {
            f: |x| 1. / (1. + (-x).exp()),
            df: |x| {
                let s = 1. / (1. + (-x).exp());
                s * (1. - s)
            },
            activation_type: ActivationType::Sigmoid,
        }
    }
}

fn apply(m: &Matrix, buffer: &mut Matrix, f: fn(f32) -> f32) {
    buffer.iter_mut().zip(m).for_each(|(out, &x)| *out = f(x));
}
