use crate::{
    activation::{Activation, ActivationType},
    layer::Layer,
    nn::NN,
    optimizers::{adam_optimizer::AdamOptimizer, default_optimizer::DefaultOptimizer, Optimizer},
    Matrix,
};
use std::{fs, path::Path, str::FromStr};

pub fn write_to<P: AsRef<Path>>(path: P, nn: &NN) -> Result<(), std::io::Error> {
    fs::write(path, nn_to_string(nn))
}

pub fn read_from<P: AsRef<Path>>(path: P) -> Result<NN, std::io::Error> {
    let contents = fs::read_to_string(path)?;

    Ok(nn_from_string(&contents))
}

fn nn_to_string(nn: &NN) -> String {
    let mut contents = String::new();

    for layer in &nn.layers {
        contents.push_str("BEGIN:LAYER\n");
        contents.push_str(&matrix_to_string(&layer.bias));
        contents.push_str("\n");
        contents.push_str(&matrix_to_string(&layer.weights));
        contents.push_str("\n");
        contents.push_str(&format!("{:?}", layer.activation.activation_type));
        contents.push_str("\n");
        contents.push_str("END:LAYER\n");
    }

    contents.push_str(&format!("OPTIMIZER:{}", "adam"));

    contents
}

fn nn_from_string(string: &str) -> NN {
    let mut lines = string.lines();
    let mut layers = vec![];
    let mut optimizer: Box<dyn Optimizer> = DefaultOptimizer::boxed();

    while let Some(line) = lines.next() {
        if line.contains("BEGIN:LAYER") {
            let bias = matrix_from_string(lines.next().unwrap());
            let weights = matrix_from_string(lines.next().unwrap());
            let activation_type = ActivationType::from_str(lines.next().unwrap());
            let activation = match activation_type.unwrap() {
                ActivationType::ReLu => Activation::ReLu(),
                ActivationType::ReLuLeaky => Activation::ReLu_leaky(),
                ActivationType::Sigmoid => Activation::sigmoid(),
            };

            todo!("fix this");
            //layers.push(Layer {
            //bias,
            //weights,
            //activation,
            //a: Default::default(),
            //z: Default::default(),
            //vw: Default::default(),
            //});

            assert!(lines.next().unwrap().contains("END:LAYER"));
        } else if line.contains("OPTIMIZER:") {
            optimizer = match &line[10..] {
                "ADAM" => AdamOptimizer::boxed(),
                _ => DefaultOptimizer::boxed(),
            };
        }
    }

    NN::new(layers, Default::default(), optimizer)
}

fn matrix_to_string(m: &Matrix) -> String {
    let mut result = String::new();

    result.push_str(&m.nrows().to_string());
    result.push_str(" ");
    result.push_str(&m.ncols().to_string());
    result.push_str(" ");

    result.push_str(
        &m.as_slice()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" "),
    );

    result
}

fn matrix_from_string(s: &str) -> Matrix {
    let mut parts = s.split(' ');
    let rows = parts.next().unwrap().parse::<usize>().unwrap();
    let cols = parts.next().unwrap().parse::<usize>().unwrap();
    let data = parts.map(|x| x.parse::<f32>().unwrap()).collect::<Vec<_>>();

    Matrix::from_column_slice(rows, cols, &data)
}

#[cfg(test)]
mod tests {
    use crate::{
        activation::Activation,
        nn::NNBuilder,
        storage::{nn_from_string, nn_to_string},
    };

    #[test]
    fn test_nn_parser() {
        let nn = NNBuilder::new(2)
            .add_layer(2, Activation::sigmoid())
            .add_layer(1, Activation::sigmoid())
            .build();
        let parsed = nn_from_string(&nn_to_string(&nn));

        assert_eq!(nn.layers.len(), parsed.layers.len());

        for (l1, l2) in nn.layers.iter().zip(parsed.layers.iter()) {
            assert_eq!(l1.bias, l2.bias);
            assert_eq!(l1.weights, l2.weights);
        }
    }
}
