use nalgebra::base::dimension::Dynamic;
use nalgebra::{DMatrix, DMatrixSlice, DMatrixSliceMut, SliceStorage, VecStorage, U1};

use rand::Rng;
use std::time::Instant;

pub type DynMatrix = DMatrix<f64>;
pub type DynMatrixSlice<'a> = DMatrixSlice<'a, f64>;
pub type DynMatrixSliceMut<'a> = DMatrixSliceMut<'a, f64>;

pub struct NeuralNetwork {
    layers: Vec<FullyConnectedNetworkLayer>,
    outputs: Vec<DynMatrix>,
}

impl NeuralNetwork {
    pub fn with_layers(
        input_shape: (usize, usize),
        layers: Vec<FullyConnectedNetworkLayer>,
    ) -> Self {
        let (rows, cols) = input_shape;
        let mut outputs = Vec::new();

        for layer in &layers {
            let (rows, cols) = layer.output_shape((rows, cols));
            //initialize with ones so bias is set automatically
            outputs.push(DynMatrix::repeat(rows + 1, cols, 1.0));
        }

        NeuralNetwork { outputs, layers }
    }

    pub fn forward_propagate(&mut self, input: &DynMatrix) -> &DynMatrix {
        self.layers
            .iter()
            .enumerate()
            .fold(&mut self.outputs, |outputs, (i, layer)| {
                let (ins, outs) = outputs.split_at_mut(i);
                if i == 0 {
                    layer.fwd(&input, &mut outs[0].rows_mut(1, outs[0].nrows() - 1));
                } else {
                    layer.fwd(&ins[i - 1], &mut outs[0].rows_mut(1, outs[0].nrows() - 1));
                }
                outputs
            });

        self.outputs.last().expect(
            "Did not find a last element in outputs should only happen with a nn with 0 layers",
        )
    }

    pub fn backward_propagate_error(&mut self, actual: &DynMatrix) {
        if let Some(first_layer) = self.layers.last_mut() {
            let output = self.outputs.last().expect("Empty Network");
            first_layer.errors = output
                .rows(1, output.nrows() - 1)
                .map(|i| i * (1.0 - i))
                .component_mul(&(output.rows(1, output.nrows() - 1) - actual));
        }

        for window in (0..self.layers.len()).collect::<Vec<_>>().windows(2).rev() {
            if let &[idx, prev_idx] = window {
                let cols = &self.layers[prev_idx].weights.ncols();

                let mut errors = &self.layers[prev_idx]
                    .weights
                    .columns(1, cols - 1)
                    .transpose()
                    * &self.layers[prev_idx].errors;

                errors = errors.zip_map(
                    &self.outputs[idx].rows(1, self.outputs[idx].nrows() - 1),
                    |f1, f2| f1 * f2 * (1.0 - f2),
                );
                self.layers[idx].errors = errors;
            }
        }
    }

    pub fn update_weights(&mut self, input: &DynMatrix, learning_rate: f64) {
        let diffs = &self.layers[0].errors * &input.transpose();
        self.layers[0].weights -= diffs * learning_rate;

        for idx in (1..self.layers.len()).collect::<Vec<_>>() {
            let diffs = &self.layers[idx].errors * &self.outputs[idx - 1].transpose();
            self.layers[idx].weights -= diffs * learning_rate;
        }
    }
}

pub struct FullyConnectedNetworkLayer {
    weights: DynMatrix,
    errors: DynMatrix,
}

impl FullyConnectedNetworkLayer {
    pub fn new(nodes: usize, weights: usize) -> Self {
        FullyConnectedNetworkLayer {
            //plus 1 because of bias weight
            weights: DynMatrix::from_fn(nodes, weights + 1, |_, _| {
                rand::thread_rng().gen::<f64>() / (nodes * weights) as f64
            }),
            errors: DynMatrix::zeros(nodes, 1),
        }
    }

    pub fn with_weights(nodes: usize, weights: usize, weight_data: &[f64]) -> Self {
        assert_eq!(nodes * (weights + 1), weight_data.len());
        FullyConnectedNetworkLayer {
            //plus 1 because of bias weight
            weights: DynMatrix::from_row_slice(nodes, weights + 1, weight_data),
            errors: DynMatrix::zeros(nodes, weights + 1),
        }
    }
    fn fwd(&self, input: &DynMatrix, output: &mut DynMatrixSliceMut) {
        let t0 = Instant::now();
        &self.weights.mul_to(input, output);

        output.apply(|f| self.activation(f));
    }

    fn activation(&self, input: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-input))
    }

    fn output_shape(&self, (rows, cols): (usize, usize)) -> (usize, usize) {
        (self.weights.nrows(), cols)
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::{DynMatrix, FullyConnectedNetworkLayer, NeuralNetwork};

    #[test]
    fn test_it_works() {
        let nl = FullyConnectedNetworkLayer::with_weights(
            4,
            2,
            &[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            // ^-- bias column
        );

        let input: DynMatrix = DynMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);

        let mut output = DynMatrix::zeros(4, 1);
        nl.fwd(
            &input,
            &mut output.slice_mut((0, 0), (output.nrows(), output.ncols())),
        );

        println!("res: {}", output);
    }

    #[test]
    fn test_big() {
        let nl = FullyConnectedNetworkLayer::new(100, 768);

        let input: DynMatrix = DynMatrix::new_random(769, 10);
        let mut output: DynMatrix = DynMatrix::zeros(100, 10);

        let t0 = std::time::Instant::now();
        nl.fwd(
            &input,
            &mut output.slice_mut((0, 0), (output.nrows(), output.ncols())),
        );
        let t1 = t0.elapsed();

        println!("res: {}", output);
        println!("in: {}µs", t1.as_micros());
        assert!(false);
    }

    #[test]
    fn test_network() {
        let mut nn = NeuralNetwork::with_layers(
            (2, 3),
            vec![
                FullyConnectedNetworkLayer::with_weights(
                    3,
                    2,
                    &[-1.0, 0.5, 0.3, 9.0, -1.0, 1.0, 1.0, 0.1, 0.1],
                ),
                FullyConnectedNetworkLayer::with_weights(
                    2,
                    3,
                    &[-1.0, 0.1, 0.1, 0.2, -1.0, 0.9, 0.9, 0.9],
                ),
            ],
        );

        let input = DynMatrix::from_row_slice(3, 3, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let out = nn.forward_propagate(&input);

        println!("{}", out);
    }

    #[test]
    fn example_network() {
        let mut nn = NeuralNetwork::with_layers(
            (2, 1),
            vec![
                FullyConnectedNetworkLayer::with_weights(
                    3,
                    2,
                    &[1.0, 1.0, 0.0, -2.0, 3.0, 4.0, 1.0, 3.0, 5.0],
                ),
                FullyConnectedNetworkLayer::with_weights(
                    2,
                    3,
                    &[1.0, 2.0, -3.0, 4.0, -1.0, -2.0, 3.0, -4.0],
                ),
            ],
        );

        let test_data = [
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]),
                DynMatrix::from_row_slice(2, 1, &[0.0, 0.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 0.0, 1.0]),
                DynMatrix::from_row_slice(2, 1, &[1.0, 1.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 1.0, 0.0]),
                DynMatrix::from_row_slice(2, 1, &[1.0, 1.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 1.0, 1.0]),
                DynMatrix::from_row_slice(2, 1, &[0.0, 0.0]),
            ),
        ];

        for _ in 1..10_000 {
            for (input, expected) in &test_data {
                let out = nn.forward_propagate(&input);
                nn.backward_propagate_error(&expected);
                nn.update_weights(&input, 1.0);
            }
        }

        for (input, expected) in &test_data {
            let out = nn.forward_propagate(&input);
            println!("{}\n{}\n========================", expected, out);
        }

        assert!(false);
    }

    #[test]
    fn example_network3() {
        let mut nn = NeuralNetwork::with_layers(
            (2, 1),
            vec![
                FullyConnectedNetworkLayer::with_weights(
                    3,
                    2,
                    &[1.0, 1.0, 0.0, -2.0, 2.0, 4.0, 1.0, 2.0, 5.0],
                ),
                FullyConnectedNetworkLayer::with_weights(
                    3,
                    3,
                    &[
                        1.0, 1.0, 0.0, -2.0, 3.0, 4.0, 1.0, 3.0, 5.0, -2.0, 1.0, -3.0,
                    ],
                ),
                FullyConnectedNetworkLayer::with_weights(
                    2,
                    3,
                    &[1.0, 2.0, -3.0, 4.0, -1.0, -2.0, 3.0, -4.0],
                ),
            ],
        );

        let test_data = [
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]),
                DynMatrix::from_row_slice(2, 1, &[0.0, 0.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 0.0, 1.0]),
                DynMatrix::from_row_slice(2, 1, &[1.0, 1.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 1.0, 0.0]),
                DynMatrix::from_row_slice(2, 1, &[1.0, 1.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 1.0, 1.0]),
                DynMatrix::from_row_slice(2, 1, &[0.0, 0.0]),
            ),
        ];

        for _ in 1..10_000 {
            for (input, expected) in &test_data {
                let out = nn.forward_propagate(&input);
                nn.backward_propagate_error(&expected);
                nn.update_weights(&input, 1.0);
            }
        }

        for (input, expected) in &test_data {
            let out = nn.forward_propagate(&input);
            println!("{}\n{}\n========================", expected, out);
        }

        assert!(false);
    }

    #[test]
    fn example_network2() {
        let mut nn = NeuralNetwork::with_layers(
            (2, 1),
            vec![
                FullyConnectedNetworkLayer::new(3, 2),
                FullyConnectedNetworkLayer::new(3, 3),
                FullyConnectedNetworkLayer::new(2, 3),
            ],
        );

        let test_data = [
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]),
                DynMatrix::from_row_slice(2, 1, &[0.0, 0.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 0.0, 1.0]),
                DynMatrix::from_row_slice(2, 1, &[1.0, 1.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 1.0, 0.0]),
                DynMatrix::from_row_slice(2, 1, &[1.0, 1.0]),
            ),
            (
                DynMatrix::from_row_slice(3, 1, &[1.0, 1.0, 1.0]),
                DynMatrix::from_row_slice(2, 1, &[0.0, 0.0]),
            ),
        ];

        for _ in 1..10_000 {
            for (input, expected) in &test_data {
                let out = nn.forward_propagate(&input);
                nn.backward_propagate_error(&expected);
                nn.update_weights(&input, 1.0);
            }
        }

        for (input, expected) in &test_data {
            let out = nn.forward_propagate(&input);
            println!("{}\n{}\n========================", expected, out);
        }

        assert!(false);
    }

    #[test]
    fn test_set() {
        let data = [
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            DynMatrix::from_row_slice(
                101,
                1,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
        ];

        let labels = [
            DynMatrix::from_row_slice(10, 1, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            DynMatrix::from_row_slice(10, 1, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        let mut nn = NeuralNetwork::with_layers(
            (100, 1),
            vec![
                FullyConnectedNetworkLayer::new(20, 100),
                FullyConnectedNetworkLayer::new(20, 20),
                FullyConnectedNetworkLayer::new(20, 20),
                FullyConnectedNetworkLayer::new(10, 20),
            ],
        );
        for i in 1..1000 {
            for (data, label) in data.iter().zip(&labels) {
                if (i == 25) {
                    print!("stop");
                }
                let out = nn.forward_propagate(&data);
                nn.backward_propagate_error(&label);
                nn.update_weights(&data, 1.0);
            }
        }

        for (data, label) in data.iter().zip(&labels) {
            let out = nn.forward_propagate(&data);
            println!("{}", out);
        }
        assert!(false);
    }
}
