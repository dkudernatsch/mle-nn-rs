mod images;
mod nn;
mod parser;

use std::error::Error;
use std::io::{BufReader, BufWriter, Read, Write};

use crate::nn::{DynMatrix, FullyConnectedNetworkLayer, NeuralNetwork};
use crate::parser::{mnist_image_parser, mnist_label_parser};

use crate::images::Image;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Loading 10K test images ...");
    let test_data = load_test_data()?;
    println!("Loading 60K training images ...");
    let train_data = load_train_data()?;
    println!("Loading done.");

    let mut nn: NeuralNetwork = NeuralNetwork::with_layers(
        (784, 1),
        vec![
            FullyConnectedNetworkLayer::new(100, 784),
            FullyConnectedNetworkLayer::new(30, 100),
            FullyConnectedNetworkLayer::new(10, 30),
        ],
    );

    let mut error = 1.0;
    let mut gen = 0;

    while error > 0.00020 {
        println!("Starting generation ({:001})", gen);
        let mut outs = Vec::new();
        let mut actuals = Vec::new();

        for image in &test_data {
            outs.push(nn.forward_propagate(&image.data).clone());
            nn.backward_propagate_error(&image.label);
            actuals.push(image.label.clone());
            nn.update_weights(&image.data, 0.05);
        }

        println!(
            "Finished gen with train error: {}",
            mean_sqaured_error(&outs, &actuals)
        );

        if gen % 5 == 0 {
            let mut outs = Vec::new();
            let mut actuals = Vec::new();

            for image in &test_data {
                outs.push(nn.forward_propagate(&image.data).clone());
                actuals.push(image.label.clone());
            }
            error = mean_sqaured_error(&outs, &actuals);
            println!("Current test error: {}", error);
        }
        gen += 1;
    }
    let mut output = String::new();
    output.push_str(&format!("actual,expected\n"));
    for Image { data, label } in test_data {
        let result = nn.forward_propagate(&data);

        output.push_str(&format!(
            "{},{}\n",
            result.rows(1, result.nrows() - 1).iamax_full().0,
            label.iamax_full().0
        ));
    }

    let mut f = std::fs::File::create("ouput.csv")?;
    f.write(&output.as_bytes())?;
    Ok(())
}

fn mean_sqaured_error(predicteds: &[DynMatrix], actuals: &[DynMatrix]) -> f64 {
    let mut error = 0.0;

    for (actual, predicted) in actuals.iter().zip(predicteds.iter()) {
        let diff = actual - (predicted.rows(1, predicted.nrows() - 1));
        let square_diff = diff.component_mul(&diff);
        error += square_diff.column_sum()[0];
    }
    error / actuals.len() as f64
}

fn load_train_data() -> Result<Vec<Image>, Box<dyn Error>> {
    let train_image_path = "data/train-images-idx3-ubyte";
    let train_label_path = "data/train-labels-idx1-ubyte";

    let image_data = load_data(train_image_path)?;
    let image_labels = load_data(train_label_path)?;

    parse_images(&image_data, &image_labels)
}

fn load_test_data() -> Result<Vec<Image>, Box<dyn Error>> {
    let test_label_path = "data/t10k-labels-idx1-ubyte";
    let test_image_path = "data/t10k-images-idx3-ubyte";

    let image_data = load_data(test_image_path)?;
    let image_labels = load_data(test_label_path)?;

    parse_images(&image_data, &image_labels)
}

fn load_data(path: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut buf = Vec::new();
    let mut rdr = BufReader::new(std::fs::File::open(path)?);
    let _ = rdr.read_to_end(&mut buf)?;
    Ok(buf)
}

fn parse_images(image_data: &[u8], label_data: &[u8]) -> Result<Vec<Image>, Box<Error>> {
    if let Ok((_, images)) = mnist_image_parser(image_data) {
        if let Ok((_, labels)) = mnist_label_parser(label_data) {
            return Ok(images
                .into_iter()
                .zip(labels.into_iter())
                .map(|(data, label)| Image { data, label })
                .collect());
        }
    }
    panic!("Unable to read file format")
}
