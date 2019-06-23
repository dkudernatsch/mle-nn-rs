use nom::{be_u32, be_u8, count, do_parse, map, map_res, named, tag, take};
use std::convert::TryFrom;

use crate::nn::DynMatrix;

named!(mnist_label_magic_num, tag!([0x00, 0x00, 0x08, 0x01]));

named!(
    pub mnist_label_parser<Vec<DynMatrix>>,
    do_parse!(
        mnist_label_magic_num
            >> length: be_u32
            >> labels:
                count!(
                    map!(be_u8, |label| DynMatrix::from_fn(10, 1, |r, _| {
                        if r == label as usize {
                            1.0
                        } else {
                            0.0
                        }
                    })),
                    length as usize
                )
            >> (labels)
    )
);

named!(mnist_image_magic_num, tag!([0x00, 0x00, 0x08, 0x03]));

named!(
    pub mnist_image_parser<Vec<DynMatrix>>,
    do_parse!(
        mnist_image_magic_num
            >> length: be_u32
            >> rows: be_u32
            >> cols: be_u32
            >> images: count!(map!(take!(rows * cols), to_image_space), length as usize)
            >> (images)
    )
);

#[allow(dead_code)]
fn to_image_space(data: &[u8]) -> DynMatrix {
    let tmp = std::iter::once(1.0)
        .chain(data.iter().map(|d| *d as f64 / 255.0))
        .collect::<Vec<_>>()
        .len();

    DynMatrix::from_iterator(
        785,
        1,
        std::iter::once(1.0).chain(data.iter().map(|d| *d as f64 / 255.0)),
    )
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::images::Image;
    use std::error::Error;
    use std::io::{BufReader, Read};

    #[test]
    fn integration_test_label_parser() {
        let data = {
            let mut buff = Vec::new();
            let mut rdr =
                BufReader::new(std::fs::File::open("data/t10k-labels-idx1-ubyte").unwrap());
            rdr.read_to_end(&mut buff);
            buff
        };

        if let Ok((input, labels)) = mnist_label_parser(&data) {
            assert_eq!(labels.len(), 10_000);
        } else {
            assert!(false);
        }
    }

}
