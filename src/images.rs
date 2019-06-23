use crate::nn::{DynMatrix, DynMatrixSlice};
use nalgebra::{ArrayStorage, Matrix, U28};
use std::convert::TryFrom;
use std::fmt::{Display, Error, Formatter};
use std::io::Write;

impl Display for Image {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        fn to_char(val: f64) -> char {
            match val {
                _ if (0.0..0.3).contains(&val) => ' ',
                _ if (0.1..0.3).contains(&val) => '.',
                _ if (0.3..0.5).contains(&val) => '*',
                _ if (0.5..0.7).contains(&val) => '&',
                _ => '#',
            }
        }

        let (mut x, mut y) = (0, 0);

        let height = 28;
        let width = 28;

        let mut buf: Vec<Vec<char>> = vec![vec![' '; width]; height];

        for (i, val) in self.data.iter().skip(1).enumerate() {
            if i != 0 && i % width == 0 {
                y += 1;
                x = 0;
            }
            buf[y][x] = to_char(*val);
            x += 1;
        }
        for row in buf.iter() {
            let s = row.iter().collect::<String>();
            writeln!(f, "{}", s)?;
        }

        Ok(())
    }
}

#[derive(PartialEq)]
pub struct Image {
    pub data: DynMatrix,
    pub label: DynMatrix,
}

impl Image {
    pub fn from_parts(data: DynMatrix, label: DynMatrix) -> Self {
        Image { data, label }
    }
}
