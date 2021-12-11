use image::{GrayImage, ImageBuffer, Luma};
use ndarray::prelude::*;
use ndarray_linalg::*;
use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::ops::Deref;
use std::path::Path;

/// Take SVD of a matrix, yielding three matrices: U, S and V^T.
trait TakeSvd {
    fn take_svd(&self, k: usize) -> (Array<f64, Ix2>, Array<f64, Ix2>, Array<f64, Ix2>);
}

/// Implementation of `TakeSvd` for a floating point 2-D array.
/// This is a simple wrapper around the `TruncatedSvd` struct provided by `ndarray-linalg`.
impl TakeSvd for Array<f64, Ix2> {
    fn take_svd(&self, k: usize) -> (Array<f64, Ix2>, Array<f64, Ix2>, Array<f64, Ix2>) {
        let (u_matr, sing, vt_matr) = TruncatedSvd::new(self.clone(), TruncatedOrder::Largest)
            .decompose(k)
            .expect("Failed to take SVD.")
            .values_vectors();

        // convert the vector of singular values into a diagonal matrix.
        let sing = Array::from_diag(&sing);

        (u_matr, sing, vt_matr)
    }
}

/// Array containing grayscale pixel values of an image.
struct GrayPixelArray(Array<u8, Ix2>);

impl Deref for GrayPixelArray {
    type Target = Array<u8, Ix2>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Conversion from a grayscale image to an array.
impl From<GrayImage> for GrayPixelArray {
    fn from(img: GrayImage) -> Self {
        let (w, h) = img.dimensions();
        let img = img.as_raw();
        let mut arr = Array::default((h as usize, w as usize));

        for (img_elem, arr_elem) in img.iter().zip(arr.iter_mut()) {
            *arr_elem = *img_elem;
        }

        GrayPixelArray(arr)
    }
}

/// Conversion from an array of grayscale pixel values to a grayscale image.
impl From<GrayPixelArray> for GrayImage {
    fn from(arr: GrayPixelArray) -> Self {
        let (h, w) = arr.dim();
        let mut img: GrayImage = ImageBuffer::new(w as u32, h as u32);

        for (img_elem, arr_elem) in img.pixels_mut().zip(arr.iter()) {
            *img_elem = Luma::from([*arr_elem]);
        }

        img
    }
}

/// Struct representing a term-document count matrix.
/// `matrix` contains the actual count matrix while `vocab` is the vocabulary
/// or the set of unique terms counted in the matrix.
struct TermDocCounts {
    pub matrix: Array<usize, Ix2>,
    pub vocab: Vec<String>
}

impl TermDocCounts {
    /// Obtain a term-document count matrix from a vector of strings (documents).
    fn from_documents(documents: Vec<String>) -> Self {
        // first keep `vocab` as a HashSet so that it only counts unique elements.
        let mut vocab = HashSet::new();
        for doc in documents.iter() {
            for token in doc.split_whitespace() {
                vocab.insert(token.to_string());
            }
        }

        // convert `vocab` from a HashSet to a Vec.
        let mut vocab = Vec::from_iter(vocab.into_iter());
        vocab.sort();

        // a mapping from terms to indices in the `vocab`.
        let mut vocab_to_idx: HashMap<&str, usize> = HashMap::new();
        vocab.iter().enumerate().for_each(|(i, v)| {
            vocab_to_idx.insert(v, i);
        });

        // obtain term-document counts and fill in the count matrix.
        let mut matrix: Array<usize, Ix2> = Array::zeros((vocab.len(), documents.len()));
        for (j, doc) in documents.iter().enumerate() {
            for i in doc.split_whitespace().map(|token| vocab_to_idx.get(token).unwrap()) {
                let elem = matrix.get_mut((*i, j)).unwrap();
                *elem += 1_usize;
            }
        }

        TermDocCounts { matrix, vocab }
    }

    /// Prune stopwords out of the count matrix and the `vocab`.
    fn prune_stopwords(&mut self, stopwords: &[String]) {
        // Indices of the non-stopword tokens in the `vocab`.
        let idxs: Vec<usize> = (0..self.vocab.len())
            .filter(|i| !stopwords.contains(&self.vocab[*i]))
            .collect();

        // Remove the stopword tokens, i.e. remove elements at indices not in `idxs` above.
        (0..self.vocab.len())
            .filter(|i| !idxs.contains(i))
            .for_each(|i| { self.vocab.remove(i); } );

        // Retain only the rows corresponding to the non-stopword tokens.
        self.matrix = self.matrix.select(Axis(0), &idxs);
    }

    /// Prune out tokens not present at least `min_count` times in all the documents
    /// taken together.
    fn prune_min_count(&mut self, min_count: usize) {
        let idxs: Vec<usize> = (0..self.vocab.len())
            .filter(|i| self.matrix.slice(s![*i, ..]).scalar_sum() >= min_count)
            .collect();

        (0..self.vocab.len())
            .filter(|i| !idxs.contains(i))
            .for_each(|i| { self.vocab.remove(i); } );

        self.matrix = self.matrix.select(Axis(0), &idxs);
    }
}

/// Print out a matrix as a neatly formatted table, with labels along the rows
/// and columns.
/// `width` and `precision` are formatting attributes.
trait PrintMatrix {
    fn print_matrix<T: Display>(
        &self,
        row_labels: Vec<T>,
        col_labels: Vec<T>,
        width: usize,
        precision: usize,
    ) -> Result<(), &str>;
}

/// Implementation of `PrintMatrix` for a floating-point 2-D array.
impl PrintMatrix for Array<f64, Ix2> {
    fn print_matrix<T: Display>(
        &self,
        row_labels: Vec<T>,
        col_labels: Vec<T>,
        width: usize,
        precision: usize,
    ) -> Result<(), &str> {
        if (row_labels.len(), col_labels.len()) != self.dim() {
            return Err("Dimensions of array do not match number of row and/or column labels.");
        }

        for _ in 0..width + 1 {
            print!(" ");
        }

        for label in col_labels.iter() {
            print!("{:>w$} ", label, w = width);
        }
        println!();

        for (i, label) in row_labels.iter().enumerate() {
            print!("{:<w$} ", label, w = width);

            for j in 0..col_labels.len() {
                print!("{:>w$.p$} ", self[[i, j]], w = width, p = precision);
            }
            println!();
        }

        Ok(())
    }
}

/// Print out a set of vectors as a neatly formatted table, with labels along the rows.
/// `width` and `precision` are formatting attributes.
trait PrintVectors {
    fn print_vectors<T: Display>(
        &self,
        labels: Vec<T>,
        width: usize,
        precision: usize,
    ) -> Result<(), &str>;
}

/// Implementation of `PrintVectors` for a floating-point 2-D array.
impl PrintVectors for Array<f64, Ix2> {
    fn print_vectors<T: Display>(
        &self,
        labels: Vec<T>,
        width: usize,
        precision: usize,
    ) -> Result<(), &str> {
        if self.len_of(Axis(0)) != labels.len() {
            return Err("Number of rows in array do not match number of labels.");
        }

        for _ in 0..width + 1 {
            print!(" ");
        }

        for (label, vec) in labels.iter().zip(self.genrows()) {
            println!("{:<w$} {:>w$.p$}", label, vec, w = width, p = precision);
        }

        Ok(())
    }
}

/// Plot a set of vectors on a graph.
/// Labels for the vectors may be optionally provided.
trait PlotVectors {
    fn plot_vectors<T: Display, P: AsRef<Path>>(
        &self,
        labels: Vec<Option<T>>,
        save_file: &P
    ) -> Result<(), &str>;
}

/// Implementation of `PlotVectors` for a 32-bit floating-point 2-D array.
impl PlotVectors for Array<f32, Ix2> {
    fn plot_vectors<T: Display, P: AsRef<Path>>(
        &self,
        labels: Vec<Option<T>>,
        save_file: &P
    ) -> Result<(), &str> {
        if self.len_of(Axis(0)) != labels.len() {
            return Err("Number of rows in array do not match number of labels.");
        }

        const FIGSIZE: u32 = 500;
        const PADDING: u32 = 50;   // to accommodate for any text overflow from the labels.

        // find the maximum magnitude of all elements in the vectors and use that
        // to set up a square coordinate system.
        let max_value = self.iter().map(|v| v.abs()).reduce(f32::max).unwrap();
        let coord: Cartesian2d<RangedCoordf32, RangedCoordf32> = Cartesian2d::new(
            -max_value..max_value,
            -max_value..max_value,
            (
                PADDING as i32..(FIGSIZE - PADDING) as i32,
                PADDING as i32..(FIGSIZE - PADDING) as i32,
            )
        );

        let root = SVGBackend::new(save_file, (FIGSIZE, FIGSIZE))
            .into_drawing_area()
            .apply_coord_spec(coord);
        root.fill(&WHITE).expect("Failed to fill figure.");

        let plot_dot = |x: f32, y: f32| {
            EmptyElement::at((x, y))
                + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
        };

        let plot_dot_and_label = |x: f32, y: f32, label: T| {
            plot_dot(x, y)
                + Text::new(format!("{}", label), (10, 0), ("sans-serif", 15.0).into_font())
        };

        for (vector, label) in self
            .genrows()
            .into_iter()
            .map(|row| row.to_vec())
            .zip(labels.into_iter())
        {
            let (x, y) = (vector[0], vector[1]);
            match label {
                Some(l) => root.draw(&plot_dot_and_label(x, y, l)),
                None => root.draw(&plot_dot(x, y)),
            }
            .expect("Failed to draw on figure.");
        }

        Ok(())
    }
}
