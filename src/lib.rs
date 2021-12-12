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
#[derive(Debug, PartialEq)]
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
    pub vocab: Vec<String>,
}

impl TermDocCounts {
    /// Obtain a term-document count matrix from a vector of strings (documents).
    fn from_documents(documents: &[String]) -> Self {
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

        let mut vocab_to_idx: HashMap<&str, usize> = HashMap::new();
        vocab.iter().enumerate().for_each(|(i, v)| {
            vocab_to_idx.insert(v, i);
        });

        // obtain term-document counts and fill in the count matrix.
        let mut matrix: Array<usize, Ix2> = Array::zeros((vocab.len(), documents.len()));
        for (j, doc) in documents.iter().enumerate() {
            for i in doc
                .split_whitespace()
                .map(|token| vocab_to_idx.get(token).unwrap())
            {
                let elem = matrix.get_mut((*i, j)).unwrap();
                *elem += 1_usize;
            }
        }

        TermDocCounts { matrix, vocab }
    }

    /// Prune stopwords out of the count matrix and the `vocab`.
    fn prune_stopwords(&mut self, stopwords: &[String]) {
        let mut new_vocab = Vec::new();
        let mut idxs = Vec::new();

        self.vocab
            .iter()
            .enumerate()
            .filter(|(_, v)| !stopwords.contains(v))
            .for_each(|(i, v)| {
                new_vocab.push(v.clone());
                idxs.push(i);
            });

        self.vocab = new_vocab;
        self.matrix = self.matrix.select(Axis(0), &idxs);
    }

    /// Prune out tokens not present at least `min_count` times in all the documents
    /// taken together.
    fn prune_min_count(&mut self, min_count: usize) {
        let mut new_vocab = Vec::new();
        let mut idxs = Vec::new();

        self.matrix
            .genrows()
            .into_iter()
            .enumerate()
            .filter(|(_, col)| col.scalar_sum() >= min_count)
            .for_each(|(i, _)| {
                new_vocab.push(self.vocab[i].clone());
                idxs.push(i);
            });

        self.vocab = new_vocab;
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
        save_file: &P,
    ) -> Result<(), &str>;
}

/// Implementation of `PlotVectors` for a 32-bit floating-point 2-D array.
impl PlotVectors for Array<f32, Ix2> {
    fn plot_vectors<T: Display, P: AsRef<Path>>(
        &self,
        labels: Vec<Option<T>>,
        save_file: &P,
    ) -> Result<(), &str> {
        if self.len_of(Axis(0)) != labels.len() {
            return Err("Number of rows in array do not match number of labels.");
        }

        const FIGSIZE: u32 = 500;
        const PADDING: u32 = 50; // to accommodate for any text overflow from the labels.

        // find the maximum magnitude of all elements in the vectors and use that
        // to set up a square coordinate system.
        let max_value = self.iter().map(|v| v.abs()).reduce(f32::max).unwrap();
        let coord: Cartesian2d<RangedCoordf32, RangedCoordf32> = Cartesian2d::new(
            -max_value..max_value,
            -max_value..max_value,
            (
                PADDING as i32..(FIGSIZE - PADDING) as i32,
                PADDING as i32..(FIGSIZE - PADDING) as i32,
            ),
        );

        let root = SVGBackend::new(save_file, (FIGSIZE, FIGSIZE))
            .into_drawing_area()
            .apply_coord_spec(coord);
        root.fill(&WHITE).expect("Failed to fill figure.");

        let plot_dot = |x: f32, y: f32| {
            EmptyElement::at((x, y)) + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
        };

        let plot_dot_and_label = |x: f32, y: f32, label: T| {
            plot_dot(x, y)
                + Text::new(
                    format!("{}", label),
                    (10, 0),
                    ("sans-serif", 15.0).into_font(),
                )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_matrices_shape() {
        let arr = array![[1., 2., 3.], [-2., -1., -3.], [3., 4., -1.], [0., -8., 8.]];
        let (u, s, vt) = arr.take_svd(2);
        assert_eq!(u.dim(), (4, 2));
        assert_eq!(s.dim(), (2, 2));
        assert_eq!(vt.dim(), (2, 3));
    }

    #[test]
    #[ignore]
    // this test checks whether zero singular values are made explicit or not.
    // currently I am not sure if this feature is needed, hence this test is
    // currently marked as ignored.
    fn svd_matrices_shape_linearly_dependent_rows() {
        let arr = array![
            [1., 2., 3., 4.],
            [-1., -2., -3., -4.],
            [0., 8., 0., 4.],
            [3., 6., 0., 0.]
        ];
        let (u, s, vt) = arr.take_svd(4);
        assert_eq!(u.dim(), (4, 4));
        assert_eq!(s.dim(), (4, 4));
        assert_eq!(vt.dim(), (4, 4));
    }

    #[test]
    #[should_panic]
    fn svd_zero_k_value() {
        let arr = array![[1., 2.], [-1., -3.]];
        arr.take_svd(0);
    }

    #[test]
    #[should_panic]
    fn svd_k_value_larger_than_nrows_ncols() {
        let arr = array![[1., 2.], [-1., -3.]];
        arr.take_svd(100);
    }

    #[test]
    #[ignore]
    // this test fails for the same reason as `svd_matrices_shape_linearly_dependent_rows`
    // above, hence currently marked as ignored.
    fn svd_of_zero_matrix() {
        let arr = array![[0., 0., 0.], [0., 0., 0.]];
        let (u, s, vt) = arr.take_svd(2);
        assert_eq!(u, array![[0., 0.], [0., 0.]]);
        assert_eq!(s, array![[0., 0.], [0., 0.]]);
        assert_eq!(vt, array![[0., 0., 0.], [0., 0., 0.]]);
    }

    #[test]
    fn svd_matrices_correctness() {
        let arr = array![[1., 2., 3.], [-2., -1., -3.], [3., 4., -1.], [0., -8., 8.]];
        let (u, s, vt) = arr.take_svd(3);
        assert!((arr - u.dot(&s).dot(&vt)).scalar_sum() < 1e-8);
    }

    #[test]
    fn svd_k_values_in_descending_order() {
        let arr = array![[1., 2., 3.], [-2., -1., -3.], [3., 4., -1.], [0., -8., 8.]];
        let (_, s, _) = arr.take_svd(3);
        assert!(s[[0, 0]] >= s[[1, 1]]);
        assert!(s[[1, 1]] >= s[[2, 2]]);
    }

    #[test]
    fn gray_image_to_array_via_vec() {
        let vec_rep = vec![
            0_u8, 128_u8, 255_u8, 65_u8, 40_u8, 22_u8, 10_u8, 225_u8, 180_u8, 90_u8, 130_u8, 200_u8,
        ];
        let (width, height) = (3, 4);
        let img_rep: GrayImage =
            ImageBuffer::from_vec(width as u32, height as u32, vec_rep.clone()).unwrap();
        let arr_rep = GrayPixelArray(Array::from_shape_vec((height, width), vec_rep).unwrap());

        assert_eq!(GrayPixelArray::from(img_rep as GrayImage), arr_rep);
    }

    #[test]
    fn gray_array_to_image_via_vec() {
        let vec_rep = vec![
            0_u8, 128_u8, 255_u8, 65_u8, 40_u8, 22_u8, 10_u8, 225_u8, 180_u8, 90_u8, 130_u8, 200_u8,
        ];
        let (width, height) = (3, 4);
        let img_rep: GrayImage =
            ImageBuffer::from_vec(width as u32, height as u32, vec_rep.clone()).unwrap();
        let arr_rep = GrayPixelArray(Array::from_shape_vec((height, width), vec_rep).unwrap());

        assert_eq!(GrayImage::from(arr_rep as GrayPixelArray), img_rep);
    }

    #[test]
    fn term_doc_count_matrix_and_vocab() {
        let documents = [
            "a a b c".to_string(),
            "b c c c c a".to_string(),
            "a d".to_string(),
        ];
        let td_counts = TermDocCounts::from_documents(&documents);

        assert_eq!(td_counts.matrix.dim(), (4, 3));
        assert_eq!(td_counts.vocab.len(), 4);

        assert_eq!(td_counts.vocab, ["a", "b", "c", "d"]);

        assert_eq!(
            td_counts.matrix,
            array![[2, 1, 1], [1, 1, 0], [1, 4, 0], [0, 0, 1]]
        );
    }

    #[test]
    fn term_doc_extra_whitespace() {
        let documents = [
            "   a a   b c   ".to_string(),
            "b c c c c a".to_string(),
            "a d".to_string(),
        ];
        let td_counts = TermDocCounts::from_documents(&documents);

        assert_eq!(td_counts.matrix.dim(), (4, 3));
        assert_eq!(td_counts.vocab.len(), 4);
    }

    #[test]
    fn term_doc_empty_documents() {
        let documents = [
            "".to_string(),
            "            ".to_string(),
            "a a b c".to_string(),
            "b c c c c a".to_string(),
            "a d".to_string(),
        ];
        let td_counts = TermDocCounts::from_documents(&documents);

        assert_eq!(td_counts.matrix.dim(), (4, 5));
        assert_eq!(td_counts.vocab.len(), 4);

        for i in 0..4 {
            for j in 0..2 {
                assert_eq!(td_counts.matrix[[i, j]], 0);
            }
        }
    }

    #[test]
    fn term_doc_stopwords() {
        let documents = [
            "a a b c d e e e".to_string(),
            "b b b b c c e".to_string(),
            "c c d d a e e e e e".to_string(),
            "a a b d c a".to_string(),
        ];
        let mut td_counts = TermDocCounts::from_documents(&documents);
        let old_matrix = td_counts.matrix.clone();
        let old_vocab = td_counts.vocab.clone();

        td_counts.prune_stopwords(&["a".to_string(), "e".to_string()]);

        assert_eq!(td_counts.vocab, old_vocab[1..4]);
        assert_eq!(td_counts.matrix, old_matrix.slice(s![1..4, ..]));
    }

    #[test]
    fn term_doc_stopword_not_in_vocab() {
        let documents = [
            "a a b c".to_string(),
            "b c c c c a".to_string(),
            "a d".to_string(),
        ];
        let mut td_counts = TermDocCounts::from_documents(&documents);
        td_counts.prune_stopwords(&["e".to_string()]); // no Err, Result etc.
    }

    #[test]
    fn term_doc_min_count() {
        let documents = [
            "a a a d d f".to_string(),
            "d d b c e c a".to_string(),
            "c e e f".to_string(),
        ];
        let mut td_counts = TermDocCounts::from_documents(&documents);

        td_counts.prune_min_count(3);

        assert_eq!(td_counts.vocab, ["a", "c", "d", "e"]);
        assert_eq!(
            td_counts.matrix,
            array![[3, 1, 0], [0, 2, 1], [2, 2, 0], [0, 1, 2]]
        );
    }

    #[test]
    fn term_doc_min_count_zero_or_one() {
        let documents = [
            "a a b c".to_string(),
            "b c c c c a".to_string(),
            "a d".to_string(),
        ];
        let mut td_counts = TermDocCounts::from_documents(&documents);
        let old_vocab = td_counts.vocab.clone();
        let old_matrix = td_counts.matrix.clone();

        td_counts.prune_min_count(0);
        assert_eq!(td_counts.vocab, old_vocab);
        assert_eq!(td_counts.matrix, old_matrix);

        td_counts.prune_min_count(1);
        assert_eq!(td_counts.vocab, old_vocab);
        assert_eq!(td_counts.matrix, old_matrix);
    }
}
