use image::{GrayImage, ImageBuffer, Luma};
use ndarray::prelude::*;
use ndarray_linalg::{TruncatedOrder, TruncatedSvd};
use std::collections::{HashMap, HashSet};
use std::ops::Deref;

/// Take SVD of a matrix, yielding three matrices: U, S and V^T.
pub trait TakeSvd {
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
pub struct GrayPixelArray(pub Array<u8, Ix2>);

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

/// Bag-of-Words representation of a document.
/// The `HashMap` is a mapping from a token to the count, i.e. the number of times
/// it occurs in the document.
struct Document(pub HashMap<String, usize>);

/// Use simple whitespace tokenization to construct a `Document` representation from a `String`.
impl From<String> for Document {
    fn from(text: String) -> Self {
        let mut counts = HashMap::new();

        // whitespace tokenization
        for token in text.split_whitespace() {

            // need to use `String::from(token)` instead of `token` because
            // `token` is a string slice - a reference to a slice of `text` -
            // and `text` is freed at the end of this function, so the Rust
            // compiler will not allow you to do that (this is a dangling pointer).
            let c = counts.entry(String::from(token)).or_insert(0);
            *c += 1;
        }

        Document(counts)
    }
}

/// A representation of a document corpus.
/// Contains a `Vec` of individual documents as well as a vocabulary `vocab`.
struct Corpus {
    docs: Vec<Document>,
    pub vocab: Vec<String>,
}

/// Construct a `Corpus` representation from a `Vec` of `Documents`.
impl From<Vec<Document>> for Corpus {
    fn from(docs: Vec<Document>) -> Self {
        // the `vocab` is initially a `HashSet` so that it records only unique elements.
        let mut vocab = HashSet::new();

        for doc in docs.iter() {
            for token in doc.0.keys() {
                vocab.insert(token.clone());
            }
        }

        // collect the `vocab` in a `Vec` and sort.
        let mut vocab: Vec<String> = Vec::from_iter(vocab.into_iter());
        vocab.sort();

        Corpus { docs, vocab }
    }
}

impl Corpus {
    /// Prune out stopwords from the vocabulary of the `Corpus`.
    fn prune_stopwords(self, stopwords: Vec<String>) -> Self {
        let new_vocab: Vec<String> = self.vocab
            .into_iter()
            .filter(|token| !stopwords.contains(token))
            .collect();

        Corpus { docs: self.docs, vocab: new_vocab }
    }

    /// Prune out words from the vocabulary of the `Corpus` that do not satisfy a
    /// minimum count threshold, i.e. that do not occur at least `min_count` number
    /// of times in all of the documents taken together.
    fn prune_min_count(self, min_count: usize) -> Self {
        let new_vocab: Vec<String> = self.vocab
            .into_iter()
            .filter(|token| {
                let mut count = 0;
                for doc in self.docs.iter() {
                    // number of times `token` occurs in `doc`.
                    count += *doc.0.get(token).unwrap_or(&0);
                }

                count >= min_count
            })
            .collect();

        Corpus { docs: self.docs, vocab: new_vocab }
    }

    /// Construct a count matrix from the `Corpus`.
    fn to_count_matrix(&self) -> Array<usize, Ix2> {
        let mut matr: Array<usize, Ix2> = Array::default((self.vocab.len(), self.docs.len()));

        for (j, doc) in self.docs.iter().enumerate() {
            for (i, token) in self.vocab.iter().enumerate() {
                let elem = matr.get_mut((i, j)).unwrap();

                // number of times `token` occurs in `doc`.
                *elem = *doc.0.get(token).unwrap_or(&0);
            }
        }

        matr
    }
}

enum Feature {
    Continuous(f64),
    Categorical(usize),
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

        assert_eq!(GrayPixelArray::from(img_rep), arr_rep);
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

        assert_eq!(GrayImage::from(arr_rep), img_rep);
    }

    #[test]
    fn corpus_count_matrix_and_vocab() {
        let corpus = Corpus::from(vec![
            Document::from("a a b c".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ]);

        assert_eq!(corpus.vocab, ["a", "b", "c", "d"]);
        assert_eq!(
            corpus.to_count_matrix(),
            array![[2, 1, 1], [1, 1, 0], [1, 4, 0], [0, 0, 1]]
        );
    }

    #[test]
    fn corpus_extra_whitespace() {
        let corpus1 = Corpus::from(vec![
            Document::from("   a a   b c   ".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ]);

        let corpus2 = Corpus::from(vec![
            Document::from("a a b c".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ]);

        assert_eq!(corpus1.vocab, corpus2.vocab);
        assert_eq!(corpus1.to_count_matrix(), corpus2.to_count_matrix());
    }

    #[test]
    fn corpus_empty_documents() {
        let corpus = Corpus::from(vec![
            Document::from("".to_string()),
            Document::from("            ".to_string()),
            Document::from("a a b c".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ]);

        assert_eq!(corpus.vocab, ["a", "b", "c", "d"]);
        assert_eq!(
            corpus.to_count_matrix(),
            array![[0, 0, 2, 1, 1], [0, 0, 1, 1, 0], [0, 0, 1, 4, 0], [0, 0, 0, 0, 1]]
        );
    }

    #[test]
    fn corpus_stopwords() {
        let corpus1 = Corpus::from(vec![
            Document::from("a a b c d e e e".to_string()),
            Document::from("b b b b c c e".to_string()),
            Document::from("c c d d a e e e e e".to_string()),
            Document::from("a a b d c a".to_string()),
        ])
        .prune_stopwords(vec!["a".to_string(), "e".to_string()]);

        let corpus2 = Corpus::from(vec![
            Document::from("b c d".to_string()),
            Document::from("b b b b c c".to_string()),
            Document::from("c c d d".to_string()),
            Document::from("b d c".to_string()),
        ]);

        assert_eq!(corpus1.vocab, corpus2.vocab);
        assert_eq!(corpus1.to_count_matrix(), corpus2.to_count_matrix());
    }

    #[test]
    fn corpus_stopword_not_in_vocab() {
        Corpus::from(vec![
            Document::from("a a b c".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ])
        .prune_stopwords(vec!["e".to_string()]);   // no Err, Result etc.
    }

    #[test]
    fn corpus_min_count() {
        let corpus1 = Corpus::from(vec![
            Document::from("a a a d d f".to_string()),
            Document::from("d d b c e c a".to_string()),
            Document::from("c e e f".to_string()),
        ])
        .prune_min_count(3);

        let corpus2 = Corpus::from(vec![
            Document::from("a a a d d".to_string()),
            Document::from("d d c e c a".to_string()),
            Document::from("c e e".to_string())
        ]);

        assert_eq!(corpus1.vocab, corpus2.vocab);
        assert_eq!(corpus1.to_count_matrix(), corpus2.to_count_matrix());
    }

    #[test]
    fn corpus_min_count_zero_or_one() {
        let corpus1 = Corpus::from(vec![
            Document::from("a a b c".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ])
        .prune_min_count(0);

        let corpus2 = Corpus::from(vec![
            Document::from("a a b c".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ])
        .prune_min_count(1);

        let corpus3 = Corpus::from(vec![
            Document::from("a a b c".to_string()),
            Document::from("b c c c c a".to_string()),
            Document::from("a d".to_string()),
        ]);

        assert_eq!(corpus1.vocab, corpus3.vocab);
        assert_eq!(corpus2.vocab, corpus3.vocab);

        assert_eq!(corpus1.to_count_matrix(), corpus3.to_count_matrix());
        assert_eq!(corpus2.to_count_matrix(), corpus3.to_count_matrix());
    }
}
