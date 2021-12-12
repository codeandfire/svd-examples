use image::{GrayImage, ImageBuffer, Luma};
use ndarray::prelude::*;
use ndarray_linalg::{TruncatedOrder, TruncatedSvd};
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
}
