use image::io::Reader as ImageReader;
use image::GrayImage;
use svd_examples::*;

fn image_example() {
    let img = ImageReader::open("face.png").unwrap().decode().unwrap();
    let img = img.into_luma8();
    let arr = GrayPixelArray::from(img);
    let arr = arr.mapv(|v| v as f64);

    for k in (5..=50).step_by(5) {
        let (u, s, vt) = arr.take_svd(k);
        let trunc_arr = u.dot(&s).dot(&vt);
        let trunc_arr = GrayPixelArray(trunc_arr.mapv(|v| v as u8));
        let trunc_img = GrayImage::from(trunc_arr);
        trunc_img.save(format!("k_{}.png", k)).expect("Failed to save image.");
    }
}

fn main() {
    image_example();
}
