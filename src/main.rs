use image::io::Reader as ImageReader;
use image::GrayImage;
use std::fs;
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

fn text_example() {
    let documents: Vec<String> = fs::read_to_string("deerwester.txt")
        .expect("Failed to read file.")
        .trim()
        .split("\n")
        .map(|s| s.into())
        .collect();
    let mut td_counts = TermDocCounts::from_documents(&documents);
    td_counts.prune_stopwords(&["a".into(), "and".into(), "of".into(), "the".into()]);
    td_counts.prune_min_count(2);

    let doc_labels: Vec<String> = (1..=documents.len()).map(|i| format!("#{}", i.to_string())).collect();
    let count_matrix = td_counts.matrix.mapv(|v| v as f64);

    count_matrix
        .t()
        .into_owned()
        .print_matrix(&doc_labels, &td_counts.vocab, 9, 0)
        .unwrap();

    let (u, s, vt) = count_matrix.take_svd(2);
    let term_vecs = u.dot(&s);
    let doc_vecs = (s.dot(&vt)).t().into_owned();

    term_vecs.print_vectors(&td_counts.vocab, 9, 4).unwrap();
    doc_vecs.print_vectors(&doc_labels, 7, 4).unwrap();

    term_vecs
        .mapv(|v| v as f32)
        .plot_vectors(&td_counts.vocab, &"term_vecs.svg").unwrap();
    doc_vecs
        .mapv(|v| v as f32)
        .plot_vectors(&doc_labels, &"doc_vecs.svg").unwrap();
}

fn main() {
    //image_example();
    text_example();
}
