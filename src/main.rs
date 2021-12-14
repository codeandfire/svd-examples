use clap::{Parser, Subcommand};
use image::io::Reader as ImageReader;
use image::GrayImage;
use std::fs;
use svd_examples::*;

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// SVD applied to a sample image (adapted from https://github.com/Ramaseshanr/anlp/blob/master/SVDImage.ipynb)
    Image {
        /// start from `min_k` singular values
        #[clap(long, default_value = "5")]
        min_k: usize,

        /// go up to (inclusive) `max_k` singular values
        #[clap(long, default_value = "50")]
        max_k: usize,

        /// take singular values in steps of `step_k` between `min_k` and `max_k`
        #[clap(long, default_value = "5")]
        step_k: usize,
    },

    /// Latent Semantic Analysis (LSA) example from Deerwester et. al. (1990)
    Text {
        /// take `k` singular values; dimensionality of vectors will be `k`
        #[clap(long, default_value = "2")]
        dim_k: usize,
    },
}

fn image_example(min_k: usize, max_k: usize, step_k: usize) {
    let img = ImageReader::open("face.png").unwrap().decode().unwrap();
    let img = img.into_luma8();
    let arr = GrayPixelArray::from(img);
    let arr = arr.mapv(|v| v as f64);

    for k in (min_k..=max_k).step_by(step_k) {
        let (u, s, vt) = arr.take_svd(k);
        let trunc_arr = u.dot(&s).dot(&vt);
        let trunc_arr = GrayPixelArray(trunc_arr.mapv(|v| v as u8));
        let trunc_img = GrayImage::from(trunc_arr);
        let filename = format!("k_{}.png", k);
        trunc_img.save(&filename).expect("Failed to save image.");
        println!("Image saved to {}", filename);
    }
}

fn text_example(dim_k: usize) {
    let documents: Vec<String> = fs::read_to_string("deerwester.txt")
        .expect("Failed to read file.")
        .trim()
        .split('\n')
        .map(|s| s.into())
        .collect();
    let mut td_counts = TermDocCounts::from_documents(&documents);
    td_counts.prune_stopwords(&["a".into(), "and".into(), "of".into(), "the".into()]);
    td_counts.prune_min_count(2);

    let doc_labels: Vec<String> = (1..=documents.len())
        .map(|i| format!("#{}", i.to_string()))
        .collect();
    let count_matrix = td_counts.matrix.mapv(|v| v as f64);

    print_matrix(
        &count_matrix.t().into_owned(),
        &doc_labels,
        &td_counts.vocab,
        9,
        0,
    )
    .unwrap();

    let (u, s, vt) = count_matrix.take_svd(dim_k);
    let term_vecs = u.dot(&s);
    let doc_vecs = (s.dot(&vt)).t().into_owned();

    print_vectors(&term_vecs, &td_counts.vocab, 9, 4).unwrap();
    print_vectors(&doc_vecs, &doc_labels, 7, 4).unwrap();

    if dim_k == 2 {
        plot_vectors(
            &term_vecs.mapv(|v| v as f32),
            &td_counts.vocab,
            "term_vecs.svg",
        )
        .unwrap();

        plot_vectors(&doc_vecs.mapv(|v| v as f32), &doc_labels, "doc_vecs.svg").unwrap();
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Image {
            min_k,
            max_k,
            step_k,
        } => image_example(min_k, max_k, step_k),
        Command::Text { dim_k } => text_example(dim_k),
    }
}
