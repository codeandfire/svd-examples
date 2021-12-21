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
    let corpus: Vec<Document> = fs::read_to_string("deerwester.txt")
        .expect("Failed to read file.")
        .trim()
        .split('\n')
        .map(|s| Document::from(s.to_string()))
        .collect();

    let doc_labels: Vec<String> = (1..=corpus.len())
        .map(|i| format!("#{}", i.to_string()))
        .collect();

    let stopwords: Vec<String> = fs::read_to_string("stopwords.txt")
        .expect("Failed to read file.")
        .trim()
        .split('\n')
        .map(|s| s.into())
        .collect();

    let corpus = Corpus::from(corpus).prune_stopwords(stopwords).prune_min_count(2);
    let count_matr = corpus.to_count_matrix().mapv(|v| v as f64);

    print_matrix(
        &count_matr.t().into_owned(),
        &doc_labels,
        &corpus.vocab,
        9,
        0,
    )
    .unwrap();

    let (u, s, vt) = count_matr.take_svd(dim_k);
    let term_vecs = u.dot(&s);
    let doc_vecs = (s.dot(&vt)).t().into_owned();

    print_vectors(&term_vecs, &corpus.vocab, 9, 4).unwrap();
    print_vectors(&doc_vecs, &doc_labels, 7, 4).unwrap();

    if dim_k == 2 {
        plot_vectors(
            &term_vecs.mapv(|v| v as f32),
            &corpus.vocab,
            "term_vecs.svg",
        )
        .unwrap();

        plot_vectors(&doc_vecs.mapv(|v| v as f32), &doc_labels, "doc_vecs.svg").unwrap();
    } else {
        println!("Skipping plots because vectors are not 2-dimensional.");
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
