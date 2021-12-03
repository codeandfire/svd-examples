use ndarray::prelude::*;
use ndarray_linalg::*;
use plotters::coord::types::RangedCoordf32;
use plotters::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct TextExample {
    docs: [&'static str; 9],
    doc_labels: [&'static str; 9],
    vocab: Vec<&'static str>,
    count_matrix: Array<f64, Ix2>,
    doc_vecs: Array<f64, Ix2>,
    term_vecs: Array<f64, Ix2>,
}

impl TextExample {
    pub fn setup(k: usize) -> TextExample {
        let docs = [
            "human machine interface for lab ABC computer applications",
            "a survey of user opinion of computer system response time",
            "the EPS user interface management system",
            "system and human system engineering testing of EPS",
            "relation of user-perceived response time to error measurement",
            "the generation of random binary unordered trees",
            "the intersection graph of paths in trees",
            "graph minors widths of trees and well quasi ordering",
            "graph minors a survey",
        ];

        let doc_labels = ["h1", "h2", "h3", "h4", "h5", "g1", "g2", "g3", "g4"];

        // prepare the vocab
        let mut vocab = HashSet::new();

        for text in &docs {
            for token in text.split_whitespace() {
                vocab.insert(token);
            }
        }

        // map tokens from words to indices in the vocab
        let mut vocab_to_idx: HashMap<&str, usize> = HashMap::new();
        vocab.iter().enumerate().for_each(|(i, v)| {
            // unwrap_or_default here is just for namesake handling
            // of this insert call, which will always return None but the
            // program is not supposed to panic because None is the right
            // value in this case.
            vocab_to_idx.insert(v, i).unwrap_or_default();
        });

        // prepare the count matrix
        let mut count_matrix: Array<usize, Ix2> = Array::zeros((vocab.len(), docs.len()));

        for (d, doc) in docs.iter().enumerate() {
            for token in doc.split_whitespace() {
                let i = *vocab_to_idx.get(token).unwrap();
                let elem = count_matrix.get_mut((i, d)).unwrap();
                *elem += 1_usize;
            }
        }

        // retain only those tokens in the vocabulary, which have occurred in
        // more than one document.
        // also remove stopwords: 'and', 'of', 'the', 'a'.
        vocab_to_idx = vocab_to_idx
            .into_iter()
            .filter(|(_, i)| count_matrix.slice(s![*i, ..]).sum() > 1)
            .filter(|(v, _)| !["and", "of", "the", "a"].contains(v))
            .collect();

        // update the vocabulary and count matrix accordingly
        let mut vocab = Vec::new();
        let mut idxs = Vec::new();
        vocab_to_idx.into_iter().for_each(|(v, i)| {
            vocab.push(v);
            idxs.push(i);
        });
        count_matrix = count_matrix.select(Axis(0), &idxs);

        // convert the count matrix from type usize to type f64
        let count_matrix = count_matrix.mapv(|v| v as f64);

        // take SVD
        let (u_matr, sing, vt_matr) =
            TruncatedSvd::new(count_matrix.clone(), TruncatedOrder::Largest)
                .decompose(k)
                .expect("Failed to take SVD.")
                .values_vectors();

        // get term and document vectors.
        // also normalize them.
        let sing: Array<f64, Ix2> = Array::from_diag(&sing);
        let normalize_vecs = |vecs: Array<f64, Ix2>| {
            &vecs
                / &vecs
                    .mapv(|v| v.powi(2))
                    .sum_axis(Axis(1))
                    .mapv(f64::sqrt)
                    .insert_axis(Axis(1))
        };
        let term_vecs = normalize_vecs(u_matr.dot(&sing));
        let doc_vecs = normalize_vecs(sing.dot(&vt_matr).reversed_axes());

        TextExample {
            docs,
            doc_labels,
            vocab,
            count_matrix,
            doc_vecs,
            term_vecs,
        }
    }

    fn display_docs(&self) {
        // maximum length of document labels
        let maxlen = self.doc_labels.iter().map(|l| l.len()).max().unwrap();

        for (label, doc) in self.doc_labels.iter().zip(self.docs) {
            println!("{:w$}: {}", label, doc, w = maxlen);
        }
    }

    fn display_count_matrix(&self) {
        // maximum length of labels
        let maxlen = self
            .doc_labels
            .iter()
            .chain(self.vocab.iter())
            .map(|l| l.len())
            .max()
            .unwrap();

        // formatting width
        let width = if maxlen > 4 { maxlen } else { 4 };

        for _ in 0..width + 1 {
            print!(" ");
        }

        for term in self.vocab.iter() {
            print!("{:>w$} ", term, w = width);
        }
        println!();

        for (d, label) in self.doc_labels.iter().enumerate() {
            print!("{:w$} ", label, w = width);

            for i in 0..self.vocab.len() {
                print!("{:w$.2} ", self.count_matrix[[i, d]], w = width);
            }

            println!();
        }
    }

    fn display_vecs(vecs: &Array<f64, Ix2>, labels: &[&str]) {
        // maximum length of the labels
        let maxlen = labels.iter().map(|l| l.len()).max().unwrap();

        // formatting width
        let width = if maxlen > 7 { maxlen } else { 7 };

        for (i, l) in labels.iter().enumerate() {
            println!("{:w$} {:+w$.4}", l, vecs.slice(s![i, ..]), w = width);
        }
    }

    fn display_sim_matrix(vecs: &Array<f64, Ix2>, labels: &[&str]) {
        // normalize and calculate the similarity matrix
        let norm_vals = vecs
            .mapv(|v| v.powi(2))
            .sum_axis(Axis(1))
            .mapv(f64::sqrt)
            .insert_axis(Axis(1));

        let norm_vecs = vecs / &norm_vals;
        let sim_matrix = norm_vecs.dot(&norm_vecs.t());

        // maximum length of the labels
        let maxlen = labels.iter().map(|l| l.len()).max().unwrap();

        // formatting width
        let width = if maxlen > 7 { maxlen } else { 7 };

        for _ in 0..width + 1 {
            print!(" ");
        }

        for l in labels.iter() {
            print!("{:>w$} ", l, w = width);
        }
        println!();

        for (i, l) in labels.iter().enumerate() {
            print!("{:w$} ", l, w = width);

            for j in 0..labels.len() {
                print!("{:w$.4} ", sim_matrix[[i, j]], w = width);
            }

            println!();
        }
    }

    pub fn display(&self, key: &str) -> Result<(), String> {
        match key {
            "docs" => self.display_docs(),
            "count_matrix" => self.display_count_matrix(),
            "term_vecs" => TextExample::display_vecs(&self.term_vecs, &self.vocab),
            "doc_vecs" => TextExample::display_vecs(&self.doc_vecs, &self.doc_labels),
            "term_sim" => TextExample::display_sim_matrix(&self.term_vecs, &self.vocab),
            "doc_sim" => TextExample::display_sim_matrix(&self.doc_vecs, &self.doc_labels),
            _ => {
                return Err(format!(
                    "Invalid key '{}'! Key must be one of 'docs', 'count_matrix', \
                    'term_vecs', 'doc_vecs', 'term_sim' and 'doc_sim'.",
                    key
                ))
            }
        };

        Ok(())
    }

    pub fn plot(&self, key: &str) -> Result<(), String> {
        // vec to store doc_labels which is an array
        let doc_labels = Vec::from(self.doc_labels);

        let (vecs, labels) = match key {
            "term_vecs" => (&self.term_vecs, &self.vocab),
            "doc_vecs" => (&self.doc_vecs, &doc_labels),
            _ => {
                return Err(format!(
                    "Invalid key '{}'! Key must be one of 'term_vecs' and 'doc_vecs'.",
                    key
                ))
            }
        };

        let dim = vecs.len_of(Axis(1));
        if dim != 2 {
            return Err(format!(
                "Vectors are of dimension {}! They need to be 2-dimensional to plot!",
                dim
            ));
        }

        let filename = format!("{}_plot.svg", key);

        // limits of x- and y-dimensions
        let find_dim_lim = |d| {
            vecs.slice(s![.., d])
                .iter()
                .map(|v| v.abs() as f32)
                .reduce(f32::max)
                .unwrap()
        };

        let (x_lim, y_lim) = (find_dim_lim(0), find_dim_lim(1));

        const BASE_SIZE: u32 = 400;

        // padding to account for any text spillovers
        const PADDING: u32 = 50;

        // size in pixels of x- and y-dimensions
        let mut x_size = BASE_SIZE;
        let mut y_size = (x_size as f64 + ((y_lim / x_lim) as f64 - 1.0) * x_size as f64) as u32;

        // coordinate system
        let coord: Cartesian2d<RangedCoordf32, RangedCoordf32> = Cartesian2d::new(
            -x_lim..x_lim,
            -y_lim..y_lim,
            (
                PADDING as i32 / 2..(x_size as i32 - PADDING as i32 / 2),
                PADDING as i32 / 2..(y_size as i32 - PADDING as i32 / 2),
            ),
        );

        x_size += PADDING;
        y_size += PADDING;

        // create figure
        let root = SVGBackend::new(&filename, (x_size, y_size))
            .into_drawing_area()
            .apply_coord_spec(coord);
        root.fill(&WHITE).expect("Failed to fill figure.");

        let dot_and_label = |x: f32, y: f32, label: &'static str| {
            EmptyElement::at((x, y))
                + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
                + Text::new(label, (5, 0), ("sans-serif", 15.0).into_font())
        };

        // plot vectors
        for (row, label) in vecs.axis_iter(Axis(0)).zip(labels) {
            let row = row.into_owned().into_raw_vec();
            let (x, y) = (row[0] as f32, row[1] as f32);
            root.draw(&dot_and_label(x, y, label))
                .expect("Failed to draw on figure.");
        }

        Ok(())
    }
}
