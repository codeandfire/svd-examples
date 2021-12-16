use ndarray::prelude::*;
use std::collections::{HashMap, HashSet};

/// Struct representing a term-document count matrix.
/// `matrix` contains the actual count matrix while `vocab` is the vocabulary
/// or the set of unique terms counted in the matrix.
pub struct TermDocCounts {
    pub matrix: Array<usize, Ix2>,
    pub vocab: Vec<String>,
}

impl TermDocCounts {
    /// Obtain a term-document count matrix from a vector of strings (documents).
    pub fn from_documents(documents: &[String]) -> Self {
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
    pub fn prune_stopwords(&mut self, stopwords: &[String]) {
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
    pub fn prune_min_count(&mut self, min_count: usize) {
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
