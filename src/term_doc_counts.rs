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

#[cfg(test)]
mod tests {
    use super::*;

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
