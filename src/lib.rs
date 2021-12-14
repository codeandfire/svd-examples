mod impl_core;
pub use crate::impl_core::{GrayPixelArray, TakeSvd};

mod term_doc_counts;
pub use crate::term_doc_counts::TermDocCounts;

mod print_plot;
pub use crate::print_plot::{plot_vectors, print_matrix, print_vectors};
