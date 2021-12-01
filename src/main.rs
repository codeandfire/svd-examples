use svd_examples::TextExample;

fn main() {
    let text_example = TextExample::setup(2);
    text_example.display("docs");
    println!();
    text_example.display("count_matrix");
    println!();
    text_example.display("doc_vecs");
    println!();
    text_example.display("term_vecs");

    text_example.plot("doc_vecs");
    text_example.plot("term_vecs");
}
