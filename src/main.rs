use clap::Parser;
use svd_examples::LSAExample;

#[derive(Parser)]
#[clap(about = "Examples of the SVD.")]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    LSA(LSAOpts),
}

#[derive(Parser)]
#[clap(about = "Latent Semantic Analysis (LSA) example from Deerwester et. al. (1990)")]
struct LSAOpts {
    #[clap(
        short = 'k',
        long,
        about = "Number of singular values",
        default_value = "2"
    )]
    dim_k: usize,

    #[clap(long, about = "Display the nine documents")]
    disp_docs: bool,

    #[clap(long, about = "Display the count matrix")]
    disp_count: bool,

    #[clap(long, about = "Display document similarity matrix")]
    disp_doc_sim: bool,

    #[clap(long, about = "Display term similarity matrix")]
    disp_term_sim: bool,
}

fn main() {
    let opts = Opts::parse();

    match opts.subcmd {
        SubCommand::LSA(lsa_opts) => {
            let lsa_example = LSAExample::setup(lsa_opts.dim_k);

            if lsa_opts.disp_docs {
                println!("Nine documents:");
                lsa_example.display("docs").unwrap();
                println!();
            }

            if lsa_opts.disp_count {
                println!("Count matrix:");
                lsa_example.display("count_matrix").unwrap();
                println!();
            }

            if lsa_opts.disp_doc_sim {
                println!("Document similarity matrix:");
                lsa_example.display("doc_sim").unwrap();
                println!();
            }

            if lsa_opts.disp_term_sim {
                println!("Term similarity matrix:");
                lsa_example.display("term_sim").unwrap();
                println!();
            }

            println!("Document vectors:");
            lsa_example.display("doc_vecs").unwrap();
            lsa_example.plot("doc_vecs").unwrap_or_else(|error| {
                println!("Skipping plot: {}", error);
            });
            println!();

            println!("Term vectors:");
            lsa_example.display("term_vecs").unwrap();
            lsa_example.plot("term_vecs").unwrap_or_else(|error| {
                println!("Skipping plot: {}", error);
            });
            println!();
        }
    }
}
