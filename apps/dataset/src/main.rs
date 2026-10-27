use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context;
use clap::Parser;
use dataset::{
    DatasetOptions, generate, write_adjacency_list, write_coordinate_documentation, write_jsonl,
    write_metadata,
};
use spatial::Space;

/// Generate a dataset of spatial intelligence questions and answers
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to write the generated questions JSONL file
    #[arg(
        short,
        long,
        default_value = "data/questions.jsonl",
        value_name = "FILE"
    )]
    output: PathBuf,

    /// Path to write the graph adjacency list (defaults to graph.json next to the questions file)
    #[arg(long, value_name = "FILE")]
    graph_output: Option<PathBuf>,

    /// Path to write the coordinate system documentation (defaults to coordinates.md next to the questions file)
    #[arg(long, value_name = "FILE")]
    coordinates_doc: Option<PathBuf>,

    /// Path to write space metadata (defaults to metadata.json next to the questions file)
    #[arg(long, value_name = "FILE")]
    metadata_output: Option<PathBuf>,

    /// Number of hexes along one side of the space
    #[arg(long, default_value_t = 3)]
    size: u32,

    /// Number of vertex pairs to sample
    #[arg(long, default_value_t = 100)]
    count: usize,

    /// Seed for sampling vertex pairs
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Generate questions using coordinate-based vertex representation
    #[arg(long)]
    coordinate: bool,

    /// Generate questions using graph-based node representation
    #[arg(long)]
    graph: bool,
}

fn main() -> ExitCode {
    if let Err(error) = run(Args::parse()) {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn sibling_output(output: &Path, filename: &str) -> PathBuf {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| parent.join(filename))
        .unwrap_or_else(|| PathBuf::from(filename))
}

fn run(args: Args) -> anyhow::Result<()> {
    let (coordinate, graph) = match (args.coordinate, args.graph) {
        (false, false) => (true, true),
        flags => flags,
    };
    let space = Space::new(args.size)?;
    let written = write_jsonl(
        &args.output,
        generate(
            &space,
            DatasetOptions {
                count: args.count,
                seed: args.seed,
                coordinate,
                graph,
            },
        )?,
    )
    .with_context(|| format!("could not write file `{}`", args.output.display()))?;
    println!("Wrote {} questions to {}", written, args.output.display());

    if coordinate {
        let path = args
            .coordinates_doc
            .unwrap_or_else(|| sibling_output(&args.output, "coordinates.md"));
        write_coordinate_documentation(&path, args.size)
            .with_context(|| format!("could not write file `{}`", path.display()))?;
        println!("Wrote coordinate documentation to {}", path.display());
    }

    if graph {
        let path = args
            .graph_output
            .unwrap_or_else(|| sibling_output(&args.output, "graph.json"));
        write_adjacency_list(&path, &space)
            .with_context(|| format!("could not write file `{}`", path.display()))?;
        println!("Wrote graph adjacency list to {}", path.display());
    }

    let metadata_path = args
        .metadata_output
        .unwrap_or_else(|| sibling_output(&args.output, "metadata.json"));
    write_metadata(&metadata_path, &space)
        .with_context(|| format!("could not write file `{}`", metadata_path.display()))?;
    println!("Wrote space metadata to {}", metadata_path.display());

    Ok(())
}
