use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context;
use clap::Parser;
use evaluation::{
    REPORT_FILENAME, agent_summaries, evaluate_files, load_answer_file, load_jsonl, load_size,
    write_report,
};
use spatial::Space;

/// Score solver answers against a spatial shortest-path dataset
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Questions JSONL produced by the dataset generator
    #[arg(long, default_value = "data/questions.jsonl", value_name = "FILE")]
    questions: PathBuf,

    /// Answers JSONL produced by the solver (repeat for multiple agents)
    #[arg(long, value_name = "FILE", required = true)]
    answers: Vec<PathBuf>,

    /// Space metadata JSON (defaults to metadata.json next to the questions file)
    #[arg(long, value_name = "FILE")]
    metadata: Option<PathBuf>,

    /// Evaluation report JSONL (defaults to evaluation.jsonl next to the questions file)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,
}

fn main() -> ExitCode {
    if let Err(error) = run(Args::parse()) {
        eprintln!("{error:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: Args) -> anyhow::Result<()> {
    let metadata = args
        .metadata
        .unwrap_or_else(|| sibling(&args.questions, "metadata.json"));
    let size = load_size(&metadata)?;
    let space = Space::new(size)?;
    let questions = load_jsonl(&args.questions)
        .with_context(|| format!("could not read questions from {}", args.questions.display()))?;
    let answer_files = args
        .answers
        .iter()
        .map(|path| {
            load_answer_file(path)
                .with_context(|| format!("could not read answers from {}", path.display()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let report = evaluate_files(&space, &questions, &answer_files)
        .map_err(|error| anyhow::anyhow!(error))?;
    let output = args
        .output
        .unwrap_or_else(|| sibling(&args.questions, REPORT_FILENAME));
    write_report(&output, &report)
        .with_context(|| format!("could not write {}", output.display()))?;
    for summary in agent_summaries(&report) {
        println!("{summary}");
    }
    println!("Wrote {}", output.display());
    Ok(())
}

fn sibling(path: &Path, filename: &str) -> PathBuf {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| parent.join(filename))
        .unwrap_or_else(|| PathBuf::from(filename))
}
