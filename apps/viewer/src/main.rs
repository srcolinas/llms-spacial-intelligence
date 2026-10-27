mod page;
mod payload;

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::Parser;
use evaluation::{REPORT_FILENAME, discover_answer_files};
use tiny_http::{Header, Response, Server};

use crate::payload::load_payload;

/// Visualize shortest-path questions on a hex space
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Data directory, or a questions JSONL file whose siblings hold the rest of the dataset
    #[arg(default_value = "data")]
    path: PathBuf,

    /// Questions JSONL (defaults to questions.jsonl in the data directory)
    #[arg(long, value_name = "FILE")]
    questions: Option<PathBuf>,

    /// Space metadata JSON (defaults to metadata.json in the data directory)
    #[arg(long, value_name = "FILE")]
    metadata: Option<PathBuf>,

    /// Graph adjacency list (defaults to graph.json in the data directory when present)
    #[arg(long, value_name = "FILE")]
    graph: Option<PathBuf>,

    /// Evaluation report JSONL (defaults to evaluation.jsonl in the data directory when present)
    #[arg(long, value_name = "FILE")]
    evaluation: Option<PathBuf>,

    /// Solver answers JSONL (repeat to override; defaults to answers.*.jsonl in the data directory)
    #[arg(long, value_name = "FILE")]
    answers: Vec<PathBuf>,

    /// Do not open a browser tab automatically
    #[arg(long)]
    no_browser: bool,
}

struct ResolvedInputs {
    questions: PathBuf,
    metadata: PathBuf,
    graph: Option<PathBuf>,
    evaluation: Option<PathBuf>,
    answers: Vec<PathBuf>,
}

fn main() -> ExitCode {
    if let Err(error) = run(Args::parse()) {
        eprintln!("{error:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn data_dir(args: &Args) -> Result<PathBuf> {
    if args.path.is_dir() {
        return Ok(args.path.clone());
    }
    if args.path.is_file() {
        return args
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow::anyhow!("could not infer data directory from `{}`", args.path.display()));
    }
    bail!("path not found: `{}`", args.path.display());
}

fn resolve_inputs(args: &Args) -> Result<ResolvedInputs> {
    let dir = data_dir(args)?;
    let questions = args
        .questions
        .clone()
        .unwrap_or_else(|| dir.join("questions.jsonl"));
    if !questions.is_file() {
        bail!(
            "questions file not found at `{}`; pass --questions or a data directory containing questions.jsonl",
            questions.display()
        );
    }
    let metadata = args
        .metadata
        .clone()
        .unwrap_or_else(|| dir.join("metadata.json"));
    let graph = args.graph.clone().or_else(|| {
        let candidate = dir.join("graph.json");
        candidate.is_file().then_some(candidate)
    });
    let evaluation = args.evaluation.clone().or_else(|| {
        let candidate = dir.join(REPORT_FILENAME);
        candidate.is_file().then_some(candidate)
    });
    let answers = if args.answers.is_empty() {
        discover_answer_files(&dir)?
    } else {
        args.answers.clone()
    };
    Ok(ResolvedInputs {
        questions,
        metadata,
        graph,
        evaluation,
        answers,
    })
}

fn run(args: Args) -> Result<()> {
    let inputs = resolve_inputs(&args)?;
    let data_dir = data_dir(&args)?;
    let payload = load_payload(
        &inputs.questions,
        &inputs.metadata,
        inputs.graph.as_deref(),
        inputs.evaluation.as_deref(),
        &inputs.answers,
    )
    .with_context(|| format!("could not load data from `{}`", data_dir.display()))?;
    let page = page::render_page(&payload).into_bytes();

    let listener = TcpListener::bind("127.0.0.1:0").context("could not bind local port")?;
    let address = listener
        .local_addr()
        .context("could not read local address")?;
    let server = Server::from_listener(listener, None)
        .map_err(|error| anyhow::anyhow!("could not start HTTP server: {error}"))?;
    let url = format!("http://{address}/");
    println!("{url}");

    if !args.no_browser {
        let _ = webbrowser::open(&url);
    }

    for request in server.incoming_requests() {
        let response = Response::from_data(page.clone())
            .with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap());
        let _ = request.respond(response);
    }

    Ok(())
}
