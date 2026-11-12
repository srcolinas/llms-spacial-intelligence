//! Score solver answers against shortest-path questions.

mod grade;

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use dataset::{Question, SpaceMetadata};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use spatial::Space;

pub use grade::{PathGrade, grade_coordinate_path, grade_graph_path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Problem {
    Coordinates,
    Graph,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coordinates => write!(f, "coordinates"),
            Self::Graph => write!(f, "graph"),
        }
    }
}

/// Solver answer row: `{id, prediction, error, mistakes?}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SolverAnswerRow {
    pub id: u32,
    #[serde(default)]
    pub prediction: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub mistakes: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AnswerFileMeta {
    pub attempts: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerFile {
    pub path: PathBuf,
    pub agent_key: String,
    pub problem: Problem,
    pub answers: Vec<SolverAnswerRow>,
    pub attempts: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentResult {
    pub valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub const REPORT_FILENAME: &str = "evaluation.jsonl";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionReport {
    pub id: u32,
    pub expected_length: usize,
    pub agents: BTreeMap<String, AgentResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub agents: Vec<String>,
    pub rows: Vec<QuestionReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSummary {
    pub agent_key: String,
    pub valid: usize,
    pub optimal_length: usize,
    pub answered: usize,
    pub total: usize,
    pub attempts: Option<u32>,
    pub mistakes: usize,
}

impl fmt::Display for AgentSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "valid: {}/{} for {} (optimal length: {}/{}",
            self.valid, self.answered, self.agent_key, self.optimal_length, self.valid,
        )?;
        if let Some(attempts) = self.attempts {
            write!(f, ", attempts: {attempts}")?;
        }
        if self.mistakes > 0 {
            write!(f, ", mistakes: {}", self.mistakes)?;
        }
        write!(f, ")")
    }
}

#[derive(Debug)]
pub enum EvaluationError {
    UnknownQuestionId { id: u32, questions: usize },
    DuplicateAnswer { agent_key: String, id: u32 },
    DuplicateAgentKey { agent_key: String },
    InvalidAnswerFilename { path: PathBuf },
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownQuestionId { id, questions } => {
                write!(f, "answer id {id} is outside the {questions} questions")
            }
            Self::DuplicateAnswer { agent_key, id } => {
                write!(f, "duplicate answer id {id} in {agent_key}")
            }
            Self::DuplicateAgentKey { agent_key } => {
                write!(f, "duplicate agent key {agent_key}")
            }
            Self::InvalidAnswerFilename { path } => write!(
                f,
                "could not parse agent metadata from filename `{}`; expected answers.<problem>.<agent>.<llm>.jsonl",
                path.display()
            ),
        }
    }
}

impl std::error::Error for EvaluationError {}

pub fn load_jsonl<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))?;
    let mut rows = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(line).with_context(|| {
            format!(
                "invalid JSON on line {} of {}",
                line_number + 1,
                path.display()
            )
        })?;
        rows.push(row);
    }
    Ok(rows)
}

pub fn load_size(path: &Path) -> anyhow::Result<u32> {
    let text =
        fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))?;
    let metadata: SpaceMetadata = serde_json::from_str(&text)
        .with_context(|| format!("invalid space metadata at {}", path.display()))?;
    Ok(metadata.space.size)
}

/// Parse `answers.<problem>.<agent>.<llm>.jsonl` into agent metadata.
pub fn parse_answer_filename(path: &Path) -> Result<(Problem, String, String), EvaluationError> {
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| EvaluationError::InvalidAnswerFilename {
            path: path.to_path_buf(),
        })?;
    let mut parts = stem.splitn(4, '.');
    let prefix = parts.next();
    let problem = parts.next();
    let agent = parts.next();
    let llm = parts.next();
    if prefix != Some("answers") || problem.is_none() || agent.is_none() || llm.is_none() {
        return Err(EvaluationError::InvalidAnswerFilename {
            path: path.to_path_buf(),
        });
    }
    let problem = match problem.unwrap() {
        "coordinates" => Problem::Coordinates,
        "graph" => Problem::Graph,
        _ => {
            return Err(EvaluationError::InvalidAnswerFilename {
                path: path.to_path_buf(),
            });
        }
    };
    Ok((
        problem,
        agent.unwrap().to_string(),
        llm.unwrap().to_string(),
    ))
}

pub fn agent_key(problem: Problem, agent: &str, llm: &str) -> String {
    format!("{problem}/{agent}/{llm}")
}

pub fn answer_meta_path(path: &Path) -> PathBuf {
    path.with_extension("meta.json")
}

pub fn load_answer_meta(path: &Path) -> anyhow::Result<Option<AnswerFileMeta>> {
    let meta_path = answer_meta_path(path);
    if !meta_path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&meta_path)
        .with_context(|| format!("could not read {}", meta_path.display()))?;
    let metadata: AnswerFileMeta = serde_json::from_str(&text)
        .with_context(|| format!("invalid answer metadata at {}", meta_path.display()))?;
    Ok(Some(metadata))
}

pub fn load_answer_file(path: &Path) -> anyhow::Result<AnswerFile> {
    let (problem, agent, llm) =
        parse_answer_filename(path).map_err(|error| anyhow::anyhow!(error))?;
    let key = agent_key(problem, &agent, &llm);
    let answers = load_jsonl(path)?;
    let attempts = load_answer_meta(path)?.map(|meta| meta.attempts);
    Ok(AnswerFile {
        path: path.to_path_buf(),
        agent_key: key,
        problem,
        answers,
        attempts,
    })
}

pub fn evaluate_files(
    space: &Space,
    questions: &[Question],
    answer_files: &[AnswerFile],
) -> Result<EvaluationReport, EvaluationError> {
    let mut agent_keys = Vec::new();
    let mut seen_keys = HashSet::new();
    for file in answer_files {
        if !seen_keys.insert(file.agent_key.clone()) {
            return Err(EvaluationError::DuplicateAgentKey {
                agent_key: file.agent_key.clone(),
            });
        }
        agent_keys.push(file.agent_key.clone());
    }

    let mut rows: Vec<QuestionReport> = questions
        .iter()
        .map(|question| QuestionReport {
            id: question.id,
            expected_length: expected_length(question),
            agents: BTreeMap::new(),
        })
        .collect();

    let row_by_id: BTreeMap<u32, usize> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| (row.id, index))
        .collect();

    for file in answer_files {
        let mut seen_ids = HashSet::new();
        for answer in &file.answers {
            if !seen_ids.insert(answer.id) {
                return Err(EvaluationError::DuplicateAnswer {
                    agent_key: file.agent_key.clone(),
                    id: answer.id,
                });
            }
            let Some(&row_index) = row_by_id.get(&answer.id) else {
                return Err(EvaluationError::UnknownQuestionId {
                    id: answer.id,
                    questions: questions.len(),
                });
            };
            let question = &questions[row_index];
            let result = score_answer(space, question, file.problem, answer);
            rows[row_index]
                .agents
                .insert(file.agent_key.clone(), result);
        }
    }

    for row in &mut rows {
        for key in &agent_keys {
            row.agents
                .entry(key.clone())
                .or_insert_with(|| AgentResult {
                    valid: false,
                    length: None,
                    error: Some("not answered".to_string()),
                });
        }
    }

    rows.sort_by_key(|row| row.id);

    Ok(EvaluationReport {
        agents: agent_keys,
        rows,
    })
}

pub fn agent_summaries(
    report: &EvaluationReport,
    answer_files: &[AnswerFile],
) -> Vec<AgentSummary> {
    let total = report.rows.len();
    report
        .agents
        .iter()
        .map(|agent_key| {
            let mut valid = 0;
            let mut optimal_length = 0;
            let mut answered = 0;
            for row in &report.rows {
                let Some(result) = row.agents.get(agent_key) else {
                    continue;
                };
                if result.error.as_deref() == Some("not answered") {
                    continue;
                }
                answered += 1;
                if result.valid {
                    valid += 1;
                    if result.length == Some(row.expected_length) {
                        optimal_length += 1;
                    }
                }
            }
            let file = answer_files
                .iter()
                .find(|file| file.agent_key == *agent_key);
            let attempts = file.and_then(|file| file.attempts);
            let mistakes = file
                .map(|file| {
                    file.answers
                        .iter()
                        .filter_map(|answer| answer.mistakes)
                        .map(|count| count as usize)
                        .sum()
                })
                .unwrap_or(0);
            AgentSummary {
                agent_key: agent_key.clone(),
                valid,
                optimal_length,
                answered,
                total,
                attempts,
                mistakes,
            }
        })
        .collect()
}

pub fn write_report(path: &Path, report: &EvaluationReport) -> std::io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in &report.rows {
        serde_json::to_writer(&mut file, row).map_err(std::io::Error::other)?;
        writeln!(file)?;
    }
    Ok(())
}

pub fn load_report(path: &Path) -> anyhow::Result<EvaluationReport> {
    let mut rows: Vec<QuestionReport> = load_jsonl(path)?;
    let mut agents = Vec::new();
    let mut seen = HashSet::new();
    for row in &rows {
        for key in row.agents.keys() {
            if seen.insert(key.clone()) {
                agents.push(key.clone());
            }
        }
    }
    agents.sort();
    rows.sort_by_key(|row| row.id);
    Ok(EvaluationReport { agents, rows })
}

/// Infer the data directory from a questions file path.
pub fn data_dir_for_questions(questions: &Path) -> anyhow::Result<PathBuf> {
    questions
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "could not infer data directory from `{}`",
                questions.display()
            )
        })
}

/// Use explicit answer paths, or discover `answers.*.jsonl` next to the questions file.
pub fn resolve_answer_paths(
    questions: &Path,
    explicit: &[PathBuf],
) -> anyhow::Result<Vec<PathBuf>> {
    if !explicit.is_empty() {
        return Ok(explicit.to_vec());
    }
    let dir = data_dir_for_questions(questions)?;
    discover_answer_files(&dir)
}

/// Discover solver answer files named `answers.<problem>.<agent>.<llm>.jsonl` in a directory.
pub fn discover_answer_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in
        fs::read_dir(dir).with_context(|| format!("could not read directory {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("could not read entry in {}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
            continue;
        }
        if parse_answer_filename(&path).is_ok() {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn expected_length(question: &Question) -> usize {
    if let Some(item) = &question.coordinates {
        return item.path.len().saturating_sub(1);
    }
    if let Some(item) = &question.graph {
        return item.path.len().saturating_sub(1);
    }
    0
}

fn score_answer(
    space: &Space,
    question: &Question,
    problem: Problem,
    answer: &SolverAnswerRow,
) -> AgentResult {
    if let Some(error) = &answer.error {
        return failed(error.clone());
    }
    let Some(predicted) = &answer.prediction else {
        return failed("missing predicted path".to_string());
    };
    match problem {
        Problem::Coordinates => score_coordinates(space, question, predicted),
        Problem::Graph => score_graph(space, question, predicted),
    }
}

fn score_coordinates(space: &Space, question: &Question, predicted: &Value) -> AgentResult {
    let Some(item) = &question.coordinates else {
        return failed("question has no coordinates representation".to_string());
    };
    let predicted: Vec<dataset::Coordinate> = match serde_json::from_value(predicted.clone()) {
        Ok(path) => path,
        Err(error) => {
            return failed(format!("predicted path is not a coordinate path: {error}"));
        }
    };
    graded(grade_coordinate_path(
        space,
        item.origin,
        item.destination,
        &predicted,
    ))
}

fn score_graph(space: &Space, question: &Question, predicted: &Value) -> AgentResult {
    let Some(item) = &question.graph else {
        return failed("question has no graph representation".to_string());
    };
    let predicted: Vec<String> = match serde_json::from_value(predicted.clone()) {
        Ok(path) => path,
        Err(error) => {
            return failed(format!("predicted path is not a graph path: {error}"));
        }
    };
    graded(grade_graph_path(
        space,
        &item.origin,
        &item.destination,
        &predicted,
    ))
}

fn graded(grade: PathGrade) -> AgentResult {
    AgentResult {
        valid: grade.valid,
        length: grade.length,
        error: None,
    }
}

fn failed(error: String) -> AgentResult {
    AgentResult {
        valid: false,
        length: None,
        error: Some(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dataset::{DatasetOptions, generate};
    use serde_json::json;

    fn questions(count: usize) -> (Space, Vec<Question>) {
        let space = Space::new(1).expect("space");
        let questions = generate(
            &space,
            DatasetOptions {
                count,
                seed: 0,
                coordinate: true,
                graph: true,
            },
        )
        .expect("questions")
        .collect::<Result<Vec<_>, _>>()
        .expect("questions");
        (space, questions)
    }

    fn answer_file(
        problem: Problem,
        agent: &str,
        llm: &str,
        rows: Vec<SolverAnswerRow>,
    ) -> AnswerFile {
        AnswerFile {
            path: PathBuf::from(format!("answers.{problem}.{agent}.{llm}.jsonl")),
            agent_key: agent_key(problem, agent, llm),
            problem,
            answers: rows,
            attempts: None,
        }
    }

    #[test]
    fn stored_graph_path_is_correct() {
        let (space, questions) = questions(1);
        let graph = questions[0].graph.as_ref().expect("graph");
        let file = answer_file(
            Problem::Graph,
            "bare",
            "test-llm",
            vec![SolverAnswerRow {
                id: questions[0].id,
                prediction: Some(json!(graph.path)),
                error: None,
                mistakes: None,
            }],
        );
        let report = evaluate_files(&space, &questions, &[file]).expect("evaluation");
        let result = report.rows[0]
            .agents
            .get("graph/bare/test-llm")
            .expect("agent");
        assert!(result.valid);
        assert_eq!(result.length, Some(report.rows[0].expected_length));
        assert_eq!(result.error, None);
    }

    #[test]
    fn a_solver_error_is_not_a_valid_path() {
        let (space, questions) = questions(1);
        let file = answer_file(
            Problem::Graph,
            "bare",
            "test-llm",
            vec![SolverAnswerRow {
                id: questions[0].id,
                prediction: None,
                error: Some("model timed out".to_string()),
                mistakes: None,
            }],
        );
        let report = evaluate_files(&space, &questions, &[file]).expect("evaluation");
        let result = report.rows[0]
            .agents
            .get("graph/bare/test-llm")
            .expect("agent");
        assert!(!result.valid);
        assert_eq!(result.error.as_deref(), Some("model timed out"));
    }

    #[test]
    fn duplicate_answer_id_is_rejected() {
        let (space, questions) = questions(1);
        let graph = questions[0].graph.as_ref().expect("graph");
        let row = SolverAnswerRow {
            id: questions[0].id,
            prediction: Some(json!(graph.path)),
            error: None,
            mistakes: None,
        };
        let file = answer_file(Problem::Graph, "bare", "test-llm", vec![row.clone(), row]);
        let error = evaluate_files(&space, &questions, &[file]).unwrap_err();
        assert!(matches!(
            error,
            EvaluationError::DuplicateAnswer { id: 0, .. }
        ));
    }

    #[test]
    fn merges_two_agent_files_by_question_id() {
        let (space, questions) = questions(2);
        let graph0 = questions[0].graph.as_ref().expect("graph").path.clone();
        let graph1 = questions[1].graph.as_ref().expect("graph").path.clone();
        let file_a = answer_file(
            Problem::Graph,
            "bare",
            "llm-a",
            vec![
                SolverAnswerRow {
                    id: 0,
                    prediction: Some(json!(graph0)),
                    error: None,
                    mistakes: None,
                },
                SolverAnswerRow {
                    id: 1,
                    prediction: Some(json!(graph1)),
                    error: None,
                    mistakes: None,
                },
            ],
        );
        let file_b = answer_file(
            Problem::Graph,
            "bare",
            "llm-b",
            vec![SolverAnswerRow {
                id: 0,
                prediction: Some(json!(graph0)),
                error: None,
                mistakes: None,
            }],
        );
        let report = evaluate_files(&space, &questions, &[file_a, file_b]).expect("evaluation");
        assert_eq!(report.agents.len(), 2);
        assert_eq!(report.rows.len(), 2);
        assert!(report.rows[0].agents["graph/bare/llm-a"].valid);
        assert!(report.rows[0].agents["graph/bare/llm-b"].valid);
        assert_eq!(
            report.rows[1].agents["graph/bare/llm-b"].error.as_deref(),
            Some("not answered")
        );
    }

    #[test]
    fn optimal_length_only_counts_valid_paths() {
        let mut agents = BTreeMap::new();
        agents.insert(
            "graph/bare/test-llm".to_string(),
            AgentResult {
                valid: false,
                length: Some(3),
                error: None,
            },
        );
        let report = EvaluationReport {
            agents: vec!["graph/bare/test-llm".to_string()],
            rows: vec![QuestionReport {
                id: 0,
                expected_length: 3,
                agents,
            }],
        };
        let file = answer_file(Problem::Graph, "bare", "test-llm", vec![]);
        let summaries = agent_summaries(&report, &[file]);
        assert_eq!(summaries[0].valid, 0);
        assert_eq!(summaries[0].optimal_length, 0);
    }

    #[test]
    fn agent_summaries_aggregate_attempts_and_mistakes() {
        let mut agents = BTreeMap::new();
        agents.insert(
            "graph/validating/test-llm".to_string(),
            AgentResult {
                valid: true,
                length: Some(2),
                error: None,
            },
        );
        let report = EvaluationReport {
            agents: vec!["graph/validating/test-llm".to_string()],
            rows: vec![QuestionReport {
                id: 0,
                expected_length: 2,
                agents,
            }],
        };
        let file = AnswerFile {
            path: PathBuf::from("answers.graph.validating.test-llm.jsonl"),
            agent_key: "graph/validating/test-llm".to_string(),
            problem: Problem::Graph,
            answers: vec![
                SolverAnswerRow {
                    id: 0,
                    prediction: Some(json!(["A", "B", "C"])),
                    error: None,
                    mistakes: Some(2),
                },
                SolverAnswerRow {
                    id: 1,
                    prediction: Some(json!(["A", "C"])),
                    error: None,
                    mistakes: Some(1),
                },
            ],
            attempts: Some(3),
        };
        let summaries = agent_summaries(&report, &[file]);
        assert_eq!(summaries[0].attempts, Some(3));
        assert_eq!(summaries[0].mistakes, 3);
    }

    #[test]
    fn resolve_answer_paths_uses_explicit_files() {
        let explicit = vec![PathBuf::from("answers.graph.bare.llm-a.jsonl")];
        let resolved =
            resolve_answer_paths(Path::new("data/questions.jsonl"), &explicit).expect("resolve");
        assert_eq!(resolved, explicit);
    }

    #[test]
    fn resolve_answer_paths_discovers_from_questions_directory() {
        let dir = std::env::temp_dir().join(format!("evaluation-resolve-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("answers.graph.bare.llm-a.jsonl"), "{}\n").expect("write");
        std::fs::write(dir.join("answers.coordinates.bare.llm-b.jsonl"), "{}\n").expect("write");
        let questions = dir.join("questions.jsonl");
        std::fs::write(&questions, "{}\n").expect("write");
        let resolved = resolve_answer_paths(&questions, &[]).expect("resolve");
        assert_eq!(resolved.len(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn discovers_answer_files_in_directory() {
        let dir = std::env::temp_dir().join(format!("evaluation-discover-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("answers.graph.bare.llm-a.jsonl"), "{}\n").expect("write");
        std::fs::write(dir.join("answers.coordinates.bare.llm-b.jsonl"), "{}\n").expect("write");
        std::fs::write(dir.join("questions.jsonl"), "{}\n").expect("write");
        let paths = discover_answer_files(&dir).expect("discover");
        assert_eq!(paths.len(), 2);
        assert!(
            paths[0]
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("answers.")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn report_does_not_contain_paths() {
        let (space, questions) = questions(1);
        let graph = questions[0].graph.as_ref().expect("graph");
        let file = answer_file(
            Problem::Graph,
            "bare",
            "test-llm",
            vec![SolverAnswerRow {
                id: questions[0].id,
                prediction: Some(json!(graph.path)),
                error: None,
                mistakes: None,
            }],
        );
        let report = evaluate_files(&space, &questions, &[file]).expect("evaluation");
        let dir = std::env::temp_dir().join(format!("evaluation-report-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join(REPORT_FILENAME);
        write_report(&path, &report).expect("write");
        let written = std::fs::read_to_string(&path).expect("read");
        assert!(!written.contains("\"path\""));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn parses_solver_answer_filename() {
        let path = Path::new("data/answers.graph.bare.ollama-llama3.1-8b.jsonl");
        let (problem, agent, llm) = parse_answer_filename(path).expect("parse");
        assert_eq!(problem, Problem::Graph);
        assert_eq!(agent, "bare");
        assert_eq!(llm, "ollama-llama3.1-8b");
        assert_eq!(
            agent_key(problem, &agent, &llm),
            "graph/bare/ollama-llama3.1-8b"
        );
    }
}
