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

/// Solver answer row: `{id, prediction, error}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SolverAnswerRow {
    pub id: u32,
    #[serde(default)]
    pub prediction: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerFile {
    pub path: PathBuf,
    pub agent_key: String,
    pub problem: Problem,
    pub answers: Vec<SolverAnswerRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentResult {
    pub correct: bool,
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
    pub correct: usize,
    pub valid: usize,
    pub optimal_length: usize,
    pub answered: usize,
    pub total: usize,
}

impl fmt::Display for AgentSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} correct for {} (valid: {}/{}, optimal length: {}/{})",
            self.correct,
            self.total,
            self.agent_key,
            self.valid,
            self.answered,
            self.optimal_length,
            self.answered,
        )
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
            Self::UnknownQuestionId { id, questions } => write!(
                f,
                "answer id {id} is outside the {questions} questions"
            ),
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

pub fn load_answer_file(path: &Path) -> anyhow::Result<AnswerFile> {
    let (problem, agent, llm) = parse_answer_filename(path).map_err(|error| {
        anyhow::anyhow!(error)
    })?;
    let key = agent_key(problem, &agent, &llm);
    let answers = load_jsonl(path)?;
    Ok(AnswerFile {
        path: path.to_path_buf(),
        agent_key: key,
        problem,
        answers,
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
            row.agents.entry(key.clone()).or_insert_with(|| AgentResult {
                correct: false,
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

pub fn agent_summaries(report: &EvaluationReport) -> Vec<AgentSummary> {
    let total = report.rows.len();
    report
        .agents
        .iter()
        .map(|agent_key| {
            let mut correct = 0;
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
                }
                if result.length == Some(row.expected_length) {
                    optimal_length += 1;
                }
                if result.correct {
                    correct += 1;
                }
            }
            AgentSummary {
                agent_key: agent_key.clone(),
                correct,
                valid,
                optimal_length,
                answered,
                total,
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

/// Discover solver answer files named `answers.<problem>.<agent>.<llm>.jsonl` in a directory.
pub fn discover_answer_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)
        .with_context(|| format!("could not read directory {}", dir.display()))?
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
        correct: grade.correct,
        valid: grade.valid,
        length: grade.length,
        error: None,
    }
}

fn failed(error: String) -> AgentResult {
    AgentResult {
        correct: false,
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

    fn answer_file(problem: Problem, agent: &str, llm: &str, rows: Vec<SolverAnswerRow>) -> AnswerFile {
        AnswerFile {
            path: PathBuf::from(format!("answers.{problem}.{agent}.{llm}.jsonl")),
            agent_key: agent_key(problem, agent, llm),
            problem,
            answers: rows,
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
            }],
        );
        let report = evaluate_files(&space, &questions, &[file]).expect("evaluation");
        let result = report.rows[0]
            .agents
            .get("graph/bare/test-llm")
            .expect("agent");
        assert!(result.correct);
        assert!(result.valid);
        assert_eq!(result.length, Some(report.rows[0].expected_length));
        assert_eq!(result.error, None);
    }

    #[test]
    fn a_solver_error_is_not_a_correct_path() {
        let (space, questions) = questions(1);
        let file = answer_file(
            Problem::Graph,
            "bare",
            "test-llm",
            vec![SolverAnswerRow {
                id: questions[0].id,
                prediction: None,
                error: Some("model timed out".to_string()),
            }],
        );
        let report = evaluate_files(&space, &questions, &[file]).expect("evaluation");
        let result = report.rows[0]
            .agents
            .get("graph/bare/test-llm")
            .expect("agent");
        assert!(!result.correct);
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
                },
                SolverAnswerRow {
                    id: 1,
                    prediction: Some(json!(graph1)),
                    error: None,
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
            }],
        );
        let report = evaluate_files(&space, &questions, &[file_a, file_b]).expect("evaluation");
        assert_eq!(report.agents.len(), 2);
        assert_eq!(report.rows.len(), 2);
        assert!(report.rows[0].agents["graph/bare/llm-a"].correct);
        assert!(report.rows[0].agents["graph/bare/llm-b"].correct);
        assert_eq!(
            report.rows[1].agents["graph/bare/llm-b"].error.as_deref(),
            Some("not answered")
        );
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
        assert!(paths[0].file_name().unwrap().to_str().unwrap().starts_with("answers."));
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
