use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use dataset::{Coordinate, Question, SpaceMetadata, space_metadata};
use evaluation::{AnswerFile, EvaluationReport, Problem, load_answer_file, load_report};
use serde::Serialize;
use serde_json::Value;
use spatial::{Space, VertexCoordinate, vertex_labels};

const HEX_RADIUS: f64 = 44.0;

#[derive(Debug, Serialize)]
pub struct ViewerPayload {
    pub metadata: SpaceMetadata,
    pub graph: Option<BTreeMap<String, Vec<String>>>,
    pub questions: Vec<Question>,
    pub space: SpaceGeometry,
    pub evaluation: Option<EvaluationReport>,
    pub answers: Option<AnswersPayload>,
}

#[derive(Debug, Serialize)]
pub struct AnswersPayload {
    pub agents: Vec<AgentAnswersMeta>,
    pub rows: Vec<QuestionAnswers>,
}

#[derive(Debug, Serialize)]
pub struct AgentAnswersMeta {
    pub key: String,
    pub problem: String,
}

#[derive(Debug, Serialize)]
pub struct QuestionAnswers {
    pub id: u32,
    pub agents: BTreeMap<String, StoredAnswer>,
}

#[derive(Debug, Serialize)]
pub struct StoredAnswer {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpaceGeometry {
    pub hexes: Vec<HexPixel>,
    pub vertices: Vec<VertexPixel>,
    pub edges: Vec<[usize; 2]>,
}

#[derive(Debug, Serialize)]
pub struct HexPixel {
    pub q: i8,
    pub r: i8,
    pub x: f64,
    pub y: f64,
    pub points: String,
}

#[derive(Debug, Serialize)]
pub struct VertexPixel {
    pub q: i8,
    pub r: i8,
    pub d: i8,
    pub label: String,
    pub x: f64,
    pub y: f64,
}

pub fn load_payload(
    questions_path: &Path,
    metadata_path: &Path,
    graph_path: Option<&Path>,
    evaluation_path: Option<&Path>,
    answer_files: &[PathBuf],
) -> Result<ViewerPayload> {
    let metadata = load_metadata(metadata_path)?;
    let space = Space::new(metadata.space.size).context("invalid space size in metadata")?;
    if space_metadata(&space) != metadata {
        bail!(
            "metadata at {} does not match a space of size {}",
            metadata_path.display(),
            metadata.space.size
        );
    }

    let questions = load_questions(questions_path)?;
    let graph = graph_path.map(load_graph).transpose()?;
    let space_geometry = build_geometry(&space);
    let evaluation = evaluation_path
        .map(load_report)
        .transpose()
        .with_context(|| {
            format!(
                "could not load evaluation from `{}`",
                evaluation_path
                    .unwrap_or(metadata_path)
                    .display()
            )
        })?;
    let loaded_answers = answer_files
        .iter()
        .map(|path| {
            load_answer_file(path)
                .with_context(|| format!("could not read answers from {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let answers = if loaded_answers.is_empty() {
        None
    } else {
        Some(build_answers_payload(&loaded_answers))
    };

    Ok(ViewerPayload {
        metadata,
        graph,
        questions,
        space: space_geometry,
        evaluation,
        answers,
    })
}

fn build_answers_payload(answer_files: &[AnswerFile]) -> AnswersPayload {
    let agents = answer_files
        .iter()
        .map(|file| AgentAnswersMeta {
            key: file.agent_key.clone(),
            problem: problem_name(file.problem).to_string(),
        })
        .collect();
    let mut rows: BTreeMap<u32, BTreeMap<String, StoredAnswer>> = BTreeMap::new();
    for file in answer_files {
        for answer in &file.answers {
            rows.entry(answer.id).or_default().insert(
                file.agent_key.clone(),
                StoredAnswer {
                    prediction: answer.prediction.clone(),
                    error: answer.error.clone(),
                },
            );
        }
    }
    let mut rows: Vec<QuestionAnswers> = rows
        .into_iter()
        .map(|(id, agents)| QuestionAnswers { id, agents })
        .collect();
    rows.sort_by_key(|row| row.id);
    AnswersPayload { agents, rows }
}

fn problem_name(problem: Problem) -> &'static str {
    match problem {
        Problem::Coordinates => "coordinates",
        Problem::Graph => "graph",
    }
}

fn load_metadata(path: &Path) -> Result<SpaceMetadata> {
    let file = File::open(path)
        .with_context(|| format!("could not open metadata file `{}`", path.display()))?;
    serde_json::from_reader(file)
        .with_context(|| format!("could not parse metadata file `{}`", path.display()))
}

fn load_questions(path: &Path) -> Result<Vec<Question>> {
    let file = File::open(path)
        .with_context(|| format!("could not open questions file `{}`", path.display()))?;
    let reader = BufReader::new(file);
    let mut questions = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("could not read `{}`", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let question: Question = serde_json::from_str(&line).with_context(|| {
            format!(
                "could not parse question on line {} of `{}`",
                line_number + 1,
                path.display()
            )
        })?;
        questions.push(question);
    }
    if questions.is_empty() {
        bail!("no questions found in `{}`", path.display());
    }
    Ok(questions)
}

fn load_graph(path: &Path) -> Result<BTreeMap<String, Vec<String>>> {
    let file = File::open(path)
        .with_context(|| format!("could not open graph file `{}`", path.display()))?;
    serde_json::from_reader(file)
        .with_context(|| format!("could not parse graph file `{}`", path.display()))
}

fn build_geometry(space: &Space) -> SpaceGeometry {
    let labels = vertex_labels(space);

    let vertices: Vec<VertexPixel> = space
        .vertices()
        .iter()
        .enumerate()
        .map(|(index, vertex)| {
            let coordinate = coordinate_value(*vertex);
            let (x, y) = vertex_pixel(coordinate.q, coordinate.r, coordinate.d);
            VertexPixel {
                q: coordinate.q,
                r: coordinate.r,
                d: coordinate.d,
                label: labels[index].clone(),
                x,
                y,
            }
        })
        .collect();

    let mut edges = Vec::new();
    let mut seen = BTreeMap::new();
    for (index, vertex) in space.vertices().iter().enumerate() {
        for neighbor in space.neighbors(*vertex) {
            let neighbor_index = space
                .vertices()
                .binary_search(neighbor)
                .expect("neighbor on space");
            let key = if index < neighbor_index {
                (index, neighbor_index)
            } else {
                (neighbor_index, index)
            };
            if seen.insert(key, ()).is_none() {
                edges.push([key.0, key.1]);
            }
        }
    }

    let hexes = space
        .hexes()
        .iter()
        .map(|hex| {
            let (x, y) = hex_center(hex.q, hex.r);
            HexPixel {
                q: hex.q,
                r: hex.r,
                x,
                y,
                points: hex_polygon(hex.q, hex.r),
            }
        })
        .collect();

    SpaceGeometry {
        hexes,
        vertices,
        edges,
    }
}

fn coordinate_value(vertex: VertexCoordinate) -> Coordinate {
    Coordinate {
        q: vertex.hex().q,
        r: vertex.hex().r,
        d: i8::try_from(vertex.direction().get()).expect("direction fits in i8"),
    }
}

fn hex_center(q: i8, r: i8) -> (f64, f64) {
    let q = f64::from(q);
    let r = f64::from(r);
    let x = HEX_RADIUS * 3_f64.sqrt() * (q + r / 2.0);
    let y = HEX_RADIUS * 1.5 * r;
    (x, y)
}

fn vertex_pixel(q: i8, r: i8, d: i8) -> (f64, f64) {
    let (cx, cy) = hex_center(q, r);
    let angle = -std::f64::consts::FRAC_PI_2 + f64::from(d) * std::f64::consts::PI / 3.0;
    (cx + HEX_RADIUS * angle.cos(), cy + HEX_RADIUS * angle.sin())
}

fn hex_polygon(q: i8, r: i8) -> String {
    let (cx, cy) = hex_center(q, r);
    (0..6)
        .map(|corner| {
            let angle =
                -std::f64::consts::FRAC_PI_2 + f64::from(corner) * std::f64::consts::PI / 3.0;
            let x = cx + HEX_RADIUS * angle.cos();
            let y = cy + HEX_RADIUS * angle.sin();
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dataset::DatasetOptions;
    use dataset::generate;
    use dataset::write_jsonl;
    use dataset::write_metadata;

    #[test]
    fn payload_round_trip_from_generated_dataset() {
        let space = Space::new(3).expect("space");
        let questions = generate(
            &space,
            DatasetOptions {
                count: 2,
                seed: 0,
                coordinate: true,
                graph: true,
            },
        )
        .expect("generator")
        .collect::<Result<Vec<_>, _>>()
        .expect("questions");

        let dir = std::env::temp_dir().join(format!("viewer-payload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let questions_path = dir.join("questions.jsonl");
        let metadata_path = dir.join("metadata.json");
        write_jsonl(&questions_path, questions.iter().cloned().map(Ok)).expect("write questions");
        write_metadata(&metadata_path, &space).expect("write metadata");

        let payload =
            load_payload(&questions_path, &metadata_path, None, None, &[]).expect("payload");
        assert_eq!(payload.questions.len(), 2);
        assert_eq!(payload.space.vertices.len(), space.vertices().len());
        assert_eq!(payload.space.hexes.len(), space.hexes().len());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn loads_evaluation_report_when_present() {
        use evaluation::{evaluate_files, write_report};

        let space = Space::new(1).expect("space");
        let questions = generate(
            &space,
            DatasetOptions {
                count: 1,
                seed: 0,
                coordinate: true,
                graph: true,
            },
        )
        .expect("generator")
        .collect::<Result<Vec<_>, _>>()
        .expect("questions");

        let dir = std::env::temp_dir().join(format!("viewer-eval-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let questions_path = dir.join("questions.jsonl");
        let metadata_path = dir.join("metadata.json");
        write_jsonl(&questions_path, questions.iter().cloned().map(Ok)).expect("write questions");
        write_metadata(&metadata_path, &space).expect("write metadata");

        let graph = questions[0].graph.as_ref().expect("graph");
        let answers_path = dir.join("answers.graph.bare.test-llm.jsonl");
        std::fs::write(
            &answers_path,
            format!(
                "{}\n",
                serde_json::json!({
                    "id": questions[0].id,
                    "prediction": graph.path,
                    "error": null,
                })
            ),
        )
        .expect("write answers");
        let answer_file = evaluation::load_answer_file(&answers_path).expect("load answers");
        let report = evaluate_files(&space, &questions, &[answer_file]).expect("evaluate");
        let evaluation_path = dir.join("evaluation.jsonl");
        write_report(&evaluation_path, &report).expect("write report");

        let payload = load_payload(
            &questions_path,
            &metadata_path,
            None,
            Some(&evaluation_path),
            &[answers_path],
        )
        .expect("payload");
        let evaluation = payload.evaluation.expect("evaluation");
        assert_eq!(evaluation.agents, vec!["graph/bare/test-llm".to_string()]);
        assert!(evaluation.rows[0].agents["graph/bare/test-llm"].correct);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn loads_answers_from_directory_by_default() {
        let space = Space::new(1).expect("space");
        let questions = generate(
            &space,
            DatasetOptions {
                count: 1,
                seed: 0,
                coordinate: true,
                graph: true,
            },
        )
        .expect("generator")
        .collect::<Result<Vec<_>, _>>()
        .expect("questions");

        let dir = std::env::temp_dir().join(format!("viewer-answers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let questions_path = dir.join("questions.jsonl");
        let metadata_path = dir.join("metadata.json");
        write_jsonl(&questions_path, questions.iter().cloned().map(Ok)).expect("write questions");
        write_metadata(&metadata_path, &space).expect("write metadata");

        let graph = questions[0].graph.as_ref().expect("graph");
        let answers_path = dir.join("answers.graph.bare.test-llm.jsonl");
        std::fs::write(
            &answers_path,
            format!(
                "{}\n",
                serde_json::json!({
                    "id": questions[0].id,
                    "prediction": graph.path,
                    "error": null,
                })
            ),
        )
        .expect("write answers");

        let payload = load_payload(&questions_path, &metadata_path, None, None, &[answers_path])
            .expect("payload");
        let answers = payload.answers.expect("answers");
        assert_eq!(answers.agents.len(), 1);
        assert_eq!(answers.agents[0].key, "graph/bare/test-llm");
        assert_eq!(answers.rows.len(), 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn graph_path_matches_coordinate_path_on_space() {
        let space = Space::new(3).expect("space");
        let questions = generate(
            &space,
            DatasetOptions {
                count: 5,
                seed: 7,
                coordinate: true,
                graph: true,
            },
        )
        .expect("generator")
        .collect::<Result<Vec<_>, _>>()
        .expect("questions");

        let dir = std::env::temp_dir().join(format!("viewer-labels-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let questions_path = dir.join("questions.jsonl");
        let metadata_path = dir.join("metadata.json");
        write_jsonl(&questions_path, questions.iter().cloned().map(Ok)).expect("write questions");
        write_metadata(&metadata_path, &space).expect("write metadata");

        let payload =
            load_payload(&questions_path, &metadata_path, None, None, &[]).expect("payload");
        let label_to_vertex: std::collections::BTreeMap<_, _> = payload
            .space
            .vertices
            .iter()
            .map(|vertex| (vertex.label.clone(), (vertex.q, vertex.r, vertex.d)))
            .collect();

        for question in &payload.questions {
            let coordinates = question.coordinates.as_ref().expect("coordinates");
            let graph = question.graph.as_ref().expect("graph");
            let mapped: Vec<_> = graph
                .path
                .iter()
                .map(|label| label_to_vertex[label])
                .collect();
            let expected: Vec<_> = coordinates
                .path
                .iter()
                .map(|coordinate| (coordinate.q, coordinate.r, coordinate.d))
                .collect();
            assert_eq!(mapped, expected, "graph labels must match coordinate path");
        }

        let _ = std::fs::remove_dir_all(dir);
    }
}
