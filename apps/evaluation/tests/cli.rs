use std::fs;

use assert_cmd::cargo::cargo_bin_cmd;
use assert_fs::prelude::*;
use dataset::{DatasetOptions, generate, write_jsonl, write_metadata};
use predicates::prelude::*;
use serde_json::json;
use spatial::Space;

#[test]
fn scores_a_stored_graph_path() -> Result<(), Box<dyn std::error::Error>> {
    let dir = assert_fs::TempDir::new()?;
    let space = Space::new(1)?;
    let questions = generate(
        &space,
        DatasetOptions {
            count: 1,
            seed: 0,
            coordinate: true,
            graph: true,
        },
    )?
    .collect::<Result<Vec<_>, _>>()?;
    let questions_path = dir.child("questions.jsonl");
    write_jsonl(questions_path.path(), questions.iter().cloned().map(Ok))?;
    let metadata_path = dir.child("metadata.json");
    write_metadata(metadata_path.path(), &space)?;

    let graph = questions[0].graph.as_ref().expect("graph");
    let answers_path = dir.child("answers.graph.bare.test-llm.jsonl");
    answers_path.write_str(&format!(
        "{}\n",
        json!({
            "id": 0,
            "prediction": graph.path,
            "error": null,
        })
    ))?;
    let report = dir.child("report.jsonl");

    cargo_bin_cmd!("evaluation")
        .arg("--questions")
        .arg(questions_path.path())
        .arg("--answers")
        .arg(answers_path.path())
        .arg("--metadata")
        .arg(metadata_path.path())
        .arg("--output")
        .arg(report.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("1/1 correct for graph/bare/test-llm"));

    let written = fs::read_to_string(report.path())?;
    assert!(written.contains(r#""correct":true"#));
    assert!(written.contains(r#""valid":true"#));
    assert!(!written.contains("\"path\""));
    Ok(())
}

#[test]
fn merges_two_answer_files() -> Result<(), Box<dyn std::error::Error>> {
    let dir = assert_fs::TempDir::new()?;
    let space = Space::new(1)?;
    let questions = generate(
        &space,
        DatasetOptions {
            count: 2,
            seed: 0,
            coordinate: true,
            graph: true,
        },
    )?
    .collect::<Result<Vec<_>, _>>()?;
    let questions_path = dir.child("questions.jsonl");
    write_jsonl(questions_path.path(), questions.iter().cloned().map(Ok))?;
    let metadata_path = dir.child("metadata.json");
    write_metadata(metadata_path.path(), &space)?;

    let graph0 = questions[0].graph.as_ref().expect("graph");
    let graph1 = questions[1].graph.as_ref().expect("graph");
    let answers_a = dir.child("answers.graph.bare.llm-a.jsonl");
    answers_a.write_str(&format!(
        "{}\n{}\n",
        json!({"id": 0, "prediction": graph0.path, "error": null}),
        json!({"id": 1, "prediction": graph1.path, "error": null}),
    ))?;
    let answers_b = dir.child("answers.graph.bare.llm-b.jsonl");
    answers_b.write_str(&format!(
        "{}\n",
        json!({"id": 0, "prediction": graph0.path, "error": null}),
    ))?;
    let report = dir.child("report.jsonl");

    cargo_bin_cmd!("evaluation")
        .arg("--questions")
        .arg(questions_path.path())
        .arg("--answers")
        .arg(answers_a.path())
        .arg("--answers")
        .arg(answers_b.path())
        .arg("--metadata")
        .arg(metadata_path.path())
        .arg("--output")
        .arg(report.path())
        .assert()
        .success();

    let written = fs::read_to_string(report.path())?;
    assert_eq!(written.lines().count(), 2);
    assert!(written.contains("graph/bare/llm-a"));
    assert!(written.contains("graph/bare/llm-b"));
    assert!(written.contains(r#""error":"not answered""#));
    Ok(())
}
