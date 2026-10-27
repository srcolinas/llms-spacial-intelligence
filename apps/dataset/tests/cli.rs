use assert_cmd::cargo::cargo_bin_cmd;
use assert_fs::prelude::*;
use predicates::prelude::*;

/// Minimal invocation: small space, fixed seed, both representations (the default).
#[test]
fn generates_both_representations_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let dir = assert_fs::TempDir::new()?;
    let output = dir.child("questions.jsonl");

    cargo_bin_cmd!("dataset")
        .args(["--size", "1", "--count", "2", "--seed", "0"])
        .arg("-o")
        .arg(output.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Wrote 2 questions to"))
        .stdout(predicate::str::contains(
            "Wrote coordinate documentation to",
        ))
        .stdout(predicate::str::contains("Wrote graph adjacency list to"))
        .stdout(predicate::str::contains("Wrote space metadata to"));

    let contents = std::fs::read_to_string(output.path())?;
    let lines: Vec<_> = contents.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains(r#""coordinates""#));
    assert!(lines[0].contains(r#""graph""#));
    assert!(lines[0].contains(r#""origin""#));
    assert!(lines[0].contains(r#""destination""#));
    assert!(lines[0].contains(r#""path""#));
    assert!(!lines[0].contains(r#""question""#));

    let graph = dir.child("graph.json");
    assert!(graph.path().is_file());
    let coordinates = dir.child("coordinates.md");
    assert!(coordinates.path().is_file());
    let metadata = dir.child("metadata.json");
    assert!(metadata.path().is_file());
    Ok(())
}

/// Reproducible run with explicit flags, like tuning a dataset for experiments.
#[test]
fn custom_size_count_and_seed() -> Result<(), Box<dyn std::error::Error>> {
    let dir = assert_fs::TempDir::new()?;
    let output = dir.child("questions.jsonl");

    cargo_bin_cmd!("dataset")
        .args([
            "--size",
            "2",
            "--count",
            "3",
            "--seed",
            "42",
            "--coordinate",
        ])
        .arg("-o")
        .arg(output.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Wrote 3 questions to"));

    let contents = std::fs::read_to_string(output.path())?;
    assert_eq!(contents.lines().count(), 3);
    assert!(contents.contains(r#""coordinates""#));
    assert!(!contents.contains(r#""graph""#));

    let graph = dir.child("graph.json");
    assert!(!graph.path().exists());
    let coordinates = dir.child("coordinates.md");
    assert!(coordinates.path().is_file());
    Ok(())
}

/// Graph-only questions for models that consume node labels instead of coordinates.
#[test]
fn graph_only_output() -> Result<(), Box<dyn std::error::Error>> {
    let dir = assert_fs::TempDir::new()?;
    let output = dir.child("graph-questions.jsonl");

    cargo_bin_cmd!("dataset")
        .args(["--size", "1", "--count", "1", "--seed", "7", "--graph"])
        .arg("-o")
        .arg(output.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Wrote 1 questions to"));

    let contents = std::fs::read_to_string(output.path())?;
    assert_eq!(contents.lines().count(), 1);
    assert!(contents.contains(r#""graph""#));
    assert!(!contents.contains(r#""coordinates""#));

    let graph = dir.child("graph.json");
    assert!(graph.path().is_file());
    let coordinates = dir.child("coordinates.md");
    assert!(!coordinates.path().exists());
    Ok(())
}

#[test]
fn invalid_space_size_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    cargo_bin_cmd!("dataset")
        .args(["--size", "0", "--count", "1"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("space size must be from 1 to"));

    Ok(())
}

#[test]
fn count_above_available_pairs_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    cargo_bin_cmd!("dataset")
        .args(["--size", "1", "--count", "31"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cannot sample 31 questions from 30 ordered pairs",
        ));

    Ok(())
}

#[test]
fn missing_output_directory_is_created() -> Result<(), Box<dyn std::error::Error>> {
    let dir = assert_fs::TempDir::new()?;
    let output = dir.child("nested/out/questions.jsonl");

    cargo_bin_cmd!("dataset")
        .args(["--size", "1", "--count", "1", "--seed", "0", "--coordinate"])
        .arg("-o")
        .arg(output.path())
        .assert()
        .success();

    assert!(output.path().is_file());
    assert!(dir.child("nested/out/coordinates.md").path().is_file());
    Ok(())
}

#[test]
fn help_lists_main_options() -> Result<(), Box<dyn std::error::Error>> {
    cargo_bin_cmd!("dataset")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Generate a dataset of spatial intelligence questions",
        ))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--size"))
        .stdout(predicate::str::contains("--coordinate"))
        .stdout(predicate::str::contains("--graph"));

    Ok(())
}
