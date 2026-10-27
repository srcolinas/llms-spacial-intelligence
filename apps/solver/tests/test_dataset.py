from pathlib import Path

import pytest
from solver.dataset import load_questions, load_size

_LINE = (
    '{"id":0,"coordinates":{"origin":{"q":1,"r":0,"d":1},"destination":{"q":1,"r":-2,"d":1},'
    '"path":[{"q":1,"r":0,"d":1},{"q":1,"r":-1,"d":2},{"q":1,"r":-1,"d":1},'
    '{"q":1,"r":-2,"d":2},{"q":1,"r":-2,"d":1}]},'
    '"graph":{"origin":"AR","destination":"AN","path":["AR","AQ","AP","AO","AN"]}}'
)


def test_load_fixture_builds_questions(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(f"\n{_LINE}\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        '{"version": 1, "space": {"size": 3}}\n',
        encoding="utf-8",
    )

    records = load_questions(dataset)
    assert load_size(dataset) == 3
    assert len(records) == 1
    record = records[0]
    assert record.id == 0
    assert record.coordinates is not None
    assert record.coordinates.origin.model_dump() == {"q": 1, "r": 0, "d": 1}
    assert record.coordinates.destination.model_dump() == {
        "q": 1,
        "r": -2,
        "d": 1,
    }
    assert record.graph is not None
    assert record.graph.origin == "AR"
    assert record.graph.destination == "AN"


def test_missing_questions_file_names_the_path(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    with pytest.raises(FileNotFoundError, match="questions file not found"):
        load_questions(dataset)


def test_missing_coordinates_field_fails_validation(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '{"id":0,"graph":{"origin":"A","destination":"B","path":["A","B"]}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid question on line 1"):
        load_questions(dataset)


def test_invalid_metadata_names_the_path(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        '{"version": 1}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid space metadata"):
        load_size(dataset)
