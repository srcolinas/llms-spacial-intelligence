from solver.agents import ValidationDeps
from solver.dataset import QuestionRecord
from spatial import validate_coordinate_path, validate_graph_path


def test_validate_coordinate_path_accepts_shortest_path() -> None:
    result = validate_coordinate_path(
        1,
        (0, 0, 0),
        (0, 0, 3),
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
    )
    assert result.valid
    assert result.length == 3


def test_validate_coordinate_path_rejects_bad_endpoints() -> None:
    result = validate_coordinate_path(
        1,
        (0, 0, 0),
        (0, 0, 3),
        [(0, 0, 0), (0, 0, 1), (0, 0, 2)],
    )
    assert not result.valid
    assert result.length == 2


def test_validate_graph_path_rejects_non_edge() -> None:
    valid = validate_graph_path(1, "A", "C", ["A", "B", "C"])
    assert valid.valid
    assert valid.length == 2

    invalid = validate_graph_path(1, "A", "C", ["A", "D", "C"])
    assert not invalid.valid


def test_validation_deps() -> None:
    record = QuestionRecord.model_validate_json(
        '{"id":0,"coordinates":{"origin":{"q":1,"r":0,"d":1},'
        '"destination":{"q":1,"r":-2,"d":1},'
        '"path":[]},"graph":{"origin":"A","destination":"B","path":[]}}'
    )
    deps = ValidationDeps(size=3, record=record)
    assert deps.size == 3
    assert deps.record.coordinates.origin.q == 1
