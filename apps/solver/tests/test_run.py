import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from solver.run import run

_LINE = (
    '{"id":0,"coordinates":{"origin":{"q":1,"r":0,"d":1},"destination":{"q":1,"r":-2,"d":1},'
    '"path":[{"q":1,"r":0,"d":1},{"q":1,"r":-1,"d":2},{"q":1,"r":-1,"d":1},'
    '{"q":1,"r":-2,"d":2},{"q":1,"r":-2,"d":1}]},'
    '"graph":{"origin":"AR","destination":"AN","path":["AR","AQ","AP","AO","AN"]}}'
)


def test_run_records_the_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    dataset = _dataset(tmp_path, _LINE)
    output = tmp_path / "answers.jsonl"

    written, failed = asyncio.run(
        run(
            llm="test",
            agent_name="bare",
            problem="graph",
            dataset=dataset,
            output=output,
            limit=1,
        )
    )

    assert (written, failed) == (1, 0)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert set(row) == {"id", "prediction", "error"}
    assert row["id"] == 0
    assert isinstance(row["prediction"], list)
    assert row["error"] is None


def test_run_solves_questions_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    dataset = _dataset(
        tmp_path,
        _LINE,
        '{"id":1,"coordinates":{"origin":{"q":1,"r":0,"d":1},"destination":{"q":1,"r":-2,"d":1},'
        '"path":[{"q":1,"r":0,"d":1},{"q":1,"r":-1,"d":2},{"q":1,"r":-1,"d":1},'
        '{"q":1,"r":-2,"d":2},{"q":1,"r":-2,"d":1}]},'
        '"graph":{"origin":"AR","destination":"AN","path":["AR","AQ","AP","AO","AN"]}}',
    )
    output = tmp_path / "answers.jsonl"
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def slow_run(_prompt: str) -> object:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"path": ["AR", "AN"]}),
        )

    with patch("solver.run.build_agent") as build_agent:
        build_agent.return_value = SimpleNamespace(run=slow_run)

        written, failed = asyncio.run(
            run(
                llm="test",
                agent_name="bare",
                problem="graph",
                dataset=dataset,
                output=output,
                limit=None,
                concurrency=2,
            )
        )

    assert (written, failed) == (2, 0)
    assert max_in_flight == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [0, 1]


def test_run_retries_failed_and_null_predictions_from_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    dataset = _dataset(
        tmp_path,
        _LINE,
        '{"id":1,"coordinates":{"origin":{"q":1,"r":0,"d":1},"destination":{"q":1,"r":-2,"d":1},'
        '"path":[{"q":1,"r":0,"d":1},{"q":1,"r":-1,"d":2},{"q":1,"r":-1,"d":1},'
        '{"q":1,"r":-2,"d":2},{"q":1,"r":-2,"d":1}]},'
        '"graph":{"origin":"AR","destination":"AN","path":["AR","AQ","AP","AO","AN"]}}',
        '{"id":2,"coordinates":{"origin":{"q":1,"r":0,"d":1},"destination":{"q":1,"r":-2,"d":1},'
        '"path":[{"q":1,"r":0,"d":1},{"q":1,"r":-1,"d":2},{"q":1,"r":-1,"d":1},'
        '{"q":1,"r":-2,"d":2},{"q":1,"r":-2,"d":1}]},'
        '"graph":{"origin":"AR","destination":"AN","path":["AR","AQ","AP","AO","AN"]}}',
    )
    output = tmp_path / "answers.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 0,
                        "prediction": ["AR", "AN"],
                        "error": None,
                    },
                    separators=(",", ":"),
                ),
                json.dumps(
                    {
                        "id": 1,
                        "prediction": None,
                        "error": "first failure",
                    },
                    separators=(",", ":"),
                ),
                json.dumps(
                    {
                        "id": 2,
                        "prediction": None,
                        "error": None,
                    },
                    separators=(",", ":"),
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    async def track_run(_prompt: str) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("still failing")
        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"path": ["AR", "AQ", "AN"]}),
        )

    with patch("solver.run.build_agent") as build_agent:
        build_agent.return_value = SimpleNamespace(run=track_run)

        written, failed = asyncio.run(
            run(
                llm="test",
                agent_name="bare",
                problem="graph",
                dataset=dataset,
                output=output,
                limit=None,
                concurrency=1,
            )
        )

    assert (written, failed) == (3, 1)
    assert calls == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["prediction"] == ["AR", "AN"]
    assert rows[0]["error"] is None
    assert rows[1]["prediction"] is None
    assert rows[1]["error"] == "still failing"
    assert rows[2]["prediction"] == ["AR", "AQ", "AN"]
    assert rows[2]["error"] is None


def _dataset(directory: Path, *lines: str) -> Path:
    dataset = directory / "questions.jsonl"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "metadata.json").write_text(
        '{"version": 1, "space": {"size": 3}}\n',
        encoding="utf-8",
    )
    return dataset
