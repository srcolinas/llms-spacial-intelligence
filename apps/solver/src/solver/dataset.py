"""Load dataset questions and the description that goes with one representation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pydantic
from spatial import render_coordinate, render_graph

ProblemKind = Literal["coordinates", "graph"]


class Coordinate(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore", frozen=True)

    q: int
    r: int
    d: int


class CoordinateQuestion(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    origin: Coordinate
    destination: Coordinate
    path: list[Coordinate]


class GraphQuestion(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    origin: str
    destination: str
    path: list[str]


class QuestionRecord(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    id: int
    coordinates: CoordinateQuestion
    graph: GraphQuestion


class CoordinatePath(pydantic.BaseModel):
    """Shortest path using canonical `(q, r, d)` vertex names."""

    path: list[Coordinate]


class GraphPath(pydantic.BaseModel):
    """Shortest path using graph node labels."""

    path: list[str]


class _SpaceMetadata(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    size: int


class _Metadata(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")

    version: int
    space: _SpaceMetadata


def load_questions(path: Path) -> list[QuestionRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"questions file not found at {path}")
    records: list[QuestionRecord] = []
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(QuestionRecord.model_validate_json(line))
        except pydantic.ValidationError as exc:
            raise ValueError(
                f"invalid question on line {line_number} of {path}: {exc}"
            ) from exc
    return records


def load_size(dataset: Path) -> int:
    path = dataset.parent / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"space metadata not found at {path}")
    try:
        metadata = _Metadata.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except pydantic.ValidationError as exc:
        raise ValueError(f"invalid space metadata at {path}: {exc}") from exc
    return metadata.space.size


def render_space_description(problem: ProblemKind, size: int) -> str:
    if problem == "coordinates":
        return render_coordinate(size)
    return render_graph(size)


def _coordinates_prompt(question: CoordinateQuestion) -> str:
    origin = question.origin
    destination = question.destination
    return (
        "What is a shortest path from vertex "
        f"({origin.q}, {origin.r}, {origin.d}) to vertex "
        f"({destination.q}, {destination.r}, {destination.d})?"
    )


def _graph_prompt(question: GraphQuestion) -> str:
    return (
        "What is a shortest path from node "
        f"{question.origin} to node {question.destination}?"
    )


PromptFn = Callable[[QuestionRecord], str]

PROMPTS: dict[ProblemKind, PromptFn] = {
    "coordinates": lambda record: _coordinates_prompt(record.coordinates),
    "graph": lambda record: _graph_prompt(record.graph),
}
