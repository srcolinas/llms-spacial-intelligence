"""Run one agent over a dataset representation and write JSONL answers."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TypedDict

import pydantic_ai
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from solver.agents import build_agent
from solver.dataset import (
    PROMPTS,
    ProblemKind,
    QuestionRecord,
    load_questions,
    load_size,
    render_space_description,
)
from solver.settings import settings

logger = logging.getLogger(__name__)


class AnswerRow(TypedDict):
    id: int
    prediction: list[dict[str, int]] | list[str] | None
    error: str | None


def _load_existing_answers(path: Path) -> dict[int, AnswerRow]:
    if not path.is_file():
        return {}
    answers: dict[int, AnswerRow] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid answer on line {line_number} of {path}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"invalid answer on line {line_number} of {path}: expected object"
            )
        if "id" not in row:
            raise ValueError(
                f"invalid answer on line {line_number} of {path}: missing id"
            )
        answers[int(row["id"])] = {
            "id": int(row["id"]),
            "prediction": row.get("prediction"),
            "error": row.get("error"),
        }
    return answers


def _needs_retry(row: AnswerRow) -> bool:
    return row["prediction"] is None or row["error"] is not None


async def run(
    *,
    llm: str,
    agent_name: str,
    problem: ProblemKind,
    dataset: Path,
    output: Path,
    limit: int | None,
    concurrency: int | None = None,
) -> tuple[int, int]:
    """Write one answer per question. Return rows written and rows that failed."""
    loaded = settings()
    _configure_logging(loaded.loglevel)
    if concurrency is None:
        concurrency = loaded.concurrency
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    size = load_size(dataset)
    questions = load_questions(dataset)
    if limit is not None:
        questions = questions[:limit]

    try:
        space_description = render_space_description(problem, size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        agent = build_agent(
            agent_name,
            problem,
            llm,
            space_description=space_description,
        )
    except pydantic_ai.exceptions.UserError as exc:
        raise SystemExit(str(exc)) from exc

    prompt = PROMPTS[problem]
    existing = _load_existing_answers(output)
    to_solve = [
        record
        for record in questions
        if record.id not in existing or _needs_retry(existing[record.id])
    ]
    skipped = len(questions) - len(to_solve)
    logger.info(
        "Solving %s %s questions with %s (%s)%s",
        len(to_solve),
        problem,
        agent_name,
        llm,
        f" ({skipped} skipped from existing answers)" if skipped else "",
    )

    semaphore = asyncio.Semaphore(concurrency)
    progress_desc = f"{agent_name}/{problem}"

    with logging_redirect_tqdm():
        pbar = tqdm(total=len(to_solve), desc=progress_desc, disable=None)

        async def solve_one(record: QuestionRecord) -> AnswerRow:
            async with semaphore:
                try:
                    prediction = (await agent.run(prompt(record))).output
                except Exception as exc:
                    logger.exception("question %s failed", record.id)
                    row: AnswerRow = {
                        "id": record.id,
                        "prediction": None,
                        "error": str(exc),
                    }
                else:
                    row = {
                        "id": record.id,
                        "prediction": prediction.model_dump()["path"],
                        "error": None,
                    }
            pbar.update(1)
            return row

        try:
            solved = await asyncio.gather(
                *(solve_one(record) for record in to_solve),
            )
        finally:
            pbar.close()
    answers = dict(existing)
    for row in solved:
        answers[row["id"]] = row

    rows = [answers[record.id] for record in questions]

    output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            if row["error"] is not None:
                failed += 1
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(questions), failed


class _SolverLogFilter(logging.Filter):
    """Keep this package's logs and drop every other library."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "solver" or record.name.startswith("solver.")


def _configure_logging(level_name: str) -> None:
    level = logging.getLevelNamesMapping()[level_name]
    handler = logging.StreamHandler()
    handler.addFilter(_SolverLogFilter())
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    logging.getLogger("solver").setLevel(level)
