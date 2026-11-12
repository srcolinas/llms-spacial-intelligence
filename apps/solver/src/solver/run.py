"""Run one agent over a dataset representation and write JSONL answers."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import NotRequired, TypedDict

import pydantic_ai
from pydantic_ai.messages import ModelRequest, RetryPromptPart
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from langfuse import propagate_attributes
from solver.agents import ValidationDeps, build_agent
from solver.agents.validating import _PATH_RETRIES
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
    mistakes: NotRequired[int | None]


def meta_path(output: Path) -> Path:
    """Companion metadata file for an answers JSONL path."""
    return output.with_suffix(".meta.json")


def _load_attempts(path: Path) -> int:
    meta = meta_path(path)
    if not meta.is_file():
        return 0
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metadata at {meta}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"invalid metadata at {meta}: expected object")
    attempts = data.get("attempts", 0)
    if not isinstance(attempts, int) or attempts < 0:
        raise ValueError(
            f"invalid metadata at {meta}: attempts must be a non-negative integer"
        )
    return attempts


def _save_attempts(path: Path, attempts: int) -> None:
    meta = meta_path(path)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        json.dumps({"attempts": attempts}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _count_output_mistakes(response: pydantic_ai.AgentRunResult[object]) -> int:
    return sum(
        1
        for message in response.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart) and part.tool_name is None
    )


def _serialize_answer_row(row: AnswerRow) -> str:
    payload: dict[str, object] = {
        "id": row["id"],
        "prediction": row["prediction"],
        "error": row["error"],
    }
    if "mistakes" in row and row["mistakes"] is not None:
        payload["mistakes"] = row["mistakes"]
    return json.dumps(payload, separators=(",", ":"))


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
        loaded: AnswerRow = {
            "id": int(row["id"]),
            "prediction": row.get("prediction"),
            "error": row.get("error"),
        }
        if "mistakes" in row:
            loaded["mistakes"] = row["mistakes"]
        answers[int(row["id"])] = loaded
    return answers


def _needs_retry(row: AnswerRow) -> bool:
    return row["prediction"] is None or row["error"] is not None


def _append_answer(path: Path, row: AnswerRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(_serialize_answer_row(row) + "\n")


def _remove_answer_ids(path: Path, ids: set[int]) -> None:
    if not path.is_file() or not ids:
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["id"]) not in ids:
            kept.append(line)
    with path.open("w", encoding="utf-8") as file:
        if kept:
            file.write("\n".join(kept) + "\n")


async def run(
    *,
    llm: str,
    agent_name: str,
    problem: ProblemKind,
    dataset: Path,
    output: Path,
    limit: int | None,
    concurrency: int | None = None,
) -> tuple[int, int, int]:
    """Write one answer per question.

    Return total rows, failed rows, and the attempt number for this run.
    """
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
    attempts = _load_attempts(output) + 1
    _save_attempts(output, attempts)
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

    answers: dict[int, AnswerRow] = dict(existing)
    retry_ids = {record.id for record in to_solve if record.id in existing}
    if retry_ids:
        _remove_answer_ids(output, retry_ids)
        for question_id in retry_ids:
            answers.pop(question_id, None)

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    progress_desc = f"{agent_name}/{problem}"

    def _with_mistakes(row: AnswerRow, mistakes: int | None) -> AnswerRow:
        if agent_name != "validating" or mistakes is None:
            return row
        return {**row, "mistakes": mistakes}

    async def solve_one(record: QuestionRecord) -> AnswerRow:
        async with semaphore:
            session_id = f"{problem}-question-{record.id}-{uuid.uuid4()}"
            row: AnswerRow
            mistakes: int | None = None
            try:
                with propagate_attributes(
                    session_id=session_id,
                    metadata={
                        "question_id": record.id,
                        "problem": problem,
                        "agent": agent_name,
                        "llm": llm,
                    },
                    tags=[agent_name, problem],
                ):
                    if agent_name == "validating":
                        response = await agent.run(
                            prompt(record),
                            deps=ValidationDeps(size=size, record=record),
                        )
                    else:
                        response = await agent.run(prompt(record))
                    prediction = response.output
                    if agent_name == "validating":
                        mistakes = _count_output_mistakes(response)
            except pydantic_ai.exceptions.UnexpectedModelBehavior as exc:
                logger.exception(
                    "question %s failed after path retries", record.id
                )
                if agent_name == "validating":
                    mistakes = _PATH_RETRIES
                row = _with_mistakes(
                    {
                        "id": record.id,
                        "prediction": None,
                        "error": str(exc),
                    },
                    mistakes,
                )
            except Exception as exc:
                logger.exception("question %s failed", record.id)
                row = _with_mistakes(
                    {
                        "id": record.id,
                        "prediction": None,
                        "error": str(exc),
                    },
                    mistakes,
                )
            else:
                row = _with_mistakes(
                    {
                        "id": record.id,
                        "prediction": prediction.model_dump()["path"],
                        "error": None,
                    },
                    mistakes,
                )
        return row

    with logging_redirect_tqdm():
        pbar = tqdm(total=len(to_solve), desc=progress_desc, disable=None)
        try:
            tasks = [
                asyncio.create_task(solve_one(record)) for record in to_solve
            ]
            for task in asyncio.as_completed(tasks):
                row = await task
                async with write_lock:
                    _append_answer(output, row)
                answers[row["id"]] = row
                pbar.update(1)
        finally:
            pbar.close()

    failed = sum(
        1 for record in questions if answers[record.id]["error"] is not None
    )
    return len(questions), failed, attempts


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
