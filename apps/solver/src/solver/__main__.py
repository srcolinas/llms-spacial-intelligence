from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pydantic

from solver.agents import AGENT_NAMES
from solver.dataset import ProblemKind
from solver.run import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve spatial shortest-path questions with an LLM agent",
    )
    parser.add_argument(
        "--llm",
        required=True,
        help="pydantic-ai model id in provider:model form, for example openai:gpt-4o",
    )
    parser.add_argument("--agent", required=True, choices=AGENT_NAMES)
    parser.add_argument(
        "--problem",
        required=True,
        choices=["coordinates", "graph"],
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/questions.jsonl"),
        help="questions JSONL; metadata.json is read from the same directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="answers JSONL (default: data/answers.<problem>.<agent>.<llm>.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="solve only the first N questions",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        help="maximum number of questions solved in parallel (default: SOLVER_CONCURRENCY or 10)",
    )
    args = parser.parse_args()
    problem: ProblemKind = args.problem
    output = args.output
    if output is None:
        output = default_output_path(problem, args.agent, args.llm)

    _langfuse()
    try:
        written, failed = asyncio.run(
            run(
                llm=args.llm,
                agent_name=args.agent,
                problem=problem,
                dataset=args.dataset,
                output=output,
                limit=args.limit,
                concurrency=args.concurrency,
            )
        )
    except (FileNotFoundError, ValueError, pydantic.ValidationError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {written} {problem} answers to {output}")
    if failed:
        print(f"{failed} questions could not be answered")


def default_output_path(problem: ProblemKind, agent: str, llm: str) -> Path:
    """Build the default answers path: answers.<problem>.<agent>.<llm>.jsonl."""
    safe_llm = llm.replace(":", "-").replace("/", "-")
    return Path(f"data/answers.{problem}.{agent}.{safe_llm}.jsonl")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _langfuse():
    import langfuse
    import pydantic_ai

    observer = langfuse.Langfuse()
    pydantic_ai.Agent.instrument_all()
    return observer

if __name__ == "__main__":
    main()
