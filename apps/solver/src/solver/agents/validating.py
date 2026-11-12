"""Agent that validates its path output and retries on invalid paths."""

from __future__ import annotations

from dataclasses import dataclass

import pydantic_ai
from pydantic_ai import ModelRetry, RunContext
from spatial import validate_coordinate_path, validate_graph_path

from solver.agents.bare import _COORDINATES_INSTRUCTIONS, _GRAPH_INSTRUCTIONS
from solver.dataset import CoordinatePath, GraphPath, QuestionRecord

_PATH_RETRY_MESSAGE = (
    "That path is invalid. It must start at the origin, end at the "
    "destination, and move only along graph edges at each step."
)

_PATH_RETRIES = 10


@dataclass(frozen=True)
class ValidationDeps:
    size: int
    record: QuestionRecord


def build_validating_coordinates(
    model: str,
    space_description: str,
) -> pydantic_ai.Agent[ValidationDeps, CoordinatePath]:
    agent: pydantic_ai.Agent[ValidationDeps, CoordinatePath] = (
        pydantic_ai.Agent(
            model,
            deps_type=ValidationDeps,
            output_type=CoordinatePath,
            retries={"output": _PATH_RETRIES, "tools": 0},
            instructions=f"{_COORDINATES_INSTRUCTIONS}\n\n{space_description.rstrip()}",
        )
    )

    @agent.output_validator
    async def validate_output(
        ctx: RunContext[ValidationDeps],
        output: CoordinatePath,
    ) -> CoordinatePath:
        question = ctx.deps.record.coordinates
        result = validate_coordinate_path(
            ctx.deps.size,
            (question.origin.q, question.origin.r, question.origin.d),
            (
                question.destination.q,
                question.destination.r,
                question.destination.d,
            ),
            [(step.q, step.r, step.d) for step in output.path],
        )
        if not result.valid:
            raise ModelRetry(_PATH_RETRY_MESSAGE)
        return output

    return agent


def build_validating_graph(
    model: str,
    space_description: str,
) -> pydantic_ai.Agent[ValidationDeps, GraphPath]:
    agent: pydantic_ai.Agent[ValidationDeps, GraphPath] = pydantic_ai.Agent(
        model,
        deps_type=ValidationDeps,
        output_type=GraphPath,
        retries={"output": _PATH_RETRIES, "tools": 0},
        instructions=f"{_GRAPH_INSTRUCTIONS}\n\n{space_description.rstrip()}",
    )

    @agent.output_validator
    async def validate_output(
        ctx: RunContext[ValidationDeps],
        output: GraphPath,
    ) -> GraphPath:
        question = ctx.deps.record.graph
        result = validate_graph_path(
            ctx.deps.size,
            question.origin,
            question.destination,
            output.path,
        )
        if not result.valid:
            raise ModelRetry(_PATH_RETRY_MESSAGE)
        return output

    return agent
