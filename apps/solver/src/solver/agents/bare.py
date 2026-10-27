"""Bare approach: answer from the space description alone."""

from __future__ import annotations

import pydantic_ai

from solver.dataset import CoordinatePath, GraphPath

_INSTRUCTIONS = (
    "You solve a shortest-path question. "
    "The space description above defines the space. "
    "Each user message asks for a path from an origin to a destination. "
    "Return one shortest path, including the origin and the destination. "
    "Each step must be a neighbor of the previous step. "
    "Use the names from the description."
)


def build_bare_coordinates(
    model: str,
    space_description: str,
) -> pydantic_ai.Agent[None, CoordinatePath]:
    return pydantic_ai.Agent(
        model,
        output_type=CoordinatePath,
        instructions=f"{_INSTRUCTIONS}\n\n{space_description.rstrip()}",
    )


def build_bare_graph(
    model: str,
    space_description: str,
) -> pydantic_ai.Agent[None, GraphPath]:
    return pydantic_ai.Agent(
        model,
        output_type=GraphPath,
        instructions=f"{_INSTRUCTIONS}\n\n{space_description.rstrip()}",
    )
