"""Bare approach: answer from the space description alone."""

from __future__ import annotations

import pydantic_ai

from solver.dataset import CoordinatePath, GraphPath

_COORDINATES_INSTRUCTIONS = (
    "You solve shortest-path problems on the vertex graph defined below. "
    "Each user message gives an origin and destination as canonical (q, r, d) vertices. "
    "Return one shortest path (minimum number of edges), including both endpoints. "
    "Each consecutive pair must be graph neighbors as defined in the Adjacency section. "
    "Express every step with integer q, r, and d fields in canonical form. "
    "Do not skip vertices or use non-adjacent steps."
)

_GRAPH_INSTRUCTIONS = (
    "You solve shortest-path problems on the graph defined below. "
    "Each user message asks for a path from an origin node to a destination node. "
    "Return one shortest path (minimum number of edges), including both endpoints. "
    "Each consecutive pair must be connected by an edge listed in the graph. "
    "Use the node labels from the description."
)


def build_bare_coordinates(
    model: str,
    space_description: str,
) -> pydantic_ai.Agent[None, CoordinatePath]:
    return pydantic_ai.Agent(
        model,
        output_type=CoordinatePath,
        instructions=f"{_COORDINATES_INSTRUCTIONS}\n\n{space_description.rstrip()}",
    )


def build_bare_graph(
    model: str,
    space_description: str,
) -> pydantic_ai.Agent[None, GraphPath]:
    return pydantic_ai.Agent(
        model,
        output_type=GraphPath,
        instructions=f"{_GRAPH_INSTRUCTIONS}\n\n{space_description.rstrip()}",
    )
