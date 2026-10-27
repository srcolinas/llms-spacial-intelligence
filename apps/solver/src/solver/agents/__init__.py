"""Agents that answer one shortest-path question from a space description."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pydantic_ai

from solver.agents.bare import build_bare_coordinates, build_bare_graph
from solver.dataset import ProblemKind

AgentBuilder = Callable[[str, str], pydantic_ai.Agent[Any, Any]]

AGENTS: dict[tuple[str, ProblemKind], AgentBuilder] = {
    ("bare", "coordinates"): build_bare_coordinates,
    ("bare", "graph"): build_bare_graph,
}

AGENT_NAMES = sorted({name for name, _ in AGENTS})


def build_agent(
    agent_name: str,
    problem: ProblemKind,
    model: str,
    *,
    space_description: str,
) -> pydantic_ai.Agent[Any, Any]:
    key = (agent_name, problem)
    try:
        builder = AGENTS[key]
    except KeyError as exc:
        known = ", ".join(
            f"{name}/{problem}" for name, problem in sorted(AGENTS)
        )
        raise SystemExit(
            f"unknown agent {agent_name!r} for problem {problem!r}; "
            f"expected one of: {known}"
        ) from exc
    return builder(model, space_description)
