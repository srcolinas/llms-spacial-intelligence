from pathlib import Path

from solver.__main__ import default_output_path


def test_default_output_path() -> None:
    assert default_output_path("graph", "bare", "openai:gpt-4o") == Path(
        "data/answers.graph.bare.openai-gpt-4o.jsonl"
    )
    assert default_output_path("coordinates", "bare", "ollama:llama3.1:8b") == Path(
        "data/answers.coordinates.bare.ollama-llama3.1-8b.jsonl"
    )
