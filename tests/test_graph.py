"""
Tests for the Essay Writer graph.

These tests never call OpenAI or Tavily: `test_should_continue_*` exercise a
pure function, and `test_full_graph_run_with_mocks` monkeypatches
`src.nodes.get_model` / `get_tavily_client` with fakes so the whole graph
wiring (including the revise-until-budget-exhausted loop) can be verified
offline, in CI, with no API keys.

Run with:  python -m pytest tests/ -v
"""

from unittest.mock import MagicMock

import pytest

from src.graph import build_graph, should_continue
from src.nodes import Queries


def test_should_continue_stops_when_budget_exhausted():
    state = {"revision_number": 3, "max_revisions": 2}
    assert should_continue(state) == "__end__"


def test_should_continue_loops_when_budget_remains():
    state = {"revision_number": 1, "max_revisions": 2}
    assert should_continue(state) == "reflect"


def test_should_continue_boundary_is_inclusive():
    # revision_number == max_revisions should still continue (only strictly
    # greater than stops), matching the notebook's original semantics.
    state = {"revision_number": 2, "max_revisions": 2}
    assert should_continue(state) == "reflect"


@pytest.fixture
def fake_model():
    """A stand-in for ChatOpenAI that returns canned, deterministic output."""
    model = MagicMock()
    model.invoke.side_effect = [
        MagicMock(content="Outline: intro, body, conclusion."),  # planner
        MagicMock(content="Draft v1 of the essay."),  # generate (rev 1 -> 2)
        MagicMock(content="Needs more depth in paragraph 2."),  # reflect
        MagicMock(content="Draft v2, revised for depth."),  # generate (rev 2 -> 3)
    ]
    structured = MagicMock()
    structured.invoke.side_effect = [
        Queries(queries=["query about topic A"]),  # research_plan
        Queries(queries=["query about the critique"]),  # research_critique
    ]
    model.with_structured_output.return_value = structured
    return model


@pytest.fixture
def fake_tavily():
    client = MagicMock()
    client.search.return_value = {"results": [{"content": "Some grounding fact."}]}
    return client


def test_full_graph_run_with_mocks(monkeypatch, fake_model, fake_tavily):
    """End-to-end run: planner -> research -> generate -> reflect -> research -> generate -> END."""
    monkeypatch.setattr("src.nodes.get_model", lambda: fake_model)
    monkeypatch.setattr("src.nodes.get_tavily_client", lambda: fake_tavily)

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-thread"}}
    initial_state = {"task": "Test topic", "max_revisions": 2, "revision_number": 1}

    final_state = None
    for event in graph.stream(initial_state, config):
        final_state = event

    result = graph.get_state(config).values
    assert result["plan"] == "Outline: intro, body, conclusion."
    assert result["draft"] == "Draft v2, revised for depth."
    assert result["revision_number"] == 3  # stopped once > max_revisions (2)
    assert len(result["content"]) == 2  # one snippet from each research pass
    assert final_state is not None
