"""
Essay Writer Agent
==================

A multi-step, self-reflecting essay-writing agent built with LangGraph.

The agent plans an essay, researches it, writes a draft, critiques its own
work, researches the critique, and rewrites — looping until it has produced
``max_revisions`` drafts. See ``docs/architecture.md`` for the full design.

Public API:
    build_graph() -> a compiled, runnable LangGraph graph.
"""

from .graph import build_graph
from .state import AgentState

__all__ = ["build_graph", "AgentState"]
__version__ = "1.0.0"
