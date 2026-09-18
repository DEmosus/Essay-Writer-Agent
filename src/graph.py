"""
Graph assembly for the Essay Writer agent.

This module is deliberately "thin": it only wires nodes together into a
graph shape. All LLM/tool logic lives in nodes.py, and all prompt text lives
in prompts.py. That separation is what makes the graph easy to read at a
glance and easy to unit-test (should_continue below needs no API key at all).

Graph shape
-----------
    planner -> research_plan -> generate --(should_continue)--> END
                                    ^  \\
                                    |   -> reflect -> research_critique -+
                                    +------------------------------------+

The loop (generate -> reflect -> research_critique -> generate) repeats
until should_continue sees revision_number > max_revisions, at which point
it routes to END instead of back to reflect.
"""

import os
from contextlib import contextmanager
from typing import Iterator, Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    generation_node,
    plan_node,
    reflection_node,
    research_critique_node,
    research_plan_node,
)
from .state import AgentState


def should_continue(state: AgentState) -> Literal["reflect", "__end__"]:
    """Conditional edge out of 'generate': keep revising, or stop.

    Pure function of state — no model or network calls — so it can be
    (and is, in tests/test_graph.py) tested directly without any API keys.
    """
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


def build_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """Assemble and compile the essay-writer StateGraph.

    Args:
        checkpointer: Where LangGraph persists state between steps, enabling
            multi-turn resumption via a thread_id. Defaults to an in-memory
            MemorySaver if omitted. Pass a SqliteSaver (see
            checkpointer_context below) for persistence across process runs.
    """
    builder = StateGraph(AgentState)

    builder.add_node("planner", plan_node)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("research_plan", research_plan_node)
    builder.add_node("research_critique", research_critique_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "research_plan")
    builder.add_edge("research_plan", "generate")
    builder.add_conditional_edges(
        "generate",
        should_continue,
        {END: END, "reflect": "reflect"},
    )
    builder.add_edge("reflect", "research_critique")
    builder.add_edge("research_critique", "generate")

    return builder.compile(checkpointer=checkpointer or MemorySaver())


@contextmanager
def checkpointer_context() -> Iterator[BaseCheckpointSaver]:
    """Yield a checkpointer chosen by the CHECKPOINT_BACKEND env var.

    'memory' (default): fast, in-process, state is lost when the process
        exits — fine for one-off runs.
    'sqlite': persisted to CHECKPOINT_DB_PATH (default
        .checkpoints/essay_writer.db), so a run can be resumed by reusing
        the same thread_id in a later process. SqliteSaver is a context
        manager in langgraph-checkpoint-sqlite, so this helper is one too —
        use it with `with checkpointer_context() as checkpointer:`.
    """
    backend = os.getenv("CHECKPOINT_BACKEND", "memory").lower()
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = os.getenv("CHECKPOINT_DB_PATH", ".checkpoints/essay_writer.db")
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with SqliteSaver.from_conn_string(db_path) as saver:
            yield saver
    else:
        yield MemorySaver()
