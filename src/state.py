"""
Shared state definition for the Essay Writer graph.

Every node in the graph receives the current ``AgentState`` and returns a
partial dict of the fields it updates. LangGraph merges that partial dict
back into the running state before handing it to the next node. This is the
single source of truth for what data moves through the graph, so it is kept
in its own module rather than buried inside graph.py.
"""

from typing import List, TypedDict


class AgentState(TypedDict):
    """State shared across every node in the essay-writing graph.

    Attributes:
        task: The user's essay topic/request, e.g. "Nvidia Blackwell AI chip".
        plan: The high-level outline produced by the planner node.
        draft: The current draft of the essay (overwritten on each revision).
        critique: The most recent critique produced by the reflection node.
        content: Accumulated research snippets pulled from Tavily, used as
            grounding context for both writing and revising the essay.
        revision_number: How many drafts have been generated so far(1-indexed).
        max_revisions: The revision budget; the graph stops once
            revision_number exceeds this value.
    """

    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int
