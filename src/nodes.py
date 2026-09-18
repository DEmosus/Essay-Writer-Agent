"""
Node functions for the Essay Writer graph.

Each function takes the current ``AgentState`` and returns a partial dict of
the fields it updates — this is the contract LangGraph expects from every
node. The chat model and the Tavily client are built lazily via small
``get_*`` factories (cached with ``lru_cache``) rather than at import time,
so importing this module — or unit-testing ``should_continue`` in graph.py —
never requires API keys to be present. The first node that actually needs
the model or the search client is what triggers construction.
"""

import os
from functools import lru_cache
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tavily import TavilyClient  # pyright: ignore[reportMissingTypeStubs]

from .prompts import (
    PLAN_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,
    RESEARCH_PLAN_PROMPT,
    WRITER_PROMPT,
)
from .state import AgentState

# Configuration is read from the environment (populated from .env by main.py)
# so behaviour can be tuned without touching code.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "2"))


class Queries(BaseModel):
    """Structured-output schema the model must fill when planning searches."""

    queries: List[str]


@lru_cache(maxsize=1)
def get_model() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance, constructed on first use."""
    return ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)


@lru_cache(maxsize=1)
def get_tavily_client() -> TavilyClient:
    """Return a cached Tavily client, constructed on first use."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Copy .env.example to .env and add your "
            "Tavily key (https://app.tavily.com) before running the agent."
        )
    return TavilyClient(api_key=api_key)


def _run_research(queries: List[str], content: List[str]) -> List[str]:
    """Search Tavily for each query and append result text to ``content``.

    Shared by research_plan_node and research_critique_node, which differ
    only in *why* they're searching (initial plan vs. post-critique), not in
    how the search-and-collect step works.
    """
    tavily = get_tavily_client()
    for query in queries:
        response = tavily.search(query=query, max_results=TAVILY_MAX_RESULTS)
        for result in response["results"]:
            content.append(result["content"])
    return content


def plan_node(state: AgentState) -> dict:
    """Turn the raw task into a high-level essay outline."""
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state["task"]),
    ]
    response = get_model().invoke(messages)
    return {"plan": response.content}


def research_plan_node(state: AgentState) -> dict:
    """Generate search queries from the task and gather grounding research."""
    queries = (
        get_model()
        .with_structured_output(Queries)
        .invoke(
            [
                SystemMessage(content=RESEARCH_PLAN_PROMPT),
                HumanMessage(content=state["task"]),
            ]
        )
    )
    content = _run_research(queries.queries, state.get("content", []))
    return {"content": content}


def generation_node(state: AgentState) -> dict:
    """Write (or rewrite) the essay draft from the plan and research so far."""
    content = "\n\n".join(state.get("content", []))
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message,
    ]
    response = get_model().invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def reflection_node(state: AgentState) -> dict:
    """Critique the current draft the way a teacher grades a submission."""
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = get_model().invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState) -> dict:
    """Generate search queries from the critique and gather more research."""
    queries = (
        get_model()
        .with_structured_output(Queries)
        .invoke(
            [
                SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
                HumanMessage(content=state["critique"]),
            ]
        )
    )
    content = _run_research(queries.queries, state.get("content", []))
    return {"content": content}
