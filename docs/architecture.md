# Architecture

## Overview

The Essay Writer is a **reflection agent**: instead of asking a language
model for an essay in one shot, it breaks the task into five specialized
steps — plan, research, write, critique, research again — and loops the
write/critique cycle until a revision budget is spent. Each step is a node
in a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph`,
and a single shared `AgentState` dict flows between them.

```
                         ┌────────────────────────────────────────────┐
                         │                                            │
                         ▼                                            │
 START → planner → research_plan → generate ──should_continue──► reflect
                                       │                              │
                                       │                              ▼
                                       │                   research_critique
                                       │                              │
                                       └──────────────◄───────────────┘
                                       (loop back into generate)
                                       │
                                       ▼ (once revision_number > max_revisions)
                                      END
```

## The state (`src/state.py`)

`AgentState` is a `TypedDict` with seven fields. Every node receives the
full state and returns a **partial dict** of just the fields it changes;
LangGraph merges that into the running state before the next node sees it.

| Field | Set by | Description |
|---|---|---|
| `task` | caller (CLI) | The essay topic, e.g. "Nvidia Blackwell AI chip". |
| `plan` | `planner` | A high-level outline the writer will follow. |
| `draft` | `generate` | The current essay text. Overwritten each revision. |
| `critique` | `reflect` | Feedback on the most recent draft. |
| `content` | `research_plan`, `research_critique` | Accumulated research snippets from Tavily. Grows across the whole run — later drafts see *all* research gathered so far, not just the latest batch. |
| `revision_number` | `generate` | Incremented every time a draft is produced. Starts at 1 (set by the caller), so after the first draft it becomes 2. |
| `max_revisions` | caller (CLI) | The stopping budget for `revision_number`. |

## The nodes (`src/nodes.py`)

1. **`plan_node`** — sends `task` to the model with `PLAN_PROMPT` and stores
   the resulting outline in `plan`.
2. **`research_plan_node`** — asks the model for up to 3 search queries
   (via structured output into the `Queries` Pydantic model), runs each
   through Tavily, and appends the result text to `content`.
3. **`generation_node`** — writes a 5-paragraph essay using `WRITER_PROMPT`,
   the `task`, the `plan`, and everything gathered in `content` so far.
   Increments `revision_number`.
4. **`reflection_node`** — grades the current `draft` like a teacher, using
   `REFLECTION_PROMPT`, and stores the result in `critique`.
5. **`research_critique_node`** — same pattern as step 2, but the search
   queries are generated from the `critique` instead of the original `task`,
   so the next draft can address specific gaps the critique identified.

All model and Tavily-client construction is **lazy** (`get_model()`,
`get_tavily_client()` in `nodes.py`, cached with `functools.lru_cache`).
Importing `src.nodes` — or unit-testing `should_continue` — never requires
`OPENAI_API_KEY` or `TAVILY_API_KEY` to be set; only actually calling a node
that needs them does. This is what lets `tests/test_graph.py` run the whole
graph offline with mocks.

## The control flow (`src/graph.py`)

`should_continue` is the only conditional edge in the graph. It is a pure
function of state — no model or network call — so it's trivially unit
tested:

```python
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
```

With `max_revisions=2`, the run goes: `generate` (rev 1→2, continue) →
`reflect` → `research_critique` → `generate` (rev 2→3, stop) → `END`. That
means **two drafts are produced**, and the second one benefits from the
critique of the first.

## Persistence: checkpointers

LangGraph checkpointers snapshot state after every node, keyed by a
`thread_id`. This project supports two, selected by the `CHECKPOINT_BACKEND`
env var:

- **`memory`** (default) — `MemorySaver`, in-process only. Simple, zero
  setup, state is gone when the process exits.
- **`sqlite`** — `SqliteSaver`, persisted to a local `.db` file. Because
  `SqliteSaver.from_conn_string(...)` is a context manager, `graph.py`
  exposes `checkpointer_context()` so `main.py` can keep the connection open
  for the lifetime of a run and reuse the same `thread_id` across separate
  CLI invocations to resume or inspect a run later.

## Why this shape?

- **Separation of concerns**: prompts, state, node logic, and graph wiring
  each live in their own module. Tuning a prompt never touches control flow;
  changing the loop condition never touches prompt text.
- **Self-correction over one-shot generation**: the critique step gives the
  agent a chance to catch shallow reasoning or missing depth before the
  final draft ships — the same reason human editors exist.
- **Grounded, not hallucinated**: both research nodes pull real search
  results into `content` before the writer touches the topic, and the
  critique loop re-researches specifically to fill gaps the critique
  surfaces, rather than asking the model to "try harder" from memory alone.
