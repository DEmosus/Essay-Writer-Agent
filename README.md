# Essay Writer Agent

<p align="center">
  <b>A self-reflecting, research-grounded autonomous essay-writing agent built with LangGraph, LangChain, OpenAI, and Tavily.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/LangGraph-Architecture-orange?style=flat-square" alt="LangGraph">
  <img src="https://img.shields.io/badge/tests-pytest--mocked-success?style=flat-square&logo=pytest&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
</p>

---

## Overview

Unlike standard LLM wrappers that generate essays in a single, unverified pass, **Essay Writer Agent** operates as a stateful multi-agent system. It plans an outline, grounds its content in live web searches, drafts the essay, critiques its own work like an expert editor, researches identified knowledge gaps, and iteratively rewrites until a polished, high-quality final piece is produced.

## How it works

```
 START → planner → research_plan → generate ──should_continue──► reflect
                                       │                              │
                                       │                              ▼
                                       │                   research_critique
                                       │                              │
                                       └──────────────◄───────────────┘
                                       │
                                       ▼ (once revision_number > max_revisions)
                                      END
```

Five nodes, one shared state object, one conditional edge deciding whether
to loop back for another revision or stop. Full write-up in
[`docs/architecture.md`](docs/architecture.md).

## Features

- **Multi-step reflection loop** — plan → research → write → critique →
  research → rewrite, with a configurable revision budget.
- **Grounded in live search** — every draft and every revision is backed by
  fresh [Tavily](https://tavily.com) search results, not just the model's
  training data.
- **Clean separation of concerns** — prompts, state, node logic, and graph
  wiring each live in their own module (`src/prompts.py`, `src/state.py`,
  `src/nodes.py`, `src/graph.py`).
- **Testable without API keys** — the model and search client are built
  lazily, so the graph's wiring can be (and is) unit-tested offline with
  mocks. See [`tests/test_graph.py`](tests/test_graph.py).
- **Two checkpoint backends** — fast in-memory by default, or persisted
  SQLite for resumable runs, switched with one env var.
- **A real CLI**, not just a notebook cell — argument parsing, a friendly
  error if API keys are missing, streamed progress, and a saved `.md`
  output file per run.

## Project structure

```
.
├── .env.example              # Template for your local .env
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt           # Top-level dependencies
├── requirements-lock.txt      # Exact pinned versions (reproducible installs)
├── .github/workflows/ci.yml   # Runs the offline test suite on every push
├── docs/
│   ├── architecture.md        # How the graph is put together, and why
│   ├── decisions.md           # ADR-style log of engineering decisions
│   └── development-log.md     # Project: what changed, and why
├── src/
│   ├── __init__.py            # Public API: build_graph(), AgentState
│   ├── state.py                # AgentState TypedDict
│   ├── prompts.py              # All system prompts, in one place
│   ├── nodes.py                 # The five node functions + Queries schema
│   ├── graph.py                  # Graph assembly + should_continue + checkpointers
│   └── main.py                    # CLI entry point
├── tests/
│   └── test_graph.py            # Offline tests (should_continue + a full mocked run)
└── outputs/                    # Generated essays land here (gitignored)
```

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Tavily API key](https://app.tavily.com) (free tier is enough to try this out)

## Installation

```bash
git clone git@github.com:DEmosus/Essay-Writer-Agent.git
cd essay-writer-agent

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-lock.txt   # exact versions this was built and tested against
# or: pip install -r requirements.txt  # looser floors, gets you the latest compatible versions

cp .env.example .env
# then open .env and fill in OPENAI_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
# Basic run — writes 2 drafts (1 initial + 1 revision) by default
python -m src.main "The impact of remote work on urban housing markets"

# Give it a bigger revision budget
python -m src.main "The ethics of AI-generated art" --max-revisions 3

# Persist checkpoints to disk and reuse a thread_id later
CHECKPOINT_BACKEND=sqlite python -m src.main "Kenya's renewable energy transition" --thread-id kenya-energy-v1

# Quiet mode — just the final essay, no step-by-step progress lines
python -m src.main "Topic" --quiet
```

Every run saves the final essay (plus the last critique) as a Markdown file
under `outputs/`, named by timestamp and topic.

Run `python -m src.main --help` for the full list of flags.

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable                | Default                        | Purpose                                            |
| ----------------------- | ------------------------------ | -------------------------------------------------- |
| `OPENAI_API_KEY`        | _(required)_                   | OpenAI API key                                     |
| `TAVILY_API_KEY`        | _(required)_                   | Tavily search API key                              |
| `OPENAI_MODEL`          | `gpt-4o-mini`                  | Chat model for planning, writing, critique         |
| `OPENAI_TEMPERATURE`    | `0`                            | Sampling temperature                               |
| `TAVILY_MAX_RESULTS`    | `2`                            | Search results fetched per query                   |
| `DEFAULT_MAX_REVISIONS` | `2`                            | Default `--max-revisions` if not passed on the CLI |
| `OUTPUT_DIR`            | `outputs`                      | Where finished essays are saved                    |
| `CHECKPOINT_BACKEND`    | `memory`                       | `memory` (ephemeral) or `sqlite` (persisted)       |
| `CHECKPOINT_DB_PATH`    | `.checkpoints/essay_writer.db` | Only used when `CHECKPOINT_BACKEND=sqlite`         |

## Running tests

```bash
pip install -r requirements-lock.txt   # includes pytest
python -m pytest tests/ -v
```

The tests never call OpenAI or Tavily — `should_continue` is tested as a
pure function, and a full graph run is exercised with the model and search
client mocked out, so the whole suite runs offline, in CI, with no API keys
and no cost. See `.github/workflows/ci.yml`.

## Using it as a library

`src/__init__.py` exposes `build_graph()` and `AgentState`, so the agent can
be embedded elsewhere (a web UI, an API endpoint) without going through the
CLI:

```python
from src import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "demo-1"}}
result = graph.invoke(
    {"task": "Why remote work reshaped mid-size cities", "max_revisions": 2, "revision_number": 1},
    config,
)
print(result["draft"])
```

## Roadmap ideas

- Swap the linear 5-paragraph `WRITER_PROMPT` for a configurable essay
  length/structure.
- Add a lightweight Streamlit front end on top of `build_graph()`.
- Human-in-the-loop approval between `reflect` and `research_critique`
  using LangGraph's `interrupt_after`.

## License

[MIT](LICENSE)

## Acknowledgments

The agent design (plan → research → generate → reflect → research-critique
loop) follows the pattern taught in DeepLearning.AI's _AI Agents in
LangGraph_ course. This repository is an original, from-scratch
reimplementation and restructuring of that pattern into a tested,
documented, standalone project.
