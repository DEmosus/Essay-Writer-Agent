"""
Command-line entry point for the Essay Writer agent.

Usage:
    python -m src.main "The impact of remote work on urban housing markets"
    python -m src.main "Topic" --max-revisions 3 --thread-id my-essay-1

Run `python -m src.main --help` for all options.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .graph import build_graph, checkpointer_context

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "TAVILY_API_KEY"]

# Friendly labels for the raw node names streamed back by the graph.
NODE_LABELS = {
    "planner": "Planning the outline",
    "research_plan": "Researching the topic",
    "generate": "Writing a draft",
    "reflect": "Critiquing the draft",
    "research_critique": "Researching the critique",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="essay-writer",
        description="Generate a researched, self-critiqued essay on any topic.",
    )
    parser.add_argument("task", help="The essay topic or writing prompt.")
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=int(os.getenv("DEFAULT_MAX_REVISIONS", "2")),
        help="How many draft/critique cycles to run before stopping (default: 2).",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="LangGraph thread id, for resuming a run (default: a fresh UUID).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "outputs"),
        help="Directory to save the final essay + run log into (default: outputs/).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step progress output; only print the final essay.",
    )
    return parser.parse_args(argv)


def check_environment() -> None:
    """Fail fast with a clear message if required API keys are missing."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        print(
            "Missing required environment variable(s): " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Copy .env.example to .env and fill in your API keys, then try again.",
            file=sys.stderr,
        )
        sys.exit(1)


def run(args: argparse.Namespace) -> dict:
    """Run the graph to completion and return the final state."""
    thread_id = args.thread_id or f"essay-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "task": args.task,
        "max_revisions": args.max_revisions,
        "revision_number": 1,
    }

    with checkpointer_context() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        final_state: dict = {}
        for event in graph.stream(initial_state, config):
            for node_name, node_output in event.items():
                final_state.update(node_output)
                if not args.quiet:
                    label = NODE_LABELS.get(node_name, node_name)
                    print(f"  [{node_name}] {label}...")
        # Pull the full final state back out (stream only yields deltas).
        full_state = graph.get_state(config).values
    full_state["thread_id"] = thread_id
    return full_state


def save_output(state: dict, output_dir: str) -> Path:
    """Write the final essay (plus a short run log) to a markdown file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_task = "-".join(state["task"].lower().split())[:40]
    out_path = out_dir / f"{timestamp}_{safe_task}.md"

    lines = [
        f"# {state['task']}",
        "",
        f"*Generated {timestamp} — thread `{state.get('thread_id', 'n/a')}` — "
        f"{state.get('revision_number', 1) - 1} revision(s)*",
        "",
        "---",
        "",
        state.get("draft", "*(no draft produced)*"),
        "",
        "---",
        "",
        "## Final critique",
        "",
        state.get("critique", "*(no critique on record)*"),
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    check_environment()

    print(f'Writing essay on: "{args.task}"')
    print(f"Revision budget: {args.max_revisions}\n")

    try:
        final_state = run(args)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - surface any failure clearly to the CLI user
        print(f"\nThe agent failed: {exc}", file=sys.stderr)
        return 1

    out_path = save_output(final_state, args.output_dir)

    print("\n" + "=" * 72)
    print(final_state.get("draft", "(no draft produced)"))
    print("=" * 72)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
