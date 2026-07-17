"""Command-line entry point.

Two modes:
  - One-shot: `python main.py "Investigate why AHU-01 ..."` — runs a single
    investigation and prints the reasoning log, timings, token usage, and
    final answer.
  - Interactive: `python main.py --interactive` — a REPL that keeps a
    conversation_id across turns, so follow-up requests ("what about its
    alarms?") get the benefit of conversation memory.
"""
from __future__ import annotations

import argparse
import json
import sys

from agenticos.agents.graph import build_graph
from agenticos.agents.state import new_initial_state
from agenticos.logging_config import configure_logging, get_logger
from agenticos.memory.store import ConversationStore, new_conversation_id
from agenticos.settings import get_settings

logger = get_logger("cli")


def _print_result(result: dict) -> None:
    print("=== Reasoning Log ===")
    for entry in result["logs"]:
        print(entry)

    print("\n=== Timings ===")
    print(json.dumps(result["timings"], indent=2))

    print("\n=== Token Usage ===")
    print(json.dumps(result["token_usage"], indent=2))

    print("\n=== Final Answer ===")
    print(result["final_answer"])


def _run_once(app, request: str, conversation_id: str, memory_store: ConversationStore | None, settings) -> dict:
    history = []
    if memory_store and settings.memory_enabled and conversation_id:
        history = memory_store.get_recent_turns(conversation_id, limit=settings.memory_max_turns)
    return app.invoke(new_initial_state(request, conversation_id=conversation_id, history=history))


def _interactive(app, memory_store: ConversationStore | None, settings) -> None:
    conversation_id = new_conversation_id()
    print(f"AgenticOS interactive mode. conversation_id={conversation_id}")
    print("Type a request, or 'exit'/'quit' to leave.\n")
    while True:
        try:
            request = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not request:
            continue
        if request.lower() in {"exit", "quit"}:
            break

        result = _run_once(app, request, conversation_id, memory_store, settings)
        print(f"\n{result['final_answer']}\n")


def main() -> None:
    configure_logging()
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run an AgenticOS investigation from the command line.")
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="The investigation request to submit. Omit with --interactive for a multi-turn REPL.",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Reuse conversation memory from a prior run instead of starting a fresh conversation.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start a multi-turn REPL that keeps conversation memory across requests.",
    )
    args = parser.parse_args()

    try:
        app = build_graph(settings=settings)
    except Exception:
        logger.exception("failed to build the investigation graph")
        print(
            "Could not start AgenticOS — is Ollama running and reachable at "
            f"{settings.ollama_base_url}? See README.md for setup.",
            file=sys.stderr,
        )
        sys.exit(1)

    memory_store = None
    if settings.memory_enabled:
        try:
            memory_store = ConversationStore(settings.resolved_memory_db_path)
        except Exception:
            logger.warning("conversation memory unavailable; continuing without it")

    if args.interactive:
        _interactive(app, memory_store, settings)
        return

    request = args.request or "Investigate why AHU-01 is consuming more energy today."
    conversation_id = args.conversation_id or ""
    result = _run_once(app, request, conversation_id, memory_store, settings)
    _print_result(result)


if __name__ == "__main__":
    sys.exit(main())
