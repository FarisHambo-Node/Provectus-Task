"""CLI entry point: multi-turn REPL or a single question.

    python -m agent.run                       # interactive, one thread
    python -m agent.run "how do we chunk?"    # single turn
    python -m agent.run --trace "..."         # include the per-turn trace

Commands inside the REPL: /history, /metrics, /health, /new, /quit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agent.container import build_container
from agent.errors import AgentError, ConfigurationError
from agent.graph import ConversationAgent
from agent.settings import Settings

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_RUNTIME = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent.run", description="Agentic RAG demo")
    parser.add_argument("question", nargs="*", help="ask one question and exit")
    parser.add_argument("--thread", default="local", help="conversation thread id")
    parser.add_argument("--trace", action="store_true", help="print the turn trace")
    parser.add_argument("--json", action="store_true", help="print the raw result")
    parser.add_argument("--seed", type=int, default=None, help="seed the simulation")
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=None,
        help="inject simulated tool failures, 0.0-1.0",
    )
    return parser.parse_args(argv)


def render(result: dict, *, show_trace: bool) -> str:
    lines = [result["answer"] or "(no answer)"]
    if result["citations"]:
        lines.append("")
        lines.append(f"[{len(result['citations'])} citations, status={result['status']}]")
    if result["failures"]:
        codes = ", ".join(str(f.get("code")) for f in result["failures"])
        lines.append(f"[degraded: {codes}]")
    if show_trace:
        lines.append("")
        lines.append("trace:")
        lines += [f"  {entry}" for entry in result["trace"]]
    return "\n".join(lines)


async def run_once(agent: ConversationAgent, question: str, args) -> int:
    result = await agent.ask(question, thread_id=args.thread)
    print(json.dumps(result, indent=2) if args.json else render(result, show_trace=args.trace))
    return EXIT_OK


async def repl(agent: ConversationAgent, args) -> int:
    print("Agentic RAG skeleton. /quit to exit, /history, /metrics, /health, /new.")
    thread = args.thread
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return EXIT_OK
        if not question:
            continue
        if question in ("/quit", "/exit"):
            return EXIT_OK
        if question == "/history":
            for entry in await agent.history(thread_id=thread):
                print(f"  {entry['role']:>6}: {entry['content'][:140]}")
            continue
        if question == "/metrics":
            print(json.dumps(agent.container.metrics.snapshot(), indent=2))
            continue
        if question == "/health":
            print(json.dumps(await agent.container.health_report(), indent=2))
            continue
        if question == "/new":
            thread = f"{args.thread}-{id(question)}"
            print(f"  started thread {thread}")
            continue

        try:
            result = await agent.ask(question, thread_id=thread)
        except AgentError as exc:
            # The REPL survives a failed turn; the thread state is unchanged.
            print(f"  error [{exc.code}]: {exc.message}", file=sys.stderr)
            continue
        print(render(result, show_trace=args.trace))


async def main_async(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    overrides = {}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.failure_rate is not None:
        overrides["simulated_failure_rate"] = args.failure_rate
    if overrides:
        settings = settings.model_copy(update=overrides)

    agent = ConversationAgent(build_container(settings))
    question = " ".join(args.question).strip()
    try:
        if question:
            return await run_once(agent, question, args)
        return await repl(agent, args)
    except AgentError as exc:
        print(f"turn failed [{exc.code}]: {exc.message}", file=sys.stderr)
        return EXIT_RUNTIME


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(main_async(argv))
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
