"""CLI entry point: multi-turn REPL or a single question.

    python -m agent.run                               # interactive, in-memory
    python -m agent.run --checkpointer sqlite         # conversation survives exit
    python -m agent.run "how do we chunk?"            # single turn
    python -m agent.run --trace "..."                 # include the per-turn trace

REPL commands: /history, /state, /checkpoints, /rewind <id>, /forget,
/metrics, /health, /new, /quit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

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
    parser.add_argument(
        "--checkpointer",
        choices=("memory", "sqlite"),
        default=None,
        help="conversation store; sqlite persists across runs",
    )
    parser.add_argument("--db", default=None, help="sqlite checkpoint file path")
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


def settings_from(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides: dict[str, object] = {}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.failure_rate is not None:
        overrides["simulated_failure_rate"] = args.failure_rate
    if args.checkpointer is not None:
        overrides["checkpoint_backend"] = args.checkpointer
    if args.db is not None:
        overrides["checkpoint_path"] = args.db
    return settings.model_copy(update=overrides) if overrides else settings


def render(result: dict, *, show_trace: bool) -> str:
    lines = [result["answer"] or "(no answer)"]
    meta = [f"status={result['status']}", f"turn={result['turn_index']}"]
    if result["citations"]:
        meta.append(f"citations={len(result['citations'])}")
    if result["summary"]:
        meta.append("summarised")
    lines.append("")
    lines.append(f"[{', '.join(meta)}]")
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


async def handle_command(agent: ConversationAgent, line: str, thread: str) -> str | None:
    """Run a REPL command. Returns the new thread id, or None to keep it."""
    command, _, argument = line.partition(" ")
    argument = argument.strip()

    if command == "/history":
        for entry in await agent.history(thread_id=thread):
            print(f"  {entry['role']:>6}: {entry['content'][:140]}")
    elif command == "/state":
        print(json.dumps(await agent.state(thread_id=thread), indent=2, default=str))
    elif command == "/checkpoints":
        for entry in await agent.checkpoints(thread_id=thread):
            print(
                f"  {entry['checkpoint_id']}  turn={entry['turn_index']:<3}"
                f" status={entry['status']:<20} next={entry['next_nodes']}"
            )
    elif command == "/rewind":
        if not argument:
            print("  usage: /rewind <checkpoint_id> <question>", file=sys.stderr)
        else:
            checkpoint_id, _, question = argument.partition(" ")
            if not question.strip():
                print("  usage: /rewind <checkpoint_id> <question>", file=sys.stderr)
            else:
                result = await agent.fork(
                    question.strip(), thread_id=thread, checkpoint_id=checkpoint_id
                )
                print(render(result, show_trace=False))
    elif command == "/forget":
        await agent.delete_thread(thread_id=thread)
        print(f"  deleted thread {thread}")
    elif command == "/metrics":
        print(json.dumps(agent.container.metrics.snapshot(), indent=2))
    elif command == "/health":
        print(json.dumps(await agent.container.health_report(), indent=2))
    elif command == "/new":
        new_thread = argument or f"thread-{uuid.uuid4().hex[:8]}"
        print(f"  started thread {new_thread}")
        return new_thread
    else:
        print(f"  unknown command: {command}", file=sys.stderr)
    return None


async def repl(agent: ConversationAgent, args) -> int:
    backend = agent.container.settings.checkpoint_backend
    print(
        f"Agentic RAG skeleton. checkpointer={backend} thread={args.thread}\n"
        "/quit to exit. /history /state /checkpoints /rewind /forget /metrics /health /new"
    )
    thread = args.thread
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return EXIT_OK
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return EXIT_OK
        if line.startswith("/"):
            try:
                moved = await handle_command(agent, line, thread)
            except AgentError as exc:
                print(f"  error [{exc.code}]: {exc.message}", file=sys.stderr)
                continue
            thread = moved or thread
            continue

        try:
            result = await agent.ask(line, thread_id=thread)
        except AgentError as exc:
            # The REPL survives a failed turn; thread state is left as it was.
            print(f"  error [{exc.code}]: {exc.message}", file=sys.stderr)
            continue
        print(render(result, show_trace=args.trace))


async def main_async(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = settings_from(args)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    question = " ".join(args.question).strip()
    try:
        async with ConversationAgent.session(build_container(settings)) as agent:
            if question:
                return await run_once(agent, question, args)
            return await repl(agent, args)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
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
