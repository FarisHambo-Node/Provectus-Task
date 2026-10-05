"""Conversation memory: what the agent remembers and in what form.

A checkpoint grows without bound if you never trim it, and an unbounded
transcript eventually pushes the retrieved chunks out of the context window —
the failure shows up as the agent answering from history instead of evidence.

The policy here is a window plus a rolling summary:

* the last `history_window_turns` exchanges stay verbatim,
* everything older is folded into `summary`,
* the old messages are then dropped from the persisted state with
  `RemoveMessage`, so the checkpoint stops growing too.

All functions are pure so the policy can be tested without a graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AnyMessage, RemoveMessage

HUMAN = "human"
AI = "ai"


def message_role(message: AnyMessage) -> str:
    return str(getattr(message, "type", "unknown"))


def count_turns(messages: list[AnyMessage]) -> int:
    """A turn is one human message, answered or not."""
    return sum(1 for message in messages or [] if message_role(message) == HUMAN)


def window(messages: list[AnyMessage], *, window_turns: int) -> list[AnyMessage]:
    """Keep the last `window_turns` exchanges, cutting on a human message.

    Cutting at a human message keeps the transcript alternating; slicing by a
    raw message count can leave a dangling AI reply with no question.
    """
    if window_turns <= 0 or not messages:
        return []
    human_positions = [
        index for index, message in enumerate(messages) if message_role(message) == HUMAN
    ]
    if len(human_positions) <= window_turns:
        return list(messages)
    return list(messages[human_positions[-window_turns] :])


def should_summarize(
    messages: list[AnyMessage],
    *,
    summarize_after_turns: int,
) -> bool:
    return count_turns(messages) > summarize_after_turns


def fold_summary(
    previous: str,
    messages: list[AnyMessage],
    *,
    max_chars: int = 1200,
) -> str:
    """Fold older exchanges into the running summary.

    Deterministic and extractive: it records what was asked and the gist of
    each answer. TODO: replace with a Bedrock Converse call that compresses the
    same input, keeping this as the fallback when generation is unavailable.
    """
    lines = [previous.strip()] if previous.strip() else []
    for message in messages or []:
        role = message_role(message)
        text = " ".join(str(message.content).split())
        if not text:
            continue
        if role == HUMAN:
            lines.append(f"User asked: {_clip(text, 160)}")
        elif role == AI:
            lines.append(f"Agent answered: {_clip(_first_substantive_line(text), 200)}")
    folded = "\n".join(lines)
    if len(folded) <= max_chars:
        return folded
    # Drop from the front: the oldest context is the least useful.
    kept: list[str] = []
    budget = max_chars
    for line in reversed(folded.split("\n")):
        if len(line) + 1 > budget:
            break
        kept.append(line)
        budget -= len(line) + 1
    return "\n".join(reversed(kept))


def messages_to_drop(
    messages: list[AnyMessage],
    *,
    window_turns: int,
) -> tuple[list[AnyMessage], list[RemoveMessage]]:
    """Split the transcript into (to summarise, removal instructions).

    The second element goes straight into the `messages` channel: `add_messages`
    interprets `RemoveMessage` as a delete, which is how the checkpoint shrinks
    rather than just the prompt.
    """
    keep = window(messages, window_turns=window_turns)
    keep_ids = {id(message) for message in keep}
    older = [message for message in messages or [] if id(message) not in keep_ids]
    removals = [
        RemoveMessage(id=message.id) for message in older if getattr(message, "id", None)
    ]
    return older, removals


@dataclass(frozen=True)
class PromptContext:
    """Everything the generator is allowed to see, assembled in one place."""

    summary: str
    recent: list[AnyMessage]
    evidence: str

    def render(self) -> str:
        blocks: list[str] = []
        if self.summary:
            blocks.append(f"Earlier in this conversation:\n{self.summary}")
        if self.recent:
            transcript = "\n".join(
                f"{message_role(m)}: {_clip(' '.join(str(m.content).split()), 240)}"
                for m in self.recent
            )
            blocks.append(f"Recent turns:\n{transcript}")
        if self.evidence:
            blocks.append(f"Evidence:\n{self.evidence}")
        return "\n\n".join(blocks)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _first_substantive_line(text: str) -> str:
    """Skip our own 'Answer to: ...' preamble when summarising an AI reply."""
    for line in text.split("- "):
        cleaned = line.strip()
        if cleaned and not cleaned.lower().startswith("answer to:"):
            return cleaned
    return text
