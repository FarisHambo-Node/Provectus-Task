"""Conversation memory policy: window, summarisation, and message removal."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from agent.memory import (
    PromptContext,
    count_turns,
    fold_summary,
    message_role,
    messages_to_drop,
    should_summarize,
    window,
)


def transcript(turns: int, *, trailing_human: bool = False) -> list:
    messages = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"question {i}", id=f"h{i}"))
        messages.append(AIMessage(content=f"answer {i}", id=f"a{i}"))
    if trailing_human:
        messages.append(HumanMessage(content="pending question", id="h-pending"))
    return messages


# --- counting and windowing -----------------------------------------------


def test_count_turns_counts_human_messages():
    assert count_turns(transcript(3)) == 3
    assert count_turns(transcript(3, trailing_human=True)) == 4
    assert count_turns([]) == 0


def test_window_keeps_the_last_n_exchanges():
    kept = window(transcript(5), window_turns=2)
    assert count_turns(kept) == 2
    assert kept[0].content == "question 3"


def test_window_cuts_on_a_human_message():
    # A raw message slice could start on an AI reply with no question above it.
    kept = window(transcript(5), window_turns=3)
    assert message_role(kept[0]) == "human"


def test_window_keeps_a_pending_question():
    kept = window(transcript(3, trailing_human=True), window_turns=1)
    assert kept[-1].content == "pending question"
    assert count_turns(kept) == 1


def test_window_returns_everything_when_the_thread_is_short():
    messages = transcript(2)
    assert window(messages, window_turns=6) == messages


def test_window_edge_cases():
    assert window([], window_turns=3) == []
    assert window(transcript(2), window_turns=0) == []


# --- summarisation trigger -------------------------------------------------


def test_summarize_only_once_past_the_threshold():
    assert should_summarize(transcript(3), summarize_after_turns=3) is False
    assert should_summarize(transcript(4), summarize_after_turns=3) is True


# --- folding ---------------------------------------------------------------


def test_fold_summary_records_questions_and_answers():
    summary = fold_summary("", transcript(2))
    assert "User asked: question 0" in summary
    assert "Agent answered: answer 0" in summary


def test_fold_summary_carries_the_previous_summary_forward():
    summary = fold_summary("User asked: an older thing", transcript(1))
    assert "an older thing" in summary
    assert "question 0" in summary


def test_fold_summary_respects_the_character_budget():
    summary = fold_summary("", transcript(60), max_chars=400)
    assert len(summary) <= 400
    # The budget is spent on the most recent context, not the oldest.
    assert "question 59" in summary


def test_fold_summary_skips_blank_messages():
    assert fold_summary("", [HumanMessage(content="   ", id="h")]) == ""


def test_fold_summary_skips_our_own_answer_preamble():
    answer = AIMessage(content="Answer to: q\n\n- The real content here.", id="a")
    summary = fold_summary("", [answer])
    assert "Answer to:" not in summary
    assert "The real content here." in summary


# --- removal instructions --------------------------------------------------


def test_messages_to_drop_splits_at_the_window():
    older, removals = messages_to_drop(transcript(5), window_turns=2)
    assert count_turns(older) == 3
    assert all(isinstance(removal, RemoveMessage) for removal in removals)
    assert {removal.id for removal in removals} == {m.id for m in older}


def test_nothing_is_dropped_from_a_short_thread():
    older, removals = messages_to_drop(transcript(2), window_turns=6)
    assert older == []
    assert removals == []


def test_messages_without_ids_cannot_be_removed():
    # add_messages assigns ids, but never emit a RemoveMessage without one.
    messages = [HumanMessage(content="q"), AIMessage(content="a")] + transcript(2)
    _, removals = messages_to_drop(messages, window_turns=1)
    assert all(removal.id for removal in removals)


# --- prompt assembly -------------------------------------------------------


def test_prompt_context_renders_all_three_blocks():
    rendered = PromptContext(
        summary="User asked about chunking.",
        recent=transcript(1),
        evidence="[1] chunk size is 1000",
    ).render()
    assert "Earlier in this conversation:" in rendered
    assert "Recent turns:" in rendered
    assert "Evidence:" in rendered


def test_prompt_context_omits_empty_blocks():
    rendered = PromptContext(summary="", recent=[], evidence="[1] x").render()
    assert "Earlier in this conversation:" not in rendered
    assert rendered.startswith("Evidence:")
