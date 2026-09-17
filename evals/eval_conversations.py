"""Multi-turn evaluation: run each golden conversation in its own thread.

Run from the project root:
    python -m evals.eval_conversations --limit 1   # one conversation first, to measure token cost
    python -m evals.eval_conversations

Only the last turn of each conversation is graded; earlier turns build the context.
Every turn's answer, tool calls, and token usage are saved to evals/results/, and
the file can be graded with: python -m evals.judge_answers PATH
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config import EMBEDDING_MODEL, LLM_MODEL, VECTOR_DISTANCE_THRESHOLD
from core.lc_agent import LangChainSearchAgent
from evals.eval_agent import RESULTS_DIR, grade_keywords

CONVERSATIONS = Path(__file__).parent / "golden_conversations.json"


def load_conversations() -> list[dict[str, Any]]:
    """Each conversation doubles as a golden case whose question is its last turn."""
    conversations = json.loads(CONVERSATIONS.read_text(encoding="utf-8"))
    return [{**conversation, "question": conversation["turns"][-1]} for conversation in conversations]


def run_turn(agent: Any, question: str, thread_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    turn: dict[str, Any] = {"question": question, "tool_calls": [], "error": None}
    for step in agent.search_stream(question, thread_id=thread_id):
        if step["step"] == "tool_call":
            turn["tool_calls"].append({"tool": step["tool_name"], "arguments": step["arguments"]})
        elif step["step"] == "api_error":
            turn["error"] = step["error"]
        elif step["step"] == "final":
            turn["answer"] = step["final_answer"]
            turn["rounds"] = step["round"]
            turn["usage"] = step.get("usage", {})
    turn["seconds"] = round(time.perf_counter() - started, 1)
    return turn


def run_conversation(agent: Any, conversation: dict[str, Any]) -> dict[str, Any]:
    # A new thread per conversation keeps conversations from seeing each other.
    thread_id = f"eval-{conversation['id']}-{uuid.uuid4().hex[:8]}"
    turns = [run_turn(agent, question, thread_id) for question in conversation["turns"]]
    final_answer = turns[-1].get("answer", "")
    return {
        "id": conversation["id"],
        "category": conversation["category"],
        "type": conversation["type"],
        "question": conversation["question"],
        "answer": final_answer,
        "turns": turns,
        **grade_keywords(conversation, final_answer),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [turn for record in records for turn in record["turns"]]
    follow_ups = [turn for record in records for turn in record["turns"][1:]]
    context_by_position: dict[int, list[int]] = defaultdict(list)
    for record in records:
        for position, turn in enumerate(record["turns"], start=1):
            context_by_position[position].append(turn.get("usage", {}).get("context_tokens", 0))

    def total(key: str) -> int:
        return sum(turn.get("usage", {}).get(key, 0) for turn in turns)

    return {
        "keyword_pass": f"{sum(r['passed'] for r in records)}/{len(records)}",
        "turns": len(turns),
        "turns_with_errors": sum(1 for turn in turns if turn["error"]),
        "model_calls": total("model_calls"),
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "tool_result_tokens": total("tool_result_tokens"),
        "avg_context_tokens_by_turn": {
            position: round(sum(values) / len(values)) for position, values in sorted(context_by_position.items())
        },
        "follow_ups_that_searched_again": f"{sum(1 for t in follow_ups if t['tool_calls'])}/{len(follow_ups)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn conversation evaluation.")
    parser.add_argument("--limit", type=int, help="run only the first N conversations")
    args = parser.parse_args()

    conversations = load_conversations()[: args.limit]
    agent = LangChainSearchAgent()
    records = []
    for index, conversation in enumerate(conversations, start=1):
        record = run_conversation(agent, conversation)
        records.append(record)
        turn_notes = " | ".join(
            f"t{n}: calls={t.get('usage', {}).get('model_calls', 0)} ctx={t.get('usage', {}).get('context_tokens', 0)}"
            + (" ERROR" if t["error"] else "")
            for n, t in enumerate(record["turns"], start=1)
        )
        status = "PASS" if record["passed"] else "FAIL"
        print(f"[{index}/{len(conversations)}] {status} {record['id']:34s} {turn_notes}", flush=True)

    summary = summarize(records)
    config = {
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "vector_distance_threshold": VECTOR_DISTANCE_THRESHOLD,
    }
    print("\nConfig:", json.dumps(config))
    for key, value in summary.items():
        print(f"  {key:30s} {value}")

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-conversations-threshold-{VECTOR_DISTANCE_THRESHOLD}.json"
    output.write_text(
        json.dumps({"config": config, "summary": summary, "cases": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved to {output}")
    print(f"Grade the final turns with: python -m evals.judge_answers {output}")


if __name__ == "__main__":
    main()
