"""End-to-end evaluation: run the full LangChain agent on every golden-set question.

Run from the project root:
    python -m evals.eval_agent

Each run calls the chat model several times per question, so it is slower and
costs more than eval_retrieval. Full answers are saved to evals/results/ so
failed cases can be inspected afterwards.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from config import EMBEDDING_MODEL, LLM_MODEL, VECTOR_DISTANCE_THRESHOLD
from core.lc_agent import LangChainSearchAgent
from evals.eval_retrieval import GOLDEN_SET, matches_keywords

RESULTS_DIR = Path(__file__).parent / "results"

# Phrases the model uses when it says the dataset has no answer.
# Extend this list after reading real answers in evals/results/.
REFUSAL_PHRASES = [
    "does not contain",
    "doesn't contain",
    "do not contain",
    "don't contain",
    "does not include",
    "doesn't include",
    "does not mention",
    "doesn't mention",
    "does not specify",
    "doesn't specify",
    "does not cover",
    "doesn't cover",
    "not mentioned",
    "not specified",
    "not covered",
    "not available",
    "no information",
    "no relevant",
    "could not find",
    "couldn't find",
    "unable to find",
    "don't have access",
    "do not have access",
]
# Deliberately not included: "outside the scope". One answer said
# "Argentina won the 2022 FIFA World Cup. (Note: This information is outside
# the scope of the provided enterprise datasets.)" - that is not a refusal.


def normalize(text: str) -> str:
    """Strip Markdown emphasis and collapse whitespace so keywords can match.

    The model often writes **two** approvals, which would not match the keyword
    "two approvals" without removing the asterisks first.
    """
    text = re.sub(r"[*`]", "", text)
    return re.sub(r"\s+", " ", text).lower()


def cites_source(answer: str, expected_source: str) -> bool:
    """Accept either the file name or its readable form, e.g. 'employee handbook'."""
    file_name = expected_source.lower()
    readable = file_name.removesuffix(".md").replace("_", " ")
    return file_name in answer or readable in answer


def is_refusal(answer: str) -> bool:
    return any(phrase in answer for phrase in REFUSAL_PHRASES)


def grade_keywords(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Keyword-based grading, shared with the LLM judge for comparison."""
    text = normalize(answer)
    grade: dict[str, Any] = {"refused": is_refusal(text)}
    if case["type"] == "answerable":
        grade["answer_correct"] = matches_keywords(text, case["answer_keywords"])
        grade["source_cited"] = cites_source(text, case["expected_source"])
        grade["passed"] = grade["answer_correct"] and grade["source_cited"]
    else:
        grade["passed"] = grade["refused"]
    return grade


def run_case(agent: LangChainSearchAgent, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    tool_calls = []
    answer = ""
    rounds = 0
    route = ""
    usage: dict[str, int] = {}
    for step in agent.search_stream(case["question"]):
        if step["step"] == "tool_call":
            tool_calls.append({"tool": step["tool_name"], "arguments": step["arguments"]})
        elif step["step"] == "final":
            answer = step["final_answer"]
            rounds = step["round"]
            route = step.get("route", "")
            usage = step.get("usage", {})

    return {
        "id": case["id"],
        "category": case["category"],
        "type": case["type"],
        "question": case["question"],
        "answer": answer,
        "tool_calls": tool_calls,
        "rounds": rounds,
        "route": route,
        "seconds": round(time.perf_counter() - started, 1),
        "usage": usage,
        **grade_keywords(case, answer),
    }


def build_agent(route: bool) -> LangChainSearchAgent:
    """route=False reproduces the agent before the router: every question sees all tools."""
    if route:
        return LangChainSearchAgent()
    return LangChainSearchAgent(
        router=lambda question: {"category": "unknown", "input_tokens": 0, "output_tokens": 0}
    )


def failure_reason(record: dict[str, Any]) -> str:
    if record["passed"]:
        return ""
    if record["type"] != "answerable":
        return "did not refuse"
    reasons = []
    if not record["answer_correct"]:
        reasons.append("refused instead of answering" if record["refused"] else "wrong or missing answer")
    if not record["source_cited"]:
        reasons.append("no source cited")
    return ", ".join(reasons)


def rate(records: list[dict[str, Any]], key: str) -> str:
    if not records:
        return "n/a"
    hits = sum(1 for r in records if r[key])
    return f"{hits}/{len(records)} ({hits / len(records):.0%})"


def summarize(records: list[dict[str, Any]]) -> dict[str, str]:
    answerable = [r for r in records if r["type"] == "answerable"]
    unrelated = [r for r in records if r["type"] == "unrelated"]
    near_miss = [r for r in records if r["type"] == "near_miss"]
    return {
        "overall_pass": rate(records, "passed"),
        "answer_accuracy": rate(answerable, "answer_correct"),
        "citation_accuracy": rate(answerable, "source_cited"),
        "unrelated_refusal": rate(unrelated, "passed"),
        "near_miss_refusal": rate(near_miss, "passed"),
        "avg_rounds": f"{sum(r['rounds'] for r in records) / len(records):.1f}",
        "total_seconds": f"{sum(r['seconds'] for r in records):.0f}",
        "input_tokens": str(sum(r["usage"].get("input_tokens", 0) for r in records)),
        "output_tokens": str(sum(r["usage"].get("output_tokens", 0) for r in records)),
        "router_tokens": str(sum(r["usage"].get("router_tokens", 0) for r in records)),
        # Which branch each question took, so a new failure can be traced to a wrong route.
        "routes": json.dumps(dict(Counter(r["route"] for r in records))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end evaluation on the golden set.")
    parser.add_argument("--no-router", action="store_true", help="show every tool on every call")
    args = parser.parse_args()

    cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    agent = build_agent(route=not args.no_router)
    records = []

    for index, case in enumerate(cases, start=1):
        record = run_case(agent, case)
        records.append(record)
        status = "PASS" if record["passed"] else "FAIL"
        reason = failure_reason(record)
        print(
            f"[{index:2d}/{len(cases)}] {status} {case['id']:28s} "
            f"rounds={record['rounds']} route={record['route']:8s} {record['seconds']:5.1f}s"
            + (f"  <- {reason}" if reason else ""),
            flush=True,
        )

    summary = summarize(records)
    config = {
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "vector_distance_threshold": VECTOR_DISTANCE_THRESHOLD,
        "router": not args.no_router,
    }

    print("\nConfig:", json.dumps(config))
    for key, value in summary.items():
        print(f"  {key:18s} {value}")

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "" if config["router"] else "-norouter"
    output = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-threshold-{VECTOR_DISTANCE_THRESHOLD}{suffix}.json"
    output.write_text(
        json.dumps({"config": config, "summary": summary, "cases": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved full answers to {output}")


if __name__ == "__main__":
    main()
