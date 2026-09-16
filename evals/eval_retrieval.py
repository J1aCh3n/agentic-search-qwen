"""Retrieval-only evaluation: no chat model, only embeddings."""
import json
from pathlib import Path

from engines.vector_db import VectorDBEngine

GOLDEN_SET = Path(__file__).parent / "golden_set.json"
TOP_K = 3


def matches_keywords(text: str, keyword_groups: list[list[str]]) -> bool:
    text = text.lower()

    return all(
        any(keyword.lower() in text for keyword in group)
        for group in keyword_groups
    )

def find_rank(hits: list[dict], case: dict) -> int | None:
    """Return the 1-based rank of the first correct chunk, or None."""
    for rank, hit in enumerate(hits, start=1):
        source_matches = (
            hit["metadata"]["source"] == case["expected_source"]
        )
        content_matches = matches_keywords(
            hit["content"],
            case["answer_keywords"],
        )

        if source_matches and content_matches:
            return rank

    return None



def main() -> None:
    cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    vector_db = VectorDBEngine()

    hit1 = hit3 = 0
    reciprocal_ranks = []
    answerable = [c for c in cases if c["type"] == "answerable"]
    unrelated = [c for c in cases if c["type"] == "unrelated"]

    for case in answerable:
        result = json.loads(vector_db.search(case["question"], collection_name="handbook", n_results=TOP_K))
        hits = result['results']
        rank = find_rank(hits, case)
        
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= TOP_K:
            hit3 += 1
        reciprocal_ranks.append(1/rank if rank is not None else 0.0)
        top_distance = hits[0]["distance"] if hits else None
        print(
            f"{case['id']}: "
            f"rank={rank}, top_distance={top_distance}"
        )

    

    blocked = 0
    for case in unrelated:
        result = json.loads(vector_db.search(case["question"], collection_name="handbook", n_results=TOP_K))
        is_blocked = 'message' in result
        if is_blocked:
            blocked += 1
        print(
            f"{case['id']}: "
            f"blocked={is_blocked}"
        )

    if answerable:
        count = len(answerable)

        print(f"Hit@1: {hit1 / count:.2%}")
        print(f"Hit@{TOP_K}: {hit3 / count:.2%}")
        print(
            f"MRR@3: "
            f"{sum(reciprocal_ranks) / count:.4f}"
        )
    else:
        print("没有 answerable 测试题，无法计算命中指标。")

    if unrelated:
        print(f"无关题拦截率: {blocked / len(unrelated):.2%}")
    else:
        print("没有 unrelated 测试题，无法计算拦截率。")


if __name__ == "__main__":
    main()