"""Model setup and retry shared by the LLM judges."""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from config import JUDGE_MODEL
from core.lc_llm import get_chat_model

T = TypeVar("T")

ATTEMPTS = 2
RETRY_DELAY_SECONDS = 3


def create_structured_judge(schema: type) -> Any:
    # function_calling sends the schema as a tool and forces the model to call it, so the
    # verdict format does not depend on prompt wording. DashScope rejects json_schema and
    # json_mode unless the prompt contains the word "json".
    # Forced tool calls are rejected in thinking mode, which qwen3.7-plus enables by default.
    return get_chat_model(temperature=0, model=JUDGE_MODEL).with_structured_output(
        schema,
        method="function_calling",
        extra_body={"enable_thinking": False},
    )


def with_retry(call: Callable[[], T]) -> T:
    """Retry once: judge failures are often transient, such as a 429 or a skipped tool call."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return call()
        except Exception:
            if attempt == ATTEMPTS:
                raise
            time.sleep(RETRY_DELAY_SECONDS)
    raise AssertionError("unreachable")
