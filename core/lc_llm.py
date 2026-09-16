"""Step 1: the LangChain version of core/llm.py.

The old LLMClient wrapped the openai SDK by hand and converted responses into
plain dicts. LangChain's ChatOpenAI does all of that for us. DashScope exposes
an OpenAI-compatible endpoint, so we only need to point base_url at it.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def has_api_key() -> bool:
    return bool(LLM_API_KEY.strip())


def get_chat_model(temperature: float = 0.1, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or LLM_MODEL,
        base_url=LLM_BASE_URL,
        # ChatOpenAI refuses an empty key, so use a placeholder when none is set.
        # The agent checks has_api_key() before ever calling the model.
        api_key=LLM_API_KEY or "missing-key",
        temperature=temperature,
    )
