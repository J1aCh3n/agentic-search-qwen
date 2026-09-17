"""Step 2b: the LangChain version of core/agent.py.

The old search_stream() hand-wrote the agent loop: call the model, parse tool
calls, run tools, append messages, repeat. create_agent() builds that exact
loop for us (as a small LangGraph graph with a "model" node and a "tools" node).

The rest of this file only translates LangChain's stream into the step dicts
that app.py already knows how to display, so the UI does not need to change.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from config import LLM_BASE_URL, LLM_MODEL, MAX_SEARCH_ROUNDS
from core.agent import SYSTEM_PROMPT, AgenticSearchAgent
from core.lc_llm import get_chat_model, has_api_key
from core.lc_tools import build_tools
from core.logger import Logger
from engines.code_search import CodeSearchEngine
from engines.database import DatabaseEngine
from engines.enterprise_sdk import EnterpriseSDK
from engines.keyword_search import KeywordSearchEngine
from engines.vector_db import VectorDBEngine


NOT_SAVED_NOTE = "This question was not saved to the conversation. Please ask again."


class LangChainSearchAgent:
    def __init__(
        self,
        model: BaseChatModel | None = None,
        tools: list | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        """Defaults build the real agent. Tests pass a fake model and tools to run offline."""
        self.logger = Logger()
        if tools is None:
            tools = build_tools(
                DatabaseEngine(),
                VectorDBEngine(),
                KeywordSearchEngine(),
                CodeSearchEngine(),
                EnterpriseSDK(),
            )
        # This one line replaces the whole hand-written loop in core/agent.py.
        # The checkpointer keeps each conversation's messages under its thread_id.
        self.agent = create_agent(
            model=model or get_chat_model(),
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer or InMemorySaver(),
        )

    def search(self, question: str, thread_id: str | None = None, verbose: bool = False) -> str:
        final_answer = ""
        for step in self.search_stream(question, thread_id):
            if verbose and step.get("step") == "tool_call":
                print(f"[*] Tool: {step.get('tool_name')} {step.get('arguments')}")
            if step.get("step") == "final":
                final_answer = step.get("final_answer", "")
        return final_answer

    def search_stream(self, question: str, thread_id: str | None = None):
        # No thread_id means a fresh conversation, so independent callers such as
        # eval_agent never see each other's history.
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": MAX_SEARCH_ROUNDS * 2 + 1,
        }
        yield {
            "step": "init",
            "question": question,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": SYSTEM_PROMPT,
        }

        # Without a key there is no model to drive the agent, so reuse the old
        # rule-based fallback search. It is not part of the LangChain lesson.
        if not has_api_key():
            trace: dict[str, Any] = {}
            answer = AgenticSearchAgent()._local_search(question, trace)
            yield {"step": "local_fallback", "round": 0, "trace": trace}
            yield {"step": "final", "round": 0, "final_answer": answer}
            return

        round_idx = 0
        pending_calls: dict[str, dict[str, Any]] = {}  # tool_call_id -> {"name", "args"}
        final_answer = ""
        messages_before = len(self.agent.get_state(config).values.get("messages", []))
        turn_completed = False
        usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "context_tokens": 0, "tool_result_tokens": 0}

        try:
            # stream_mode="updates" yields one dict per finished node:
            #   {"model": {"messages": [AIMessage]}}    the model spoke
            #   {"tools": {"messages": [ToolMessage]}}  a tool returned
            # Each round is two graph steps (model + tools), hence * 2.
            stream = self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
                stream_mode="updates",
            )
            for chunk in stream:
                for node, update in chunk.items():
                    for message in (update or {}).get("messages", []):
                        if node == "model":
                            round_idx += 1
                            call_usage = message.usage_metadata or {}
                            usage["model_calls"] += 1
                            usage["input_tokens"] += call_usage.get("input_tokens", 0)
                            usage["output_tokens"] += call_usage.get("output_tokens", 0)
                            usage["context_tokens"] = call_usage.get("input_tokens", 0)
                            for call in message.tool_calls:
                                pending_calls[call["id"]] = call
                            if not message.tool_calls:
                                final_answer = message.text
                        elif node == "tools":
                            usage["tool_result_tokens"] += count_tokens_approximately([message])
                            call = pending_calls.pop(message.tool_call_id, {})
                            yield {
                                "step": "tool_call",
                                "round": round_idx,
                                "tool_name": message.name,
                                "arguments": call.get("args", {}),
                                "result": message.content,
                            }
            turn_completed = True
        except GraphRecursionError:
            final_answer = (
                f"Stopped after {MAX_SEARCH_ROUNDS} rounds without reaching a final answer.\n\n"
                f"{NOT_SAVED_NOTE}"
            )
        except Exception as exc:
            # The failure can come from the model API or from a tool, so the message names neither.
            yield {
                "step": "api_error",
                "round": round_idx,
                "error": str(exc),
                "model": LLM_MODEL,
                "base_url": LLM_BASE_URL,
            }
            final_answer = (
                "The search failed, so no answer was generated.\n\n"
                f"- Error: {exc}\n\n"
                f"{NOT_SAVED_NOTE}"
            )
        finally:
            # finally also runs on GeneratorExit, when the caller stops reading mid-turn.
            if not turn_completed:
                self._rollback_turn(config, messages_before)

        final_answer = final_answer or "The model did not return an answer."
        self.logger.log_final_answer(question, final_answer, round_idx)
        yield {"step": "final", "round": round_idx, "final_answer": final_answer, "usage": usage}

    def _rollback_turn(self, config: dict, messages_before: int) -> None:
        """Remove every message an unfinished turn added, so the thread ends on a complete turn."""
        messages = self.agent.get_state(config).values.get("messages", [])
        added = messages[messages_before:]
        if added:
            self.agent.update_state(config, {"messages": [RemoveMessage(id=m.id) for m in added]})