from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from config import LLM_BASE_URL, LLM_MODEL, MAX_SEARCH_ROUNDS
from core.llm import LLMClient
from core.local_search import LocalFallbackSearch
from core.logger import Logger
from core.prompts import SYSTEM_PROMPT
from engines.code_search import CodeSearchEngine
from engines.database import DatabaseEngine
from engines.enterprise_sdk import EnterpriseSDK
from engines.keyword_search import KeywordSearchEngine
from engines.vector_db import VectorDBEngine


class AgenticSearchAgent:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.logger = Logger()
        self.db = DatabaseEngine()
        self.vector_db = VectorDBEngine()
        self.keyword_search = KeywordSearchEngine()
        self.code_search = CodeSearchEngine()
        self.enterprise_sdk = EnterpriseSDK()
        self.local_fallback = LocalFallbackSearch(self.db, self.keyword_search, self.code_search, self.enterprise_sdk)
        self.tools = self._define_tools()

    def _define_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "database_search",
                    "description": "Search local SQLite structured records.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query."},
                            "table": {
                                "type": "string",
                                "description": "Optional table name.",
                                "enum": ["employees", "departments", "projects", "contracts", "products"],
                            },
                            "limit": {"type": "integer", "description": "Maximum result count.", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "database_sql",
                    "description": "Execute a read-only SQLite SELECT query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "A SELECT statement over employees, departments, projects, contracts, or products.",
                            }
                        },
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "Search semantic document collections.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "collection_name": {
                                "type": "string",
                                "enum": ["company_info", "tech_docs", "meeting_notes"],
                            },
                            "n_results": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "keyword_search",
                    "description": "Search policy and engineering article indexes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "index_name": {"type": "string", "enum": ["policies", "tech_articles"]},
                            "limit": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "code_search",
                    "description": "Search the generated sample code repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "file_pattern": {"type": "string"},
                            "search_type": {"type": "string", "enum": ["content", "filename", "symbol"]},
                            "limit": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "code_read_file",
                    "description": "Read part of a file from the generated sample code repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer", "default": 1},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "enterprise_call",
                    "description": "Call a simulated enterprise system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system": {"type": "string", "enum": ["hr", "finance", "project", "wiki"]},
                            "action": {"type": "string"},
                            "params": {"type": "object"},
                        },
                        "required": ["system", "action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_sources",
                    "description": "List available demo data sources and schemas.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def search(self, question: str, verbose: bool = False) -> str:
        final_answer = ""
        for step in self.search_stream(question):
            if verbose and step.get("step") == "tool_call":
                print(f"[*] Tool: {step.get('tool_name')} {step.get('arguments')}")
            if step.get("step") == "final":
                final_answer = step.get("final_answer", "")
        return final_answer

    def search_stream(self, question: str):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        yield {
            "step": "init",
            "question": question,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": SYSTEM_PROMPT,
        }

        for round_idx in range(1, MAX_SEARCH_ROUNDS + 1):
            response = self.llm.chat_with_tools(messages, self.tools)
            content = response.get("content", "")
            api_error = response.get("error")

            if api_error or self._is_api_error(content):
                error_message = str(api_error or content)
                if self.llm.has_api_key:
                    final_answer = self._format_api_error(error_message)
                    yield {
                        "step": "api_error",
                        "round": round_idx,
                        "error": error_message,
                        "model": LLM_MODEL,
                        "base_url": LLM_BASE_URL,
                    }
                    yield {
                        "step": "final",
                        "round": round_idx,
                        "final_answer": final_answer,
                        "final_prompt_type": "model api error",
                        "final_messages_context": messages,
                    }
                    return
                trace: dict[str, Any] = {}
                fallback = self.local_fallback.search(question, trace)
                yield {"step": "local_fallback", "round": round_idx, "trace": trace}
                yield {
                    "step": "final",
                    "round": round_idx,
                    "final_answer": fallback,
                    "final_prompt_type": "local fallback",
                    "final_messages_context": messages,
                }
                return

            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                final_answer = content or "The model did not return an answer."
                self.logger.log_final_answer(question, final_answer, round_idx)
                yield {
                    "step": "final",
                    "round": round_idx,
                    "final_answer": final_answer,
                    "final_prompt_type": "model final answer",
                    "final_messages_context": messages,
                }
                return

            assistant_message: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": []}

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                arguments = self._parse_arguments(tool_call.get("arguments", "{}"))
                result = self._execute_tool(tool_name, arguments)

                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call.get("id", f"call_{round_idx}_{tool_name}"),
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                    }
                )

                yield {
                    "step": "tool_call",
                    "round": round_idx,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "llm_text_content": content,
                }

                messages.append(assistant_message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"call_{round_idx}_{tool_name}"),
                        "content": result,
                    }
                )

        final_prompt = "Provide the best answer possible from the gathered tool results."
        messages.append({"role": "user", "content": final_prompt})
        final_answer = self.llm.chat(messages)
        if self._is_api_error(final_answer):
            if self.llm.has_api_key:
                yield {
                    "step": "api_error",
                    "round": MAX_SEARCH_ROUNDS,
                    "error": final_answer,
                    "model": LLM_MODEL,
                    "base_url": LLM_BASE_URL,
                }
                final_answer = self._format_api_error(final_answer)
            else:
                trace = {}
                final_answer = self.local_fallback.search(question, trace)
                yield {"step": "local_fallback", "round": MAX_SEARCH_ROUNDS, "trace": trace}
        self.logger.log_final_answer(question, final_answer, MAX_SEARCH_ROUNDS)
        yield {
            "step": "final",
            "round": MAX_SEARCH_ROUNDS,
            "final_answer": final_answer,
            "final_prompt_type": "max rounds reached",
            "final_messages_context": messages,
        }

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        try:
            if tool_name == "database_search":
                return self.db.search(
                    arguments.get("query", ""),
                    table=arguments.get("table"),
                    limit=int(arguments.get("limit", 10)),
                )
            if tool_name == "database_sql":
                return self._safe_sql(arguments.get("sql", ""))
            if tool_name == "vector_search":
                return self.vector_db.search(
                    arguments.get("query", ""),
                    collection_name=arguments.get("collection_name"),
                    n_results=int(arguments.get("n_results", 5)),
                )
            if tool_name == "keyword_search":
                return self.keyword_search.search(
                    arguments.get("query", ""),
                    index_name=arguments.get("index_name"),
                    limit=int(arguments.get("limit", 10)),
                )
            if tool_name == "code_search":
                return self.code_search.search(
                    arguments.get("query", ""),
                    file_pattern=arguments.get("file_pattern"),
                    search_type=arguments.get("search_type", "content"),
                    limit=int(arguments.get("limit", 10)),
                )
            if tool_name == "code_read_file":
                return self.code_search.read_file(
                    arguments.get("file_path", ""),
                    start_line=int(arguments.get("start_line", 1)),
                    end_line=arguments.get("end_line"),
                )
            if tool_name == "enterprise_call":
                return self.enterprise_sdk.call(
                    arguments.get("system", ""),
                    arguments.get("action", ""),
                    arguments.get("params", {}) or {},
                )
            if tool_name == "list_sources":
                return self._list_sources()
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def _safe_sql(self, sql: str) -> str:
        normalized = sql.strip().lower()
        if not normalized.startswith("select"):
            return json.dumps({"error": "Only SELECT statements are allowed."})
        blocked = [" insert ", " update ", " delete ", " drop ", " alter ", " create ", " pragma "]
        padded = f" {normalized} "
        if any(token in padded for token in blocked):
            return json.dumps({"error": "Only read-only SELECT statements are allowed."})
        return self.db.execute_sql(sql)

    def _list_sources(self) -> str:
        payload = {
            "sqlite_schema": json.loads(self.db.get_schema()),
            "vector_collections": json.loads(self.vector_db.get_all_collections_info()),
            "keyword_indexes": json.loads(self.keyword_search.get_all_indexes_info()),
            "enterprise_systems": json.loads(self.enterprise_sdk.list_systems()),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        try:
            return json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}

    def _is_api_error(self, content: str) -> bool:
        if not content:
            return False
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict) or "error" not in payload:
            return False
        error = str(payload["error"]).lower()
        return any(
            term in error
            for term in ["api", "key", "auth", "unauthorized", "connection", "timeout", "rate limit", "not found"]
        )

    def _format_api_error(self, error: str) -> str:
        return (
            "Model API call failed, so the search result was not generated.\n\n"
            f"- Model: {LLM_MODEL}\n"
            f"- Base URL: {LLM_BASE_URL}\n"
            f"- Error: {error}\n\n"
            "Check that the DashScope API key, model entitlement, and endpoint region match."
        )
