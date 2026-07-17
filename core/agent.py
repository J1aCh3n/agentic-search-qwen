from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from config import LLM_BASE_URL, LLM_MODEL, MAX_SEARCH_ROUNDS
from core.llm import LLMClient
from core.logger import Logger
from engines.code_search import CodeSearchEngine
from engines.database import DatabaseEngine
from engines.enterprise_sdk import EnterpriseSDK
from engines.keyword_search import KeywordSearchEngine
from engines.vector_db import VectorDBEngine


SYSTEM_PROMPT = """You are an enterprise search assistant.

You can answer by calling tools over five synthetic data sources:
- SQLite structured records for employees, departments, projects, contracts, and products
- Chroma vector collections for company profile, technical documentation, and meeting notes
- Whoosh keyword indexes for company policies and engineering articles
- A generated sample code repository
- Simulated enterprise systems for HR, finance, project management, and internal wiki content

Rules:
- Use tools when the question asks for specific records, policies, code, financial data, or project status.
- Prefer finance tools for revenue, expense, payment, transaction, invoice, budget, or reimbursement questions.
- Prefer HR tools for employee, department, leave, hiring, org chart, or onboarding questions.
- Prefer project tools for progress, milestones, deadline, owner, or delivery questions.
- Prefer code search tools for source-code, function, endpoint, service, or repository questions.
- When you have enough evidence, provide a concise final answer with source-aware details.
- If data is missing, say that the demo dataset does not contain enough evidence.
"""


class AgenticSearchAgent:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.logger = Logger()
        self.db = DatabaseEngine()
        self.vector_db = VectorDBEngine()
        self.keyword_search = KeywordSearchEngine()
        self.code_search = CodeSearchEngine()
        self.enterprise_sdk = EnterpriseSDK()
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
                fallback = self._local_search(question, trace)
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
                final_answer = self._local_search(question, trace)
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

    def _local_search(self, question: str, trace: dict[str, Any]) -> str:
        keywords = self._extract_keywords(question)
        intent = self._detect_intent(question)
        trace["keywords"] = keywords
        trace["intent"] = intent

        sections: list[str] = []

        if intent in {"finance", "general"}:
            finance_params = {"limit": 10}
            if "cloud" in question.lower():
                finance_params["category"] = "cloud"
            month = self._detect_month(question)
            if month:
                finance_params["month"] = month
            finance = json.loads(self.enterprise_sdk.call("finance", "get_transactions", finance_params))
            if finance.get("data"):
                summary = finance.get("summary", {})
                lines = [
                    "Finance findings:",
                    f"- Matching records: {summary.get('count', 0)}",
                    f"- Total income: {summary.get('total_income', 0):,}",
                    f"- Total expense: {summary.get('total_expense', 0):,}",
                    f"- Net amount: {summary.get('net_amount', 0):,}",
                ]
                for item in finance["data"][:5]:
                    lines.append(
                        f"- {item['date']}: {item['type']} {item['amount']:,} "
                        f"for {item['category']} ({item['counterpart']}, {item['status']})"
                    )
                sections.append("\n".join(lines))

        if intent in {"hr", "general"}:
            db_result = json.loads(self.db.search(question, table="employees", limit=10))
            if db_result:
                lines = ["Employee matches:"]
                for employee in db_result[:8]:
                    lines.append(
                        f"- {employee['name']}: {employee['position']}, "
                        f"{employee['department']}, email {employee['email']}"
                    )
                sections.append("\n".join(lines))

        if intent in {"project", "general"}:
            projects = json.loads(self.enterprise_sdk.call("project", "search_projects", {"keyword": keywords[0] if keywords else ""}))
            if projects.get("data"):
                lines = ["Project matches:"]
                for item in projects["data"][:8]:
                    lines.append(
                        f"- {item['name']}: {item['status']}, {item['progress']}% complete, "
                        f"deadline {item['deadline']}, priority {item['priority']}"
                    )
                sections.append("\n".join(lines))

        if intent in {"code", "general"}:
            code_query = self._choose_code_query(question, keywords)
            content_results = json.loads(self.code_search.search(code_query, search_type="content", limit=5))
            filename_results = json.loads(self.code_search.search(code_query, search_type="filename", limit=5))
            code_lines = self._format_code_results(code_query, content_results, filename_results)
            if code_lines:
                sections.append(code_lines)

        if intent in {"policy", "general"}:
            keyword_query = self._choose_policy_query(question, keywords)
            policy_results = json.loads(self.keyword_search.search(keyword_query, index_name="policies", limit=5))
            article_results = json.loads(self.keyword_search.search(keyword_query, index_name="tech_articles", limit=5))
            policy_lines = self._format_keyword_results(keyword_query, policy_results, article_results)
            if policy_lines:
                sections.append(policy_lines)

        if not sections:
            db_result = json.loads(self.db.search(question, limit=5))
            if db_result:
                sections.append(self._format_database_results(db_result))

        if not sections:
            return "The local fallback search did not find enough evidence in the demo dataset."

        return "\n\n".join(sections)

    def _choose_code_query(self, question: str, keywords: list[str]) -> str:
        text = question.lower()
        if "validation" in text or "validator" in text:
            return "validate"
        if "config" in text or "configuration" in text:
            return "settings"
        if "chat" in text:
            return "chat"
        if "service" in text:
            return "service"
        return keywords[0] if keywords else question

    def _detect_month(self, question: str) -> str | None:
        months = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
        }
        text = question.lower()
        for name, number in months.items():
            if name in text:
                return number
        return None

    def _choose_policy_query(self, question: str, keywords: list[str]) -> str:
        text = question.lower()
        if "remote" in text:
            return "remote"
        if "expense" in text or "reimbursement" in text:
            return "expense"
        if "review" in text:
            return "review"
        if "api" in text:
            return "api"
        return keywords[0] if keywords else question

    def _format_code_results(
        self,
        query: str,
        content_results: dict[str, Any],
        filename_results: dict[str, Any],
    ) -> str:
        results = content_results.get("results", [])
        filenames = filename_results.get("results", [])
        if not results and not filenames:
            return ""

        lines = [f"Code repository findings for '{query}':"]
        seen_files = set()
        for item in [*filenames, *results]:
            file_path = item.get("file_path")
            if not file_path or file_path in seen_files:
                continue
            seen_files.add(file_path)
            lines.append(f"- {file_path}")
            for match in item.get("matches", [])[:3]:
                lines.append(f"  - line {match['line_number']}: {match['content']}")
        return "\n".join(lines)

    def _format_keyword_results(
        self,
        query: str,
        policy_results: dict[str, Any],
        article_results: dict[str, Any],
    ) -> str:
        combined = []
        for source_name, payload in [("policy", policy_results), ("technical article", article_results)]:
            for item in payload.get("results", []):
                combined.append((source_name, item))
        if not combined:
            return ""

        lines = [f"Keyword search findings for '{query}':"]
        for source_name, item in combined[:6]:
            lines.append(f"- {item.get('title', 'Untitled')} ({source_name}): {item.get('content', '')[:180]}")
        return "\n".join(lines)

    def _format_database_results(self, db_result: dict[str, Any]) -> str:
        lines = ["SQLite structured data findings:"]
        for table_name, rows in db_result.items():
            if not rows:
                continue
            lines.append(f"- {table_name}: {len(rows)} match(es)")
            for row in rows[:3]:
                label = row.get("name") or row.get("client_name") or row.get("title") or f"id={row.get('id')}"
                detail = row.get("position") or row.get("status") or row.get("category") or row.get("description", "")
                lines.append(f"  - {label}: {detail}")
        return "\n".join(lines)

    def _extract_keywords(self, question: str) -> list[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "for",
            "in",
            "on",
            "with",
            "what",
            "which",
            "who",
            "how",
            "is",
            "are",
            "show",
            "find",
            "tell",
            "me",
        }
        tokens = [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in question.split()
            if token.strip(".,!?;:()[]{}\"'")
        ]
        keywords = [token for token in tokens if token not in stopwords and len(token) > 1]
        return keywords or [question.strip()]

    def _detect_intent(self, question: str) -> str:
        text = question.lower()
        finance_keywords = {
            "payment",
            "transaction",
            "income",
            "expense",
            "revenue",
            "invoice",
            "budget",
            "cost",
            "finance",
            "cloud service",
        }
        hr_keywords = {"employee", "department", "leave", "hr", "hiring", "onboarding", "org", "manager"}
        project_keywords = {"project", "milestone", "deadline", "delivery", "progress", "sprint", "roadmap"}
        code_keywords = {"code", "function", "class", "endpoint", "repository", "service", "api", "validation", "validator"}
        policy_keywords = {"policy", "process", "remote", "reimbursement", "expense", "review", "standard"}

        if any(keyword in text for keyword in finance_keywords):
            return "finance"
        if any(keyword in text for keyword in hr_keywords):
            return "hr"
        if any(keyword in text for keyword in project_keywords):
            return "project"
        if any(keyword in text for keyword in code_keywords):
            return "code"
        if any(keyword in text for keyword in policy_keywords):
            return "policy"
        return "general"

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
