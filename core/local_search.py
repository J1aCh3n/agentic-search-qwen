"""Rule-based search used when no model API key is configured.

It picks data sources from keywords in the question instead of letting a model choose tools.
Moved unchanged from core/agent.py so the LangChain agent no longer builds the old agent.
"""
from __future__ import annotations

import json
from typing import Any

from engines.code_search import CodeSearchEngine
from engines.database import DatabaseEngine
from engines.enterprise_sdk import EnterpriseSDK
from engines.keyword_search import KeywordSearchEngine


class LocalFallbackSearch:
    def __init__(
        self,
        db: DatabaseEngine,
        keyword_search: KeywordSearchEngine,
        code_search: CodeSearchEngine,
        enterprise_sdk: EnterpriseSDK,
    ) -> None:
        self.db = db
        self.keyword_search = keyword_search
        self.code_search = code_search
        self.enterprise_sdk = enterprise_sdk

    def search(self, question: str, trace: dict[str, Any]) -> str:
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
