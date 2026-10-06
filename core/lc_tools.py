"""Step 2a: the LangChain version of the tools in core/agent.py.

The old agent described every tool twice: once as a JSON schema in
_define_tools() and once as an if-branch in _execute_tool(). With @tool, a
single Python function is the whole tool:

- function name  -> tool name
- type hints     -> JSON schema for the arguments
- docstring      -> the description the model reads to decide when to call it
"""
from __future__ import annotations

import json
from typing import Any, Literal

from langchain.tools import tool

from engines.code_search import CodeSearchEngine
from engines.database import DatabaseEngine
from engines.enterprise_sdk import EnterpriseSDK
from engines.keyword_search import KeywordSearchEngine
from engines.vector_db import VectorDBEngine


def build_tools(
    db: DatabaseEngine,
    vector_db: VectorDBEngine,
    keyword_engine: KeywordSearchEngine,
    code_engine: CodeSearchEngine,
    enterprise_sdk: EnterpriseSDK,
) -> list:
    """Create the tool list. The engines are passed in so they are created only once."""

    @tool
    def database_search(
        query: str,
        table: Literal["employees", "departments", "projects", "contracts", "products"] | None = None,
        limit: int = 10,
    ) -> str:
        """Search local SQLite structured records: employees, departments, projects, contracts, products."""
        return db.search(query, table=table, limit=limit)

    @tool
    def database_sql(sql: str) -> str:
        """Execute a read-only SQLite SELECT query over employees, departments, projects, contracts, or products."""
        # These checks only give the model a clear message. They miss a keyword after a
        # newline, so the protection is the read-only connection in execute_sql().
        normalized = f" {sql.strip().lower()} "
        if not normalized.strip().startswith("select"):
            return json.dumps({"error": "Only SELECT statements are allowed."})
        blocked = [" insert ", " update ", " delete ", " drop ", " alter ", " create ", " pragma "]
        if any(token in normalized for token in blocked):
            return json.dumps({"error": "Only read-only SELECT statements are allowed."})
        return db.execute_sql(sql)

    @tool
    def vector_search(
        query: str,
        collection_name: Literal["company_info", "tech_docs", "meeting_notes", "handbook"] | None = None,
        n_results: int = 5,
    ) -> str:
        """Search semantic document collections: company profile, technical docs, meeting notes,
            and the employee handbook plus engineering guide (HR policies, expenses, leave, code review,
            release process, on-call)."""
        return vector_db.search(query, collection_name=collection_name, n_results=n_results)

    @tool
    def keyword_search(
        query: str,
        index_name: Literal["policies", "tech_articles"] | None = None,
        limit: int = 10,
    ) -> str:
        """Keyword search over company policy and engineering article indexes."""
        return keyword_engine.search(query, index_name=index_name, limit=limit)

    @tool
    def code_search(
        query: str,
        file_pattern: str | None = None,
        search_type: Literal["content", "filename", "symbol"] = "content",
        limit: int = 10,
    ) -> str:
        """Search the generated sample code repository by file content, file name, or symbol name."""
        return code_engine.search(query, file_pattern=file_pattern, search_type=search_type, limit=limit)

    @tool
    def code_read_file(file_path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read part of a file from the generated sample code repository."""
        return code_engine.read_file(file_path, start_line=start_line, end_line=end_line)

    @tool
    def enterprise_call(
        system: Literal["hr", "finance", "project", "wiki"],
        action: str,
        params: dict[str, Any] | str | None = None,
    ) -> str:
        """Call a simulated enterprise system.

        Available actions:
        - hr: get_employee(employee_id), search_employees(keyword), get_org_chart
        - finance: get_transactions / get_payments(type, category, status, month, limit)
          type is "income" or "expense"; month is two digits, e.g. "05" for May
        - project: get_project(project_id), search_projects(keyword, status), get_milestones(project_id)
        - wiki: search_policies(keyword), search_tech_docs(keyword), get_all_categories
        """
        # Qwen sometimes sends params as a JSON string instead of an object.
        if isinstance(params, str):
            try:
                params = json.loads(params) if params.strip() else {}
            except json.JSONDecodeError:
                return json.dumps({"error": "params must be a JSON object."})
        return enterprise_sdk.call(system, action, params or {})

    @tool
    def list_sources() -> str:
        """List available demo data sources and their schemas."""
        repo = json.loads(code_engine.list_files())
        payload = {
            "sqlite_schema": json.loads(db.get_schema()),
            "vector_collections": json.loads(vector_db.get_all_collections_info()),
            "keyword_indexes": json.loads(keyword_engine.get_all_indexes_info()),
            "enterprise_systems": json.loads(enterprise_sdk.list_systems()),
            # Top-level entries only. The full file list would add about 500 tokens to every
            # call, and code_search already finds a file by name when one is needed.
            "code_repository": {
                "total_files": repo.get("total_files", 0),
                "top_level": sorted({path.replace("\\", "/").split("/")[0] for path in repo.get("files", [])}),
            },
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    return [
        database_search,
        database_sql,
        vector_search,
        keyword_search,
        code_search,
        code_read_file,
        enterprise_call,
        list_sources,
    ]
