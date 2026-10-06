"""Question router: decides which tool descriptions the model is shown.

Every model call resends every tool description, so eight tools cost about 970
tokens on each of the two or three calls a question takes. Most questions use
one or two of them. The router spends one cheap call (the question and four
category names, no tool descriptions) to pick a category, and the model node
then sends only that category's tools.

Routing can be wrong, so "unknown" keeps every tool available and the router
falls back to it whenever it is unsure or the call fails.
"""
from __future__ import annotations

from typing import Any, Callable, Literal, get_args

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from core.logger import Logger

Category = Literal["docs", "code", "records", "unknown"]
ROUTES: tuple[str, ...] = get_args(Category)

# Tool names per category. This mapping lives in code, so it costs no tokens.
TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    # list_sources is in every group: it is 43 tokens and it is what the model
    # uses to say which data sources exist when it cannot answer.
    #
    # enterprise_call is left out of docs on purpose. Its wiki holds four short
    # policy pages, and each is a near copy of an entry in the keyword indexes,
    # which docs already searches. Adding it would cost 230 tokens per call to
    # reach text the group can already find.
    "docs": ("vector_search", "keyword_search", "list_sources"),
    "code": ("code_search", "code_read_file", "list_sources"),
    "records": ("database_search", "database_sql", "enterprise_call", "list_sources"),
}

ROUTER_PROMPT = """You route an enterprise search question to the data source that can answer it.

docs: policies, the employee handbook, the engineering guide, company documents, meeting notes
code: the sample code repository, source files, configuration written in code
records: employee, project, finance and HR records, and the internal wiki
unknown: anything else, including follow-up questions that do not name a topic

Choose unknown when you are unsure. It keeps every tool available, which costs
more tokens but never blocks the right tool."""


class Route(BaseModel):
    category: Category = Field(description="Which data source can answer the question.")


def tools_for(tools: list, category: str) -> list:
    """The tools of one category, or all of them for 'unknown' and anything unexpected."""
    names = TOOL_GROUPS.get(category)
    if not names:
        return tools
    return [tool for tool in tools if tool.name in names]


def create_router(model: BaseChatModel) -> Callable[[str], dict[str, Any]]:
    """Return classify(question) -> {"category", "input_tokens", "output_tokens"}.

    include_raw=True keeps the raw reply next to the parsed one, so the router's own
    token cost can be added to the turn instead of being invisible.
    """
    # function_calling for the same reason as the judges: DashScope rejects json_schema
    # and json_mode unless the prompt contains the word "json". A forced tool call is
    # rejected in thinking mode, which some Qwen models enable by default.
    classifier = model.with_structured_output(
        Route,
        method="function_calling",
        include_raw=True,
        extra_body={"enable_thinking": False},
    )

    def classify(question: str) -> dict[str, Any]:
        try:
            result = classifier.invoke([("system", ROUTER_PROMPT), ("human", question)])
        except Exception as exc:
            # A failed routing call must not fail the turn: fall back to every tool.
            # The warning is the only sign of it, because the answer still arrives;
            # only the token saving is lost.
            Logger().warning(f"Routing call failed, showing every tool: {exc}")
            return {"category": "unknown", "input_tokens": 0, "output_tokens": 0}
        parsed: Route | None = result.get("parsed")
        if parsed is None:
            Logger().warning("Routing reply had no category, showing every tool")
        usage = getattr(result.get("raw"), "usage_metadata", None) or {}
        return {
            "category": parsed.category if parsed else "unknown",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    return classify
