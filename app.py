from __future__ import annotations

import json
import uuid
from typing import Any

import streamlit as st

from config import EMBEDDING_MODEL, LLM_MODEL, MAX_SEARCH_ROUNDS, VECTOR_DISTANCE_THRESHOLD
from core.lc_agent import LangChainSearchAgent
from core.prompts import SYSTEM_PROMPT
from seed_data_large import ensure_seed_data

st.set_page_config(page_title="Agentic search", page_icon=":material/travel_explore:", layout="wide")

SUGGESTIONS = {
    ":material/restaurant: Daily meal allowance": "How much can I spend on meals per day when travelling?",
    ":material/rocket_launch: Release window": "On which days are production releases deployed?",
    ":material/engineering: Projects in progress": "Which engineering projects are currently in progress?",
    ":material/code: Validation code": "What does the code repository contain about validation?",
}
TOOL_RESULT_PREVIEW_CHARS = 2000


@st.cache_resource(show_spinner="Loading search engines...")
def get_agent() -> LangChainSearchAgent:
    ensure_seed_data()
    # One agent is shared by every browser tab. Its checkpointer keeps a separate
    # conversation per thread_id, and each tab holds its own thread_id in session state.
    return LangChainSearchAgent()


def start_new_conversation() -> None:
    # Reset both together: thread_id selects the agent's memory, messages are what the page shows.
    # Resetting only one would leave the page and the agent remembering different conversations.
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pop("suggestion", None)


def queue_suggestion() -> None:
    selected = st.session_state.suggestion
    if selected:
        st.session_state.pending_question = SUGGESTIONS[selected]


def trace_label(reply: dict[str, Any]) -> str:
    calls = len(reply["tool_calls"])
    parts = ["Search trace", f"{calls} tool call{'' if calls == 1 else 's'}" if calls else "no tool calls"]
    if reply.get("route"):
        parts.append(f"{reply['route']} tools")
    context_tokens = reply["usage"].get("context_tokens")
    if context_tokens:
        parts.append(f"{context_tokens:,} context tokens")
    if reply["error"]:
        parts.append("failed")
    return " · ".join(parts)


def render_tool_call(call: dict[str, Any]) -> None:
    st.markdown(f"**{call['tool']}**")
    st.code(json.dumps(call["arguments"], ensure_ascii=False, indent=2, default=str), language="json")
    result = str(call["result"])
    if len(result) > TOOL_RESULT_PREVIEW_CHARS:
        result = result[:TOOL_RESULT_PREVIEW_CHARS] + "\n... (truncated)"
    st.code(result, language="json", wrap_lines=True)


def render_trace_details(reply: dict[str, Any]) -> None:
    if not reply["tool_calls"]:
        st.caption("No tool calls in this turn.")
    for call in reply["tool_calls"]:
        render_tool_call(call)
    usage = reply["usage"]
    if usage:
        st.caption(
            f"{usage.get('model_calls', 0)} model calls · {usage.get('input_tokens', 0):,} input tokens · "
            f"{usage.get('output_tokens', 0):,} output tokens · "
            f"about {usage.get('tool_result_tokens', 0):,} tokens of tool results · "
            f"{usage.get('router_tokens', 0):,} tokens to pick the tool group"
        )
    if reply["error"]:
        st.error(reply["error"], icon=":material/error:")


def run_turn(agent: LangChainSearchAgent, question: str, thread_id: str) -> dict[str, Any]:
    """Run one turn, showing tool calls live, and return everything needed to redraw it later."""
    reply: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [],
        "usage": {},
        "route": "",
        "error": None,
        "fallback": False,
    }
    with st.status("Searching...", expanded=False) as status:
        for step in agent.search_stream(question, thread_id=thread_id):
            kind = step["step"]
            if kind == "tool_call":
                call = {"tool": step["tool_name"], "arguments": step["arguments"], "result": step["result"]}
                reply["tool_calls"].append(call)
                status.update(label=f"Searching... {len(reply['tool_calls'])} tool call(s)")
                render_tool_call(call)
            elif kind == "api_error":
                reply["error"] = step["error"]
                st.error(step["error"], icon=":material/error:")
            elif kind == "local_fallback":
                reply["fallback"] = True
            elif kind == "final":
                reply["content"] = step["final_answer"]
                reply["route"] = step.get("route", "")
                reply["usage"] = step.get("usage", {})
        status.update(label=trace_label(reply), state="error" if reply["error"] else "complete")
    return reply


def render_assistant_extras(reply: dict[str, Any]) -> None:
    if reply.get("fallback"):
        st.warning("No API key is configured, so this answer comes from the local fallback search "
                   "and has no conversation memory.", icon=":material/warning:")


st.session_state.setdefault("messages", [])
if "thread_id" not in st.session_state:
    start_new_conversation()

agent = get_agent()

with st.sidebar:
    st.button("New conversation", icon=":material/add_comment:", on_click=start_new_conversation, width="stretch")
    # The sidebar is drawn before the current question runs, so it shows no turn count that would lag by one.
    st.caption(f"Conversation `{st.session_state.thread_id[:8]}`")

    st.subheader("Configuration")
    st.markdown(
        f"- Agent model: `{LLM_MODEL}`\n"
        f"- Embedding model: `{EMBEDDING_MODEL}`\n"
        f"- Distance threshold: `{VECTOR_DISTANCE_THRESHOLD}`\n"
        f"- Max search rounds per turn: `{MAX_SEARCH_ROUNDS}`"
    )

    st.subheader("Data sources")
    st.markdown(
        "- SQLite records\n"
        "- Chroma vector collections, including the employee handbook and engineering guide\n"
        "- Whoosh keyword indexes\n"
        "- Sample code repository\n"
        "- Simulated HR, finance, project, and wiki systems"
    )

    with st.expander("System prompt", icon=":material/description:"):
        st.code(SYSTEM_PROMPT, language="markdown", wrap_lines=True)

st.title("Agentic search")
st.caption("Ask follow-up questions: the agent remembers this conversation until you start a new one.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            with st.expander(trace_label(message), icon=":material/manage_search:"):
                render_trace_details(message)
            render_assistant_extras(message)
        st.markdown(message["content"])

# submit_mode="disable" blocks a second question while a turn is running, so two turns
# never write to the same thread at once.
question = st.chat_input("Ask about policies, projects, finance, or code", submit_mode="disable")
question = question or st.session_state.pop("pending_question", None)

if not st.session_state.messages and not question:
    st.pills("Try asking", list(SUGGESTIONS), key="suggestion", on_change=queue_suggestion,
             label_visibility="collapsed")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        reply = run_turn(agent, question, st.session_state.thread_id)
        render_assistant_extras(reply)
        st.markdown(reply["content"])

    st.session_state.messages.append(reply)
