from __future__ import annotations

import json

import streamlit as st

from core.agent import AgenticSearchAgent
from seed_data_large import ensure_seed_data


st.set_page_config(
    page_title="Agentic Search",
    page_icon="Search",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.4rem;
    }
    .trace-label {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.35rem;
        font-weight: 600;
    }
    .trace-prompt { background: #e0f2fe; color: #075985; }
    .trace-tool { background: #fef3c7; color: #92400e; }
    .trace-final { background: #dcfce7; color: #166534; }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        margin: 0.15rem;
        border-radius: 0.35rem;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_agent() -> AgenticSearchAgent:
    ensure_seed_data()
    return AgenticSearchAgent()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_searching" not in st.session_state:
        st.session_state.is_searching = False


def render_json_block(title: str, payload: object) -> None:
    with st.expander(title, expanded=False):
        st.code(json.dumps(payload, indent=2, ensure_ascii=False, default=str), language="json")


def perform_search(agent: AgenticSearchAgent, question: str) -> str:
    status_area = st.empty()
    trace_area = st.container()
    final_area = st.empty()

    final_answer = ""

    try:
        for step in agent.search_stream(question):
            step_name = step.get("step")

            if step_name == "init":
                status_area.info("Agent planning in progress...")
                with trace_area:
                    st.markdown("##### User Question")
                    st.info(step.get("question", question))
                    st.markdown(
                        "<span class='trace-label trace-prompt'>1. System Prompt</span>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("View system prompt", expanded=False):
                        st.code(step.get("system_prompt", ""), language="markdown")

            elif step_name == "tool_call":
                with trace_area:
                    st.markdown("---")
                    st.markdown(
                        f"<span class='trace-label trace-tool'>Tool Call: {step.get('tool_name')}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("Arguments")
                    st.code(json.dumps(step.get("arguments", {}), indent=2), language="json")
                    render_json_block("Tool result", step.get("result"))

            elif step_name == "local_fallback":
                status_area.warning("No model API key is configured. Showing local fallback results.")
                render_json_block("Fallback trace", step.get("trace", {}))

            elif step_name == "api_error":
                status_area.error("Model API call failed. Check the configured model, key, and endpoint region.")
                render_json_block(
                    "API error",
                    {
                        "model": step.get("model"),
                        "base_url": step.get("base_url"),
                        "error": step.get("error"),
                    },
                )

            elif step_name == "final":
                status_area.success(f"Search completed in {step.get('round', 0)} round(s).")
                final_answer = step.get("final_answer", "")
                final_area.markdown("### Final Answer")
                final_area.markdown(final_answer)

        return final_answer or "No answer was generated."
    except Exception as exc:
        status_area.error(f"Search failed: {exc}")
        return f"Search failed: {exc}"


init_session_state()
agent = get_agent()

st.markdown("<p class='main-header'>Agentic Search</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>A Qwen-compatible enterprise search demo with tool calling, multi-source retrieval, and explicit API error reporting.</p>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        "Question",
        placeholder="Example: What is the May cloud service expense trend?",
        label_visibility="collapsed",
        key="question_input",
    )
with col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

if search_button and question:
    st.session_state.is_searching = True
    answer = perform_search(agent, question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.is_searching = False

st.divider()
st.markdown("### Conversation History")

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("Enter a question above to start searching.")

with st.sidebar:
    st.markdown("### Data Sources")
    st.markdown(
        """
        <span class="source-badge">SQLite records</span>
        <span class="source-badge">Chroma vectors</span>
        <span class="source-badge">Whoosh keyword index</span>
        <span class="source-badge">Sample code repository</span>
        <span class="source-badge">Enterprise system simulator</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### System")
    st.markdown(
        """
        - Model: Qwen-compatible chat model
        - Max search rounds: 6
        - Fallback: local deterministic search
        - Demo data: synthetic English enterprise records
        """
    )

    st.markdown("### Status")
    st.write("Searching..." if st.session_state.is_searching else "Ready")

    if st.button("Clear history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
