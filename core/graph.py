"""The agent's graph: router -> model <-> tools.

    START -> router -> model --tool calls?--> tools --> model
                          |
                          +--no tool calls--> END

The router writes a category into the state; the model node uses it to decide
which tool descriptions to send. See core/router.py for why.
"""
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from core.router import ROUTES, tools_for


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # No reducer on these two, so each turn's router overwrites the previous turn's values.
    route: str
    router_usage: dict


def build_graph(model, tools, system_prompt, checkpointer, router: Callable | None = None):
    # One bound model per category, prepared once. bind_tools only records which tool
    # descriptions to send with the request, so holding four of them costs nothing.
    models_by_route = {name: model.bind_tools(tools_for(tools, name)) for name in ROUTES}
    builder = StateGraph(State)

    def classify_question(state: State) -> dict[str, Any]:
        # Without a router (the offline tests) every tool stays available, so the graph
        # behaves exactly as it did before this node existed.
        if router is None:
            return {"route": "unknown", "router_usage": {}}
        question = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        result = router(question.text)
        return {"route": result["category"], "router_usage": result}

    def graph_state(state:State):
        model_with_tools = models_by_route.get(state.get("route"), models_by_route["unknown"])
        messages = [
                SystemMessage(system_prompt),
                *state["messages"]
            ]
        respond = model_with_tools.invoke(messages)
        return {'messages' : [respond]}

    def route(state: State):
        # A rolled-back turn can leave no messages at all, so do not assume there is a last one.
        last_message = state["messages"][-1] if state["messages"] else None
        if getattr(last_message, "tool_calls", None):
            return 'tools'
        else: return END

    builder.add_node('router', classify_question)
    builder.add_node('model', graph_state)
    # The tool node keeps every tool: running a tool costs no tokens, and this way a
    # model that was shown a smaller group can still be answered by the same node.
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, "router")
    builder.add_edge("router", "model")
    builder.add_conditional_edges("model", route, ["tools", END])
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=checkpointer)
