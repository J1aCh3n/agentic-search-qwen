
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(model, tools, system_prompt, checkpointer):
    model_with_tools = model.bind_tools(tools)
    builder = StateGraph(State)
    def graph_state(state:State):
        messages = [
                SystemMessage(system_prompt),
                *state["messages"]
            ]
        respond = model_with_tools.invoke(messages)
        return {'messages' : [respond]}
    
    def route(state: State):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return 'tools'
        else: return END

    builder.add_node('model', graph_state)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, ["tools", END])
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=checkpointer)