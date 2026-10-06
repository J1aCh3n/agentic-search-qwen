"""Offline tests for conversation memory in LangChainSearchAgent.

A fake model replays scripted replies and records what it was sent, so these tests
check memory, rollback, and usage accounting without calling DashScope.

    python -m unittest tests.test_lc_agent
"""
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from core import router
from core.lc_agent import LangChainSearchAgent


class FakeModel(GenericFakeChatModel):
    """Replays scripted replies and records the messages sent on each call."""

    received: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@tool
def lookup(query: str) -> str:
    """Return a retrieved passage."""
    return "The daily meal allowance is 85 Canadian dollars per person."


@tool
def broken(query: str) -> str:
    """Always fails."""
    raise RuntimeError("tool crashed")


def reply(text, input_tokens=0, output_tokens=0):
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens},
    )


def tool_call(name, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"query": "x"}, "id": call_id}])


def make_agent(*replies, tools=(lookup, broken), router=None):
    model = FakeModel(messages=iter(replies))
    return LangChainSearchAgent(model=model, tools=list(tools), router=router), model


def thread_messages(agent, thread_id):
    state = agent.agent.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []), state.next


def humans(messages):
    return [m.content for m in messages if isinstance(m, HumanMessage)]


@patch("core.lc_agent.has_api_key", return_value=True)
class ConversationMemoryTests(unittest.TestCase):
    def test_same_thread_sends_earlier_turns_to_the_model(self, _key):
        agent, model = make_agent(reply("answer 1"), reply("answer 2"))
        agent.search("question 1", thread_id="t")
        agent.search("question 2", thread_id="t")
        self.assertEqual(humans(model.received[1]), ["question 1", "question 2"])

    def test_different_threads_do_not_share_history(self, _key):
        agent, model = make_agent(reply("answer a"), reply("answer b"))
        agent.search("question a", thread_id="a")
        agent.search("question b", thread_id="b")
        self.assertEqual(humans(model.received[1]), ["question b"])

    def test_no_thread_id_starts_a_fresh_conversation_each_call(self, _key):
        agent, model = make_agent(reply("answer 1"), reply("answer 2"))
        agent.search("question 1")
        agent.search("question 2")
        self.assertEqual(humans(model.received[1]), ["question 2"])

    def test_tool_error_rolls_back_the_turn_and_next_turn_works(self, _key):
        agent, model = make_agent(reply("answer 1"), tool_call("broken", "c1"), reply("answer 3"))
        agent.search("question 1", thread_id="t")

        steps = list(agent.search_stream("question 2", thread_id="t"))
        self.assertEqual(steps[-1]["step"], "final")
        self.assertIn("not saved to the conversation", steps[-1]["final_answer"])
        messages, pending = thread_messages(agent, "t")
        self.assertEqual(humans(messages), ["question 1"])
        self.assertEqual(pending, ())

        self.assertEqual(agent.search("question 3", thread_id="t"), "answer 3")
        self.assertEqual(humans(model.received[-1]), ["question 1", "question 3"])

    def test_closing_the_stream_mid_turn_rolls_back_the_turn(self, _key):
        agent, _ = make_agent(reply("answer 1"), tool_call("lookup", "c1"), reply("never reached"))
        agent.search("question 1", thread_id="t")

        stream = agent.search_stream("question 2", thread_id="t")
        for step in stream:
            if step["step"] == "tool_call":
                stream.close()
                break

        messages, pending = thread_messages(agent, "t")
        self.assertEqual(humans(messages), ["question 1"])
        self.assertFalse(any(isinstance(m, ToolMessage) for m in messages))
        self.assertEqual(pending, ())

    def test_hitting_the_round_limit_rolls_back_the_turn(self, _key):
        endless = [tool_call("lookup", f"c{i}") for i in range(10)]
        agent, _ = make_agent(reply("answer 1"), *endless)
        agent.search("question 1", thread_id="t")

        with patch("core.lc_agent.MAX_SEARCH_ROUNDS", 1):
            answer = agent.search("question 2", thread_id="t")

        self.assertIn("Stopped after", answer)
        self.assertIn("not saved to the conversation", answer)
        messages, pending = thread_messages(agent, "t")
        self.assertEqual(humans(messages), ["question 1"])
        self.assertEqual(pending, ())

    def test_exactly_the_round_limit_still_reaches_an_answer(self, _key):
        # The limit counts tool rounds. The router step and the final answer must not use one up.
        agent, _ = make_agent(tool_call("lookup", "c1"), tool_call("lookup", "c2"), reply("done"))
        with patch("core.lc_agent.MAX_SEARCH_ROUNDS", 2):
            self.assertEqual(agent.search("question", thread_id="t"), "done")

    def test_a_failed_first_turn_leaves_an_empty_thread(self, _key):
        # Rollback removes every message, so the next turn starts from nothing.
        agent, _ = make_agent(tool_call("broken", "c1"), reply("answer 2"))
        list(agent.search_stream("question 1", thread_id="t"))
        messages, pending = thread_messages(agent, "t")
        self.assertEqual(messages, [])
        self.assertEqual(pending, ())
        self.assertEqual(agent.search("question 2", thread_id="t"), "answer 2")

    def test_usage_sums_model_calls_and_keeps_last_context_size(self, _key):
        agent, _ = make_agent(
            AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "c1"}],
                      usage_metadata={"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020}),
            reply("85 CAD per person", input_tokens=1300, output_tokens=40),
        )
        final = list(agent.search_stream("meal allowance?", thread_id="t"))[-1]
        usage = final["usage"]
        self.assertEqual(usage["model_calls"], 2)
        self.assertEqual(usage["input_tokens"], 2300)
        self.assertEqual(usage["output_tokens"], 60)
        self.assertEqual(usage["context_tokens"], 1300)
        self.assertGreater(usage["tool_result_tokens"], 0)

    def test_missing_usage_metadata_counts_as_zero(self, _key):
        agent, _ = make_agent(AIMessage(content="answer"))
        final = list(agent.search_stream("question"))[-1]
        self.assertEqual(final["usage"]["model_calls"], 1)
        self.assertEqual(final["usage"]["input_tokens"], 0)


class GraphStructureTests(unittest.TestCase):
    def test_the_graph_has_the_nodes_and_edges_the_stream_reader_expects(self):
        # search_stream() reads updates by node name and counts two graph steps per round,
        # so the shape of the graph is part of the contract, not just an implementation detail.
        agent, _ = make_agent(reply("answer"))
        graph = agent.agent.get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertEqual(set(graph.nodes), {"__start__", "router", "model", "tools", "__end__"})
        self.assertEqual(
            edges,
            {("__start__", "router"), ("router", "model"), ("model", "tools"),
             ("model", "__end__"), ("tools", "model")},
        )


class RouterTests(unittest.TestCase):
    def test_each_category_selects_its_own_tools_and_unknown_keeps_all(self):
        tools = [lookup, broken]
        with patch.dict(router.TOOL_GROUPS, {"docs": ("lookup",), "code": ("broken",)}, clear=True):
            self.assertEqual(router.tools_for(tools, "docs"), [lookup])
            self.assertEqual(router.tools_for(tools, "code"), [broken])
            self.assertEqual(router.tools_for(tools, "unknown"), tools)

    @patch("core.lc_agent.has_api_key", return_value=True)
    def test_the_turn_reports_the_route_and_counts_the_routing_call(self, _key):
        def fake_router(question):
            return {"category": "code", "input_tokens": 90, "output_tokens": 4}

        agent, _ = make_agent(reply("answer", input_tokens=1000, output_tokens=20), router=fake_router)
        final = list(agent.search_stream("where is the config?"))[-1]
        self.assertEqual(final["route"], "code")
        self.assertEqual(final["usage"]["router_tokens"], 94)
        self.assertEqual(final["usage"]["input_tokens"], 1090)
        self.assertEqual(final["usage"]["output_tokens"], 24)
        # The routing call is not a search round, so it does not count as a model call.
        self.assertEqual(final["usage"]["model_calls"], 1)

    def test_a_failed_routing_call_falls_back_to_every_tool(self):
        class Boom:
            def with_structured_output(self, *a, **k):
                return self

            def invoke(self, *a, **k):
                raise RuntimeError("router is down")

        result = router.create_router(Boom())("any question")
        self.assertEqual(result["category"], "unknown")


class FallbackTests(unittest.TestCase):
    @patch("core.lc_agent.has_api_key", return_value=False)
    @patch("core.lc_agent.LocalFallbackSearch")
    def test_no_api_key_uses_local_fallback_without_the_model(self, fallback_class, _key):
        fallback_class.return_value.search.return_value = "fallback answer"
        agent, model = make_agent(reply("never used"))
        steps = list(agent.search_stream("question"))
        self.assertEqual([s["step"] for s in steps], ["init", "local_fallback", "final"])
        self.assertEqual(steps[-1]["final_answer"], "fallback answer")
        self.assertEqual(model.received, [])

    def test_langchain_agent_does_not_depend_on_the_original_agent(self):
        source = (Path(__file__).resolve().parents[1] / "core" / "lc_agent.py").read_text(encoding="utf-8")
        self.assertNotIn("core.agent", source)


if __name__ == "__main__":
    unittest.main()
