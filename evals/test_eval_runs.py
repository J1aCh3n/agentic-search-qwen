"""Offline tests that a finished evaluation run actually writes its result file.

A run costs real tokens, so a crash after the last question but before the file is
written throws away everything: the answers cannot be graded again without paying
for them a second time. These tests replay both entry points with a stub agent.

    python -m unittest evals.test_eval_runs
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals import eval_agent, eval_conversations
from evals.eval_agent import build_agent

CASE = {
    "id": "test-case",
    "category": "direct",
    "type": "answerable",
    "question": "How many paid sick days are there?",
    "expected_source": "employee_handbook.md",
    "reference_answer": "Ten paid sick days.",
    "answer_keywords": [["ten"]],
}

CONVERSATION = {
    "id": "conv-test",
    "category": "pronoun_follow_up",
    "type": "answerable",
    "turns": ["How much is the meal allowance?", "Is that per person?"],
    "expected_source": "employee_handbook.md",
    "reference_answer": "Yes, per person.",
    "answer_keywords": [["per person"]],
}


class StubAgent:
    def search_stream(self, question, thread_id=None):
        yield {"step": "tool_call", "tool_name": "vector_search", "arguments": {"query": question},
               "result": "ten paid sick days, per person (employee handbook)"}
        yield {"step": "final", "round": 2, "route": "docs",
               "final_answer": "Ten paid sick days, per person (employee_handbook.md).",
               "usage": {"model_calls": 2, "input_tokens": 3000, "output_tokens": 100,
                         "context_tokens": 1500, "tool_result_tokens": 300, "router_tokens": 430}}


class ResultFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = Path(self.temp.name) / "results"

    def written(self) -> list[dict]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.results.glob("*.json"))]

    def run_main(self, module, cases_file: str, cases: list[dict], argv: list[str]):
        source = Path(self.temp.name) / cases_file
        source.write_text(json.dumps(cases), encoding="utf-8")
        # Both names are patched on the module under test: eval_conversations imported
        # RESULTS_DIR and build_agent from eval_agent, so it holds its own references.
        cases_name = "GOLDEN_SET" if module is eval_agent else "CONVERSATIONS"
        with patch.multiple(module, RESULTS_DIR=self.results, build_agent=lambda route: StubAgent(),
                            **{cases_name: source}), patch("sys.argv", ["eval", *argv]):
            module.main()

    def test_single_turn_run_saves_its_answers_with_and_without_the_router(self):
        self.run_main(eval_agent, "golden.json", [CASE], [])
        self.run_main(eval_agent, "golden.json", [CASE], ["--no-router"])
        files = self.written()
        self.assertEqual(len(files), 2)
        self.assertEqual({file["config"]["router"] for file in files}, {True, False})
        self.assertEqual(files[0]["cases"][0]["route"], "docs")

    def test_conversation_run_saves_its_turns_with_and_without_the_router(self):
        self.run_main(eval_conversations, "conversations.json", [CONVERSATION], [])
        self.run_main(eval_conversations, "conversations.json", [CONVERSATION], ["--no-router"])
        files = self.written()
        self.assertEqual(len(files), 2)
        self.assertEqual({file["config"]["router"] for file in files}, {True, False})
        self.assertEqual(len(files[0]["cases"][0]["turns"]), 2)


class BaselineAgentTests(unittest.TestCase):
    def test_no_router_sends_every_question_to_the_unknown_branch(self):
        with patch("evals.eval_agent.LangChainSearchAgent") as agent_class:
            build_agent(route=False)
            router = agent_class.call_args.kwargs["router"]
            self.assertEqual(router("any question")["category"], "unknown")
            self.assertEqual(router("any question")["input_tokens"], 0)

    def test_the_default_agent_builds_its_own_router(self):
        with patch("evals.eval_agent.LangChainSearchAgent") as agent_class:
            build_agent(route=True)
            self.assertEqual(agent_class.call_args.kwargs, {})


if __name__ == "__main__":
    unittest.main()
