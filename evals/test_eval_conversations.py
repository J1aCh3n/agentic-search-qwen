"""Offline tests for the conversation runner, using a stub agent instead of Qwen."""
import unittest

from evals.eval_conversations import load_conversations, run_conversation, summarize


class StubAgent:
    """Records (question, thread_id) and answers with fixed usage numbers."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def search_stream(self, question, thread_id=None):
        self.calls.append((question, thread_id))
        if question == self.fail_on:
            yield {"step": "api_error", "error": "boom"}
        else:
            yield {"step": "tool_call", "tool_name": "vector_search", "arguments": {"query": question}}
        context = 1000 * len(self.calls)
        yield {"step": "final", "round": 2, "final_answer": f"answer to {question} (employee handbook, per person)",
               "usage": {"model_calls": 2, "input_tokens": context * 2, "output_tokens": 50,
                         "context_tokens": context, "tool_result_tokens": 300}}


CONVERSATION = {
    "id": "conv-test",
    "category": "pronoun_follow_up",
    "type": "answerable",
    "turns": ["first question", "Is that per person?"],
    "question": "Is that per person?",
    "expected_source": "employee_handbook.md",
    "reference_answer": "Yes, per person.",
    "answer_keywords": [["per person"]],
}


class ConversationRunnerTests(unittest.TestCase):
    def test_turns_run_in_order_on_one_thread(self):
        agent = StubAgent()
        record = run_conversation(agent, CONVERSATION)
        self.assertEqual([q for q, _ in agent.calls], CONVERSATION["turns"])
        self.assertEqual(len({t for _, t in agent.calls}), 1)
        self.assertIsNotNone(agent.calls[0][1])
        self.assertEqual(record["answer"], record["turns"][-1]["answer"])

    def test_each_conversation_gets_its_own_thread(self):
        agent = StubAgent()
        run_conversation(agent, CONVERSATION)
        run_conversation(agent, CONVERSATION)
        self.assertEqual(len({t for _, t in agent.calls}), 2)

    def test_only_the_last_turn_is_graded(self):
        record = run_conversation(StubAgent(), CONVERSATION)
        self.assertTrue(record["passed"])
        self.assertEqual(record["question"], "Is that per person?")

    def test_turn_errors_are_recorded(self):
        record = run_conversation(StubAgent(fail_on="first question"), CONVERSATION)
        self.assertEqual(record["turns"][0]["error"], "boom")
        self.assertIsNone(record["turns"][1]["error"])

    def test_summary_tracks_context_growth_and_repeat_searches(self):
        records = [run_conversation(StubAgent(), CONVERSATION)]
        summary = summarize(records)
        self.assertEqual(summary["avg_context_tokens_by_turn"], {1: 1000, 2: 2000})
        self.assertEqual(summary["input_tokens"], 6000)
        self.assertEqual(summary["follow_ups_that_searched_again"], "1/1")

    def test_conversation_file_is_well_formed(self):
        conversations = load_conversations()
        self.assertEqual(len({c["id"] for c in conversations}), len(conversations))
        for conversation in conversations:
            with self.subTest(conversation=conversation["id"]):
                self.assertTrue(conversation["id"].startswith("conv-"))
                self.assertGreaterEqual(len(conversation["turns"]), 2)
                self.assertEqual(conversation["question"], conversation["turns"][-1])
                if conversation["type"] == "answerable":
                    self.assertTrue(conversation["answer_keywords"])


if __name__ == "__main__":
    unittest.main()
