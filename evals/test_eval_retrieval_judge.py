"""Offline tests: fake judge verdicts test plumbing, NOT Qwen's accuracy."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from config import JUDGE_MODEL
from evals.eval_retrieval_judge import (
    GOLDEN_SET,
    ChunkVerdict,
    RetrievalVerdict,
    create_judge,
    evaluate_case,
    find_rank,
    judge_hits,
    main,
    summarize,
)


def make_hit(source="employee_handbook.md"):
    return {
        "id": "chunk_0",
        "content": "The daily meal allowance is 85 Canadian dollars per person.",
        "distance": 0.4,
        "metadata": {"source": source},
    }


def make_verdict(*sufficient):
    return RetrievalVerdict(chunks=[
        ChunkVerdict(rank=rank, sufficient=value, reason="Test judgment")
        for rank, value in enumerate(sufficient, start=1)
    ])


class RetrievalJudgeTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "meal",
            "type": "answerable",
            "question": "What is the daily meal allowance?",
            "expected_source": "employee_handbook.md",
            "reference_answer": "85 Canadian dollars per person per day.",
            "answer_keywords": [["85"]],
        }

    def test_first_correct_chunk_needs_source_and_sufficiency(self):
        hits = [make_hit("wrong.md"), make_hit(), make_hit()]
        self.assertEqual(find_rank(hits, self.case, make_verdict(True, True, True)), 2)
        self.assertEqual(find_rank(hits, self.case, make_verdict(True, False, True)), 3)
        self.assertIsNone(find_rank(hits, self.case, make_verdict(True, False, False)))

    def test_empty_hits_do_not_call_judge(self):
        judge = Mock()
        self.assertEqual(judge_hits([], self.case, judge).chunks, [])
        judge.invoke.assert_not_called()

    def test_one_call_contains_reference_and_all_chunks_not_keywords(self):
        judge = Mock()
        judge.invoke.return_value = make_verdict(True, False)
        judge_hits([make_hit(), make_hit()], self.case, judge)
        judge.invoke.assert_called_once()
        messages = judge.invoke.call_args.args[0]
        payload = json.loads(messages[1][1])
        self.assertEqual(payload["reference_answer"], self.case["reference_answer"])
        self.assertEqual([c["rank"] for c in payload["chunks"]], [1, 2])
        self.assertNotIn("answer_keywords", payload)
        self.assertNotIn("type", payload)

    def test_negative_case_has_no_reference_or_golden_label_in_prompt(self):
        judge = Mock()
        judge.invoke.return_value = make_verdict(False)
        judge_hits([make_hit()], {"id": "negative", "question": "Weather?"}, judge)
        payload = json.loads(judge.invoke.call_args.args[0][1][1])
        self.assertIsNone(payload["reference_answer"])

    def test_bad_rank_coverage_aborts_instead_of_scoring(self):
        for ranks in ([1], [1, 1], [2, 1], [1, 3]):
            with self.subTest(ranks=ranks):
                judge = Mock()
                judge.invoke.return_value = RetrievalVerdict(chunks=[
                    ChunkVerdict(rank=r, sufficient=True, reason="Test") for r in ranks
                ])
                with self.assertRaisesRegex(ValueError, "chunk ranks"):
                    judge_hits([make_hit(), make_hit()], self.case, judge)

    def test_schema_rejects_coercion_and_unknown_keys(self):
        invalid = [
            {"rank": "1", "sufficient": True, "reason": "Test"},
            {"rank": 1, "sufficient": "false", "reason": "Test"},
            {"rank": True, "sufficient": True, "reason": "Test"},
            {"rank": 0, "sufficient": True, "reason": "Test"},
            {"rank": 1, "sufficient": True, "reason": ""},
            {"rank": 1, "sufficient": True, "reason": "Test", "extra": 0},
        ]
        for chunk in invalid:
            with self.subTest(chunk=chunk), self.assertRaises(ValidationError):
                ChunkVerdict.model_validate(chunk)

    def test_api_errors_are_not_converted_to_failed_cases(self):
        judge = Mock()
        judge.invoke.side_effect = RuntimeError("API unavailable")
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            judge_hits([make_hit()], self.case, judge)

    def test_missing_tool_call_is_an_error_not_an_empty_verdict(self):
        # PydanticToolsParser returns None when the model answers without calling the tool.
        judge = Mock()
        judge.invoke.return_value = None
        with self.assertRaisesRegex(ValueError, "did not call the verdict tool"):
            judge_hits([make_hit()], self.case, judge)

    @patch("evals.judge_model.time.sleep")
    def test_judge_errors_are_retried_once_then_recorded_and_excluded(self, _sleep):
        vector_db = SimpleNamespace(search=Mock(return_value=json.dumps({"results": [make_hit()]})))
        judge = Mock()
        judge.invoke.side_effect = RuntimeError("API unavailable")
        failed = evaluate_case(vector_db, judge, self.case)
        self.assertEqual(judge.invoke.call_count, 2)
        self.assertIn("API unavailable", failed["judge_error"])
        self.assertIsNone(failed["rank"])

        passed = {"type": "answerable", "rank": 1}
        self.assertEqual(summarize([passed, failed])["Hit@1"], 1.0)

    def test_positive_case_records_chunks_reasons_and_rank(self):
        vector_db = SimpleNamespace(search=Mock(return_value=json.dumps({"results": [make_hit()]})))
        judge = Mock()
        judge.invoke.return_value = make_verdict(True)
        record = evaluate_case(vector_db, judge, self.case)
        self.assertEqual(record["rank"], 1)
        self.assertEqual(record["top_distance"], 0.4)
        self.assertEqual(record["hits"], [make_hit()])
        self.assertEqual(record["judgments"][0]["reason"], "Test judgment")
        vector_db.search.assert_called_once_with(
            self.case["question"], collection_name="handbook", n_results=3,
        )

    def test_semantic_rejection_is_not_threshold_rejection(self):
        vector_db = SimpleNamespace(search=Mock(return_value=json.dumps({"results": [make_hit()]})))
        judge = Mock()
        judge.invoke.return_value = make_verdict(False)
        case = {"id": "trap", "type": "near_miss", "question": "Bereavement leave?"}
        record = evaluate_case(vector_db, judge, case)
        self.assertFalse(record["threshold_blocked"])
        self.assertTrue(record["no_answer_evidence"])
        self.assertIsNone(record["rank"])

        judge.invoke.return_value = make_verdict(True)
        record = evaluate_case(vector_db, judge, case)
        self.assertFalse(record["no_answer_evidence"])

    def test_empty_negative_is_blocked_without_model_call(self):
        vector_db = SimpleNamespace(search=Mock(return_value='{"results": [], "message": "No content"}'))
        judge = Mock()
        record = evaluate_case(vector_db, judge, {
            "id": "weather", "type": "unrelated", "question": "Weather?",
        })
        self.assertTrue(record["threshold_blocked"])
        self.assertTrue(record["no_answer_evidence"])
        self.assertIsNone(record["top_distance"])
        judge.invoke.assert_not_called()

    def test_metric_denominators_and_empty_groups(self):
        records = [
            {"type": "answerable", "rank": rank} for rank in (1, 2, None)
        ] + [
            {"type": "unrelated", "threshold_blocked": True, "no_answer_evidence": True},
            {"type": "near_miss", "threshold_blocked": False, "no_answer_evidence": True},
        ]
        summary = summarize(records)
        self.assertAlmostEqual(summary["Hit@1"], 1 / 3)
        self.assertAlmostEqual(summary["Hit@3"], 2 / 3)
        self.assertEqual(summary["MRR@3"], 0.5)
        self.assertEqual(summary["unrelated_threshold_blocked"], 1.0)
        self.assertEqual(summary["near_miss_threshold_blocked"], 0.0)
        self.assertEqual(summary["near_miss_no_answer_evidence"], 1.0)
        self.assertTrue(all(value is None for value in summarize([]).values()))

    def test_function_calling_binding_and_real_langchain_parser_offline(self):
        model = ChatOpenAI(model="qwen-test", api_key="offline-test-key")
        with patch("evals.judge_model.get_chat_model", return_value=model) as factory:
            judge = create_judge()
        factory.assert_called_once_with(temperature=0, model=JUDGE_MODEL)
        bound = judge.steps[0].kwargs
        self.assertEqual(bound["tool_choice"]["function"]["name"], "RetrievalVerdict")
        self.assertEqual(bound["extra_body"], {"enable_thinking": False})

        def tool_call(args):
            return AIMessage(content="", tool_calls=[{"name": "RetrievalVerdict", "id": "call_1", "args": args}])

        parsed = judge.steps[-1].invoke(tool_call({
            "chunks": [{"rank": 1, "sufficient": False, "reason": "No evidence"}],
        }))
        self.assertFalse(parsed.chunks[0].sufficient)
        with self.assertRaises(ValidationError):
            judge.steps[-1].invoke(tool_call({
                "chunks": [{"rank": 1, "sufficient": "false", "reason": "No evidence"}],
            }))

    def test_missing_or_empty_collection_aborts_before_judge_creation(self):
        for collections, count, message in (
            ([], 0, "Seed the handbook"),
            (["handbook"], 0, "collection is empty"),
        ):
            with self.subTest(collections=collections):
                vector_db = Mock()
                vector_db.list_collections.return_value = collections
                vector_db.client.get_collection.return_value.count.return_value = count
                with (
                    patch("sys.argv", ["eval_retrieval_judge"]),
                    patch("evals.eval_retrieval_judge.has_api_key", return_value=True),
                    patch("evals.eval_retrieval_judge.VectorDBEngine", return_value=vector_db),
                    patch("evals.eval_retrieval_judge.create_judge") as factory,
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        main()
                    factory.assert_not_called()
                    vector_db.search.assert_not_called()

    def test_all_answerable_cases_have_reference_answers(self):
        cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
        for case in cases:
            if case["type"] == "answerable":
                with self.subTest(case=case["id"]):
                    self.assertTrue(case["reference_answer"].strip())


if __name__ == "__main__":
    unittest.main()
