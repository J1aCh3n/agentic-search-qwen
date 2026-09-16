"""Offline tests for the answer judge: fake verdicts test plumbing, NOT Qwen's accuracy.

Judge accuracy is checked against human labels with:
    python -m evals.judge_answers --calibrate
"""
import json
import unittest
from unittest.mock import Mock

from evals.eval_retrieval import GOLDEN_SET
from evals.judge_answers import CALIBRATION_SET, Verdict, judge_answer

ANSWERABLE = {
    "id": "meal",
    "type": "answerable",
    "question": "What is the daily meal allowance?",
    "expected_source": "employee_handbook.md",
    "reference_answer": "The daily meal allowance is 85 Canadian dollars per person.",
    "answer_keywords": [["85"]],
}
NEAR_MISS = {
    "id": "trap",
    "type": "near_miss",
    "question": "How many days of bereavement leave do employees get?",
    "refusal_reason": "The handbook does not define bereavement leave.",
}


def sent_payload(judge: Mock) -> dict:
    return json.loads(judge.invoke.call_args.args[0][1][1])


class AnswerJudgeTests(unittest.TestCase):
    def test_answerable_payload_uses_reference_and_source(self):
        judge = Mock()
        judge.invoke.return_value = Verdict(reasoning="ok", passed=True, failure_type="none")
        judge_answer(judge, ANSWERABLE, "85 CAD per day (employee handbook).")
        payload = sent_payload(judge)
        self.assertEqual(payload["reference"], ANSWERABLE["reference_answer"])
        self.assertEqual(payload["expected_source"], "employee_handbook.md")
        self.assertNotIn("type", payload)

    def test_refusal_payload_uses_refusal_reason_and_null_source(self):
        judge = Mock()
        judge.invoke.return_value = Verdict(reasoning="ok", passed=True, failure_type="none")
        judge_answer(judge, NEAR_MISS, "The dataset does not contain this.")
        payload = sent_payload(judge)
        self.assertEqual(payload["reference"], NEAR_MISS["refusal_reason"])
        self.assertIsNone(payload["expected_source"])

    def test_errors_become_judge_error_instead_of_a_verdict(self):
        cases = {
            "api failure": RuntimeError("API unavailable"),
            "no tool call": None,
            "inconsistent": Verdict(reasoning="x", passed=True, failure_type="wrong_fact"),
        }
        for label, outcome in cases.items():
            with self.subTest(label):
                judge = Mock()
                if isinstance(outcome, Exception):
                    judge.invoke.side_effect = outcome
                else:
                    judge.invoke.return_value = outcome
                result = judge_answer(judge, ANSWERABLE, "answer")
                self.assertIn("judge_error", result)
                self.assertNotIn("passed", result)

    def test_golden_set_has_a_reference_for_every_case(self):
        for case in json.loads(GOLDEN_SET.read_text(encoding="utf-8")):
            field = "reference_answer" if case["type"] == "answerable" else "refusal_reason"
            with self.subTest(case=case["id"]):
                self.assertTrue(case.get(field, "").strip())

    def test_calibration_fixtures_point_at_real_cases_with_consistent_labels(self):
        case_ids = {case["id"] for case in json.loads(GOLDEN_SET.read_text(encoding="utf-8"))}
        for fixture in json.loads(CALIBRATION_SET.read_text(encoding="utf-8")):
            with self.subTest(fixture=fixture["id"]):
                self.assertIn(fixture["case_id"], case_ids)
                self.assertEqual(fixture["expected_pass"], fixture["expected_failure_type"] == "none")


if __name__ == "__main__":
    unittest.main()
