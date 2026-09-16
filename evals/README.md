# Evaluation

Run every command from the project root with the project's Python environment.

## Files

| File | Role |
| --- | --- |
| `golden_set.json` | 23 questions: 15 answerable (direct, paraphrase, multi-fact, short), 4 unrelated, 4 near-miss |
| `eval_retrieval.py` | Retrieval layer, keyword check. Embeddings only, no chat model |
| `eval_retrieval_judge.py` | Retrieval layer, LLM judge decides whether each retrieved chunk answers the question |
| `eval_agent.py` | End-to-end: runs the full agent, keyword grading, saves every answer to `results/` |
| `judge_answers.py` | LLM judge for the answers saved by `eval_agent.py`, with a calibration mode |
| `judge_calibration.json` | 14 hand-labelled answers used to check the answer judge |
| `test_*.py` | Offline tests with fake verdicts. They test plumbing, not judge accuracy |

## Commands

```powershell
python -m evals.eval_retrieval               # fast, near-free
python -m evals.eval_retrieval_judge         # one chat call per question with surviving chunks
python -m evals.eval_agent                   # slow: 23 full agent runs
python -m evals.judge_answers --calibrate    # check the answer judge against human labels
python -m evals.judge_answers                # judge the latest eval_agent result
python -m unittest evals.test_eval_retrieval_judge evals.test_judge_answers
```

## Design

**Two layers.** Retrieval evaluation asks whether the right chunk was found; end-to-end evaluation asks whether the answer was right. Together they locate a failure:

| Retrieval | Answer | Diagnosis |
| --- | --- | --- |
| found | correct | working |
| missed | wrong | retrieval problem: chunking, embeddings, threshold |
| found | wrong | generation problem: prompt |
| missed | correct | check it: the agent rewrote the query, used another tool, or guessed |

**Question types.**

- `answerable` cases have `answer_keywords` for keyword grading and `reference_answer` for the judges.
- `unrelated` and `near_miss` cases have `refusal_reason`, which explains why the dataset cannot answer. It is kept out of `reference_answer` so the retrieval judge does not treat it as a fact to look for.
- Near-miss questions retrieve related chunks at distances similar to real answers (bereavement leave hits the sick leave section at 0.45). No threshold can reject them; the model has to.

**Grading is separate from generation.** `eval_agent.py` saves full answers. `judge_answers.py` grades those files, so changing the judge prompt does not require running the agent again.

**Keyword grading and the LLM judge run side by side.** Keyword grading is free and deterministic but misses paraphrases (`10 paid sick days`), new refusal wording (`do not contain`), negations (`not 85`), and refusals followed by a guess. The judge understands those but can also be wrong. `judge_answers.py` lists every case where the two disagree.

**Calibrate the judge before trusting it.** `judge_calibration.json` mixes real agent answers with answers written to fool keyword grading. The first judge prompt passed an answer with no citation, reasoning that a simple fact did not need one; calibration caught it and the rubric now lists every pass condition explicitly. Add a fixture whenever you find a new misjudgement. Leave out answers whose correct label is debatable.

**Structured output uses `method="function_calling"`.** LangChain sends the Pydantic verdict model as a tool schema and forces the model to call that tool; no function is executed. DashScope rejects `json_schema` and `json_mode` unless the prompt contains the word "json", so those modes break silently when someone edits the prompt. When the model replies without calling the tool, LangChain returns `None`, which both judges treat as an error.

**Judge errors are not failures.** An API error, a missing tool call, or an invalid verdict is recorded as `judge_error`, excluded from the metrics, and reported. The run continues.

## Limitations

- The calibration set has 14 answers. It catches obvious rubric problems, not rare ones.
- Qwen judges Qwen. Correlated mistakes and self-preference are possible; use a different judge model for a stronger check.
- Every configuration has been run once. The agent is not deterministic, so a one-case difference between runs can be noise.
- Unrelated questions are full sentences that the agent refuses without searching, so they do not measure the distance threshold. Short noisy queries such as `cake` would.
- Questions and retrieved text are sent to DashScope.
