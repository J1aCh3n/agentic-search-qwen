# Evaluation

Run every command from the project root with the project's Python environment.

## Files

| File | Role |
| --- | --- |
| `golden_set.json` | 23 questions: 15 answerable (direct, paraphrase, multi-fact, short), 4 unrelated, 4 near-miss |
| `eval_retrieval.py` | Retrieval layer, keyword check. Embeddings only, no chat model |
| `eval_retrieval_judge.py` | Retrieval layer, LLM judge decides whether each retrieved chunk answers the question |
| `eval_agent.py` | End-to-end: runs the full agent, keyword grading, saves every answer to `results/` |
| `golden_conversations.json` | 8 multi-turn conversations (17 turns): pronoun and ellipsis follow-ups, follow-ups asking for a new fact, a three-turn chain, a topic switch, and near-miss and unrelated follow-ups |
| `eval_conversations.py` | Multi-turn: one thread per conversation, grades the last turn, records per-turn tokens and whether follow-ups searched again |
| `judge_answers.py` | LLM judge for the answers saved by `eval_agent.py`, with a calibration mode |
| `judge_calibration.json` | 21 hand-labelled answers used to check the answer judge |
| `judge_model.py` | Judge model setup and retry, shared by both judges |
| `test_*.py` | Offline tests with fake verdicts. They test plumbing, not judge accuracy |

## Commands

```powershell
python -m evals.eval_retrieval               # fast, near-free
python -m evals.eval_retrieval_judge         # one chat call per question with surviving chunks
python -m evals.eval_agent                   # slow: 23 full agent runs
python -m evals.eval_conversations --limit 1 # one conversation first, to check token cost
python -m evals.eval_conversations           # all conversations; grade with judge_answers PATH
python -m evals.judge_answers --calibrate    # check the answer judge against human labels
python -m evals.judge_answers                # judge the latest eval_agent result
python -m unittest tests.test_lc_agent tests.test_tools evals.test_eval_retrieval_judge evals.test_judge_answers evals.test_eval_conversations
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

**The answer judge checks claims against the source text, not only the reference.** A reference answer lists the key facts, but an agent answer often adds other true details. A judge that saw only the reference had to guess whether those details were real: it failed correct answers that added real rules, and graded one unchanged answer differently on two runs. The judge now receives every text source the agent can search (`data/docs/`, the vector collections, the keyword indexes, and the wiki system, about 2,500 tokens) and fails any claim about the company that no source states. Giving it only `data/docs/` was not enough: an answer quoting the two-day remote work rule from the keyword policy index was called a hallucination. Structured records (SQLite and the HR, finance, and project systems) are not included, so an answer that cites them would show up as a disagreement for a human to check.

**Calibrate the judge before trusting it.** `judge_calibration.json` mixes real agent answers with answers written to fool keyword grading. The first judge prompt passed an answer with no citation, reasoning that a simple fact did not need one; calibration caught it and the rubric now lists every pass condition explicitly. Add a fixture whenever you find a new misjudgement. Leave out answers whose correct label is debatable.

**The judge model is separate from the agent model.** `DASHSCOPE_JUDGE_MODEL` defaults to `qwen3.7-plus-2026-05-26` while the agent uses `qwen3.6-plus-2026-04-02`. Both are dated snapshots. A dated snapshot keeps scores comparable, because an alias such as `qwen3.7-plus` can be moved to a newer model. A different model generation reduces self-preference, and DashScope free quota is per model, so judging does not spend the agent's quota. Calibration results for the models tried:

| Judge model | Agrees with labels | Notes |
| --- | --- | --- |
| `qwen3.7-plus-2026-05-26` | 14/14, two runs; 21/21 on the current set with source documents, two runs | about 35 s and 990 tokens per call before source documents were added |
| `qwen3.6-plus` | 14/14, two runs | the agent's own model |
| `qwen3.6-flash` | 13/14 | one call returned no verdict; wrote about 600 output tokens per call |
| `qwen3.7-flash` | 13/14 | passed an uncited answer, claiming it cited employee_handbook.md |

A failure that produces no verdict is visible as `judge_error`; a wrong verdict is silent. Prefer the model whose failures are visible. Re-run `--calibrate` whenever the judge model or prompt changes.

**Structured output uses `method="function_calling"` with thinking disabled.** LangChain sends the Pydantic verdict model as a tool schema and forces the model to call that tool; no function is executed. DashScope rejects `json_schema` and `json_mode` unless the prompt contains the word "json", so those modes break when someone edits the prompt. Forced tool calls are also rejected in thinking mode, which `qwen3.7-plus` enables by default, so `judge_model.py` sends `enable_thinking: false`. When the model replies without calling the tool, LangChain returns `None`, which both judges treat as an error.

**Judge errors are retried once, then excluded.** An API error such as a 429, a missing tool call, or an invalid verdict is retried after three seconds. If it fails again it is recorded as `judge_error`, excluded from the metrics, and reported. The run continues.

## Limitations

- The calibration set has 21 answers. It catches obvious rubric problems, not rare ones.
- Failure types for borderline answers can still vary between runs. `fake-conv-per-person-wrong-referent` fails in every run, but was labelled `missing_fact` in one run and `wrong_fact` in the next.
- The judge does not see structured records, so answers built on SQLite or the enterprise systems cannot be checked for unsupported details.
- The judge and the agent are different generations of the same Qwen family, so correlated mistakes are still possible. A judge from another model family would be a stronger check.
- Every configuration has been run once. The agent is not deterministic, so a one-case difference between runs can be noise.
- Unrelated questions are full sentences that the agent refuses without searching, so they do not measure the distance threshold. Short noisy queries such as `cake` would.
- Questions and retrieved text are sent to DashScope.
