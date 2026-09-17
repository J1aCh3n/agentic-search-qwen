# Agentic Search Qwen

Agentic Search Qwen is a Streamlit demo of an enterprise search assistant that uses a Qwen-compatible chat model to choose tools, retrieve evidence, and answer questions over multiple synthetic data sources.

The agent is built with LangChain (`create_agent`), which runs the tool-calling loop on top of LangGraph. Retrieval is real RAG: documents are chunked, embedded with a DashScope embedding model, and filtered by a distance threshold so that irrelevant results are reported as missing evidence instead of being answered from general knowledge.

The repository has been converted to an English-only public demo. The data is synthetic and is intended for portfolio review, not production use.

## Features

- Streamlit interface for interactive search and trace inspection
- CLI entry point for terminal-based testing
- LangChain tool-calling agent with a Qwen-compatible chat model
- SQLite search over structured enterprise records
- Chroma vector search over synthetic company documents, engineering documents, and Markdown source documents
- Real semantic embeddings from DashScope (`text-embedding-v4`), with a hash-based offline fallback kept for comparison
- Markdown documents in `data/docs/` are loaded, split into overlapping chunks, and stored with `source` and `chunk_index` metadata
- Distance threshold on vector results, so an unrelated query returns an explicit "no relevant content" message
- Source-aware answers: the model is instructed to cite the file, table, collection, index, or system a fact came from
- Whoosh keyword search over policy and technical article indexes
- Sample code repository search
- Simulated enterprise systems for HR, finance, project management, and wiki queries
- Explicit API error reporting when model credentials, region, or entitlement are wrong
- Local fallback search only when no model API key is configured
- Conversation memory: a LangGraph checkpointer keeps each conversation under its own `thread_id`, so follow-up questions such as "Is that per person?" work; a turn that fails midway is rolled back so it cannot corrupt later turns
- Chat interface with a collapsible search trace per answer, showing tool calls, retrieved text, and token usage

## Project Structure

```text
.
+-- app.py                  # Streamlit UI
+-- main.py                 # CLI entry point
+-- config.py               # Runtime paths, model, embedding, and threshold configuration
+-- seed_data_large.py      # English synthetic data generator, including document chunking
+-- core/
|   +-- lc_agent.py         # LangChain agent (create_agent) used by the app and CLI
|   +-- lc_tools.py         # Tools defined with the @tool decorator
|   +-- lc_llm.py           # ChatOpenAI client pointed at the DashScope endpoint
|   +-- prompts.py          # System prompt shared by both agents and the UI
|   +-- local_search.py     # Rule-based fallback search used when no API key is configured
|   +-- agent.py            # Earlier hand-written agent, kept for comparison
|   +-- llm.py              # Earlier hand-written OpenAI SDK wrapper, kept for comparison
|   +-- logger.py           # Search trace logging
+-- engines/
|   +-- database.py         # SQLite search
|   +-- vector_db.py        # Chroma vector search, embedding functions, distance filtering
|   +-- keyword_search.py   # Whoosh keyword search
|   +-- code_search.py      # Sample repository search
|   +-- enterprise_sdk.py   # Simulated enterprise systems
+-- evals/                  # Golden set, retrieval and end-to-end evaluation, LLM judges (see evals/README.md)
+-- data/
    +-- docs/               # Markdown source documents that are chunked into the vector store
    +-- code_repo/          # Generated English sample repository
```

Generated runtime stores under `data/db/`, `data/vector_db/`, `data/keyword_index/`, and `data/logs/` are ignored by Git. They are rebuilt by the seed script. `data/docs/` is source data and is committed.

## Retrieval Pipeline

```text
data/docs/*.md
  -> RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
  -> DashScope embeddings (text-embedding-v4, 1024 dimensions)
  -> Chroma collection "handbook" with source and chunk_index metadata
  -> vector_search tool, results above VECTOR_DISTANCE_THRESHOLD are dropped
  -> agent answers with citations, or reports that the dataset has no answer
```

Notes on the defaults:

- `chunk_size=500` was chosen by comparing 100, 500, and 2000 on these documents. At 100 characters sentences were split mid-fact; at 2000 characters the distance between relevant and irrelevant results collapsed.
- `VECTOR_DISTANCE_THRESHOLD=0.70` keeps the one-word query `promotion` (0.650 to the correct chunk) while still dropping one-word noise such as `cake` (0.706) and `pizza` (0.734). An earlier value of 0.65 dropped the correct promotion chunk. The margin is thin, so the threshold only filters obvious noise: near-miss questions such as bereavement leave retrieve related chunks at 0.45, and the model, not the threshold, has to recognise that they do not answer the question. Re-calibrate with `evals/` if the embedding model or the documents change.

## Setup

Use Python 3.10 or 3.11. Some ChromaDB releases are less predictable on newer Python versions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Conda is also fine if you prefer it:

```powershell
conda create -n agentic-search-qwen python=3.11
conda activate agentic-search-qwen
pip install -r requirements.txt
```

Edit `.env` and add your DashScope API key:

```text
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.6-plus-2026-04-02
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_JUDGE_MODEL=qwen3.7-plus-2026-05-26
VECTOR_DISTANCE_THRESHOLD=0.70
```

Use the DashScope endpoint that matches the region where the API key was created. This project defaults to the international endpoint because a China-region endpoint can reject an international key with authentication or entitlement errors.

The same API key is used for chat completion, embeddings, and the evaluation judge. DashScope free quota is tracked per model, so the judge runs on its own model and does not use the agent model's quota. Seeding calls the embedding endpoint once per chunk, in batches of ten, which is the DashScope batch limit.

The app can still demonstrate local fallback behavior without an API key, but vector search needs the embedding endpoint. If an API key is present, model/API errors are shown directly instead of being hidden by fallback search.

## Run

Streamlit:

```powershell
streamlit run app.py
```

CLI:

```powershell
python main.py
```

The CLI keeps one conversation for the whole session; type `new` to start another. In Streamlit, each browser tab has its own conversation and the sidebar button starts a new one.

Offline tests (no API calls; a fake model stands in for Qwen):

```powershell
python -m unittest tests.test_lc_agent evals.test_eval_retrieval_judge evals.test_judge_answers evals.test_eval_conversations
```

Force-regenerate demo data:

```powershell
python seed_data_large.py
```

Regenerate after changing the embedding model or the chunking settings:

```powershell
Remove-Item -Recurse -Force data\vector_db
python seed_data_large.py
```

Vectors from one embedding model cannot be searched with another, and `add_documents` skips ids that already exist, so the store has to be deleted before it is rebuilt.

## Evaluation

`evals/` holds a 23-question golden set and evaluates retrieval and final answers separately, with keyword grading and an LLM judge side by side. See [evals/README.md](evals/README.md) for commands and design.

Baseline at `VECTOR_DISTANCE_THRESHOLD=0.70` with `qwen3.6-plus` and `text-embedding-v4`, one run each:

| Evaluation | Result |
| --- | --- |
| Retrieval, 15 answerable questions | Hit@1 100%, MRR@3 1.00 |
| Retrieval, 4 unrelated questions | 4/4 blocked by the threshold |
| Retrieval, 4 near-miss questions | 0/4 blocked by the threshold, 4/4 judged as not answering the question |
| End-to-end answers (LLM judge) | 23/23 |
| Regression after switching the agent to `qwen3.6-plus-2026-04-02` and adding memory | keyword grading 23/23, LLM judge 23/23; 130,185 input and 9,730 output tokens |
| Answer judge vs. 21 hand-labelled answers | `qwen3.7-plus-2026-05-26` 21/21 in two runs, keyword grading 14/21 |

The near-miss row is the main finding: related chunks pass any usable threshold, so refusing those questions depends on the model and the system prompt.

Multi-turn baseline with conversation memory, agent `qwen3.6-plus-2026-04-02`, one run:

| Evaluation | Result |
| --- | --- |
| 8 conversations, last turn graded (LLM judge) | 8/8 |
| Follow-up turns that searched again | 2/9 (a topic switch and a near-miss); the rest answered from retrieval results kept in memory |
| Tokens for 17 turns | 64,101 input, 3,934 output, about 4,000 per turn |
| Average context size by turn | turn 1: 2,528, turn 2: 2,840, turn 3: 2,641 |

About 1,650 tokens of every model call are the system prompt and tool definitions, so in short conversations that fixed cost is larger than the retrieval results.

The answer judge checks every claim against the text sources the agent can search: the Markdown documents, vector collections, keyword indexes, and wiki. An earlier version saw only the reference answer, so it could not tell whether an extra detail was true: it failed correct answers that added real rules (the client entertainment approval rule, the 90-day home office receipt deadline, and the two-day rule in the keyword policy index), and one unchanged answer passed in one grading run and failed in the next. With the source text, those answers pass, and invented extra details and misstated related facts fail.

## Example Questions

- What is the May cloud service expense trend?
- Which engineering projects are currently in progress?
- Show employees related to Engineering.
- What does the code repository contain about validation?
- How much can I spend on meals per day when travelling?
- What is our remote work policy?
- How many approvals does a pull request need?

## Notes

- This is a personal practice project exploring agentic search, tool calling, and multi-source retrieval patterns.
- All committed data is synthetic, including the documents under `data/docs/`.
- Placeholder values such as `your-secret-key` and `root:password` appear only inside generated sample code and are not real credentials.
- Do not commit a real `.env` file or generated local databases.
