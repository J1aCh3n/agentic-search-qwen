# Agentic Search Qwen

Agentic Search Qwen is a Streamlit demo of an enterprise search assistant that uses a Qwen-compatible chat model to choose tools, retrieve evidence, and answer questions over multiple synthetic data sources.

The repository has been converted to an English-only public demo. The data is synthetic and is intended for portfolio review, not production use.

## Features

- Streamlit interface for interactive search and trace inspection
- CLI entry point for terminal-based testing
- Tool-calling agent with a Qwen-compatible OpenAI API client
- SQLite search over structured enterprise records
- Chroma vector search over synthetic company and engineering documents using a local deterministic embedding function
- Whoosh keyword search over policy and technical article indexes
- Sample code repository search
- Simulated enterprise systems for HR, finance, project management, and wiki queries
- Explicit API error reporting when model credentials, region, or entitlement are wrong
- Local fallback search only when no model API key is configured

## Project Structure

```text
.
+-- app.py                  # Streamlit UI
+-- main.py                 # CLI entry point
+-- config.py               # Runtime paths and model configuration
+-- seed_data_large.py      # English synthetic data generator
+-- core/
|   +-- agent.py            # Tool-calling search agent
|   +-- llm.py              # Qwen-compatible OpenAI client wrapper
|   +-- logger.py           # Search trace logging
+-- engines/
|   +-- database.py         # SQLite search
|   +-- vector_db.py        # Chroma vector search
|   +-- keyword_search.py   # Whoosh keyword search
|   +-- code_search.py      # Sample repository search
|   +-- enterprise_sdk.py   # Simulated enterprise systems
+-- data/
    +-- code_repo/          # Generated English sample repository
```

Generated runtime stores under `data/db/`, `data/vector_db/`, `data/keyword_index/`, and `data/logs/` are ignored by Git. They are rebuilt by the seed script.

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
DASHSCOPE_MODEL=qwen3.6-plus
```

Use the DashScope endpoint that matches the region where the API key was created. This project defaults to the international endpoint because a China-region endpoint can reject an international key with authentication or entitlement errors.

The app can still demonstrate local fallback behavior without an API key. If an API key is present, model/API errors are shown directly instead of being hidden by fallback search.

## Run

Streamlit:

```powershell
streamlit run app.py
```

CLI:

```powershell
python main.py
```

Force-regenerate demo data:

```powershell
python seed_data_large.py
```

## Example Questions

- What is the May cloud service expense trend?
- Which engineering projects are currently in progress?
- Show employees related to Engineering.
- What does the code repository contain about validation?
- Which policy describes remote work?

## Notes

- This is a personal practice project exploring agentic search, tool calling, and multi-source retrieval patterns.
- All committed data is synthetic.
- Placeholder values such as `your-secret-key` and `root:password` appear only inside generated sample code and are not real credentials.
- Do not commit a real `.env` file or generated local databases.
