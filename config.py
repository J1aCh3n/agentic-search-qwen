import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "db")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
CODE_REPO_DIR = os.path.join(DATA_DIR, "code_repo")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
KEYWORD_INDEX_DIR = os.path.join(DATA_DIR, "keyword_index")

DB_PATH = os.path.join(DB_DIR, "enterprise.db")

LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.6-plus")
EMBEDDING_MODEL = os.environ.get("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
VECTOR_DISTANCE_THRESHOLD = float(os.environ.get("VECTOR_DISTANCE_THRESHOLD", "0.70"))

MAX_SEARCH_ROUNDS = 20

for d in [DATA_DIR, DB_DIR, VECTOR_DB_DIR, CODE_REPO_DIR, LOGS_DIR, KEYWORD_INDEX_DIR]:
    os.makedirs(d, exist_ok=True)
