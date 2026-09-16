from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from config import CODE_REPO_DIR, DB_PATH, DATA_DIR
from engines.database import DatabaseEngine
from engines.keyword_search import KeywordSearchEngine
from engines.vector_db import VectorDBEngine
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMPLOYEES = [
    {
        "name": "Alex Chen",
        "department": "Engineering",
        "position": "Senior Backend Engineer",
        "email": "alex.chen@example.com",
        "phone": "416-555-0101",
        "hire_date": "2019-03-15",
        "salary": 135000,
        "status": "active",
    },
    {
        "name": "Nina Li",
        "department": "Product",
        "position": "Product Manager",
        "email": "nina.li@example.com",
        "phone": "416-555-0102",
        "hire_date": "2020-06-01",
        "salary": 128000,
        "status": "active",
    },
    {
        "name": "Maya Patel",
        "department": "Engineering",
        "position": "Engineering Manager",
        "email": "maya.patel@example.com",
        "phone": "416-555-0103",
        "hire_date": "2018-01-10",
        "salary": 158000,
        "status": "active",
    },
    {
        "name": "Jordan Smith",
        "department": "Marketing",
        "position": "Marketing Lead",
        "email": "jordan.smith@example.com",
        "phone": "416-555-0104",
        "hire_date": "2017-09-20",
        "salary": 120000,
        "status": "active",
    },
]

DEPARTMENTS = [
    {"name": "Engineering", "manager": "Maya Patel", "budget": 5000000, "headcount": 25, "location": "Toronto HQ 3F"},
    {"name": "Product", "manager": "Nina Li", "budget": 2000000, "headcount": 10, "location": "Toronto HQ 4F"},
    {"name": "Marketing", "manager": "Jordan Smith", "budget": 3000000, "headcount": 15, "location": "Toronto HQ 2F"},
    {"name": "Finance", "manager": "Priya Shah", "budget": 800000, "headcount": 8, "location": "Toronto HQ 2F"},
]

PROJECTS = [
    {
        "name": "Smart Support V2",
        "department": "Engineering",
        "lead": "Maya Patel",
        "status": "in progress",
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
        "budget": 800000,
        "description": "Upgrade the AI support assistant with multi-turn conversation and internal knowledge retrieval.",
    },
    {
        "name": "Data Platform Refresh",
        "department": "Engineering",
        "lead": "Alex Chen",
        "status": "in progress",
        "start_date": "2024-02-15",
        "end_date": "2024-09-30",
        "budget": 1200000,
        "description": "Unify ingestion, cleaning, storage, and analytics workflows on a shared data platform.",
    },
    {
        "name": "Mobile App Redesign",
        "department": "Product",
        "lead": "Nina Li",
        "status": "planned",
        "start_date": "2024-04-01",
        "end_date": "2024-08-31",
        "budget": 500000,
        "description": "Refresh the mobile product experience and improve performance.",
    },
]

CONTRACTS = [
    {
        "client_name": "Northstar Analytics",
        "type": "service contract",
        "amount": 580000,
        "status": "active",
        "sign_date": "2024-01-15",
        "expire_date": "2025-01-14",
        "responsible_person": "Jordan Smith",
        "description": "Platform support and analytics consulting.",
    },
    {
        "client_name": "BlueBridge Networks",
        "type": "implementation contract",
        "amount": 890000,
        "status": "active",
        "sign_date": "2024-03-01",
        "expire_date": "2025-02-28",
        "responsible_person": "Nina Li",
        "description": "Implementation of workflow automation and reporting.",
    },
]

PRODUCTS = [
    {
        "name": "CloudEngine Pro",
        "category": "cloud management",
        "price": 9999,
        "stock": 999,
        "description": "Enterprise cloud management platform with multi-cloud automation.",
        "launch_date": "2023-06-01",
    },
    {
        "name": "DataInsight",
        "category": "analytics",
        "price": 59999,
        "stock": 999,
        "description": "Real-time data analytics platform with dashboards and alerting.",
        "launch_date": "2023-09-15",
    },
    {
        "name": "SmartChat",
        "category": "AI",
        "price": 19999,
        "stock": 999,
        "description": "LLM-based support assistant with retrieval over internal knowledge.",
        "launch_date": "2024-03-01",
    },
]

COMPANY_DOCS = [
    "The company was founded in 2015 and operates from Toronto with a remote-friendly engineering culture.",
    "The core technology stack includes microservices, Kubernetes, streaming data pipelines, and LLM-enabled support workflows.",
    "The current strategy focuses on AI product expansion, data security, and operational automation.",
    "Engineering is organized into frontend, backend, data, AI, and DevOps groups.",
]

TECH_DOCS = [
    "Microservice standard: services expose health checks, structured logs, and graceful shutdown hooks.",
    "Database standard: production services use reviewed SQL, migrations, backups, and read/write separation where required.",
    "CI/CD standard: pull requests run unit tests, lint checks, and quality gates before staging deployment.",
    "Security standard: APIs require authentication, sensitive data is encrypted, and dependencies are scanned regularly.",
]

MEETING_NOTES = [
    "Q1 engineering review: Smart Support V2 is on track, the data platform architecture passed review, and security audit work continues.",
    "Product planning: Mobile App Redesign starts in April, SmartChat 2.0 is planned for Q3, and DataInsight will add streaming analytics.",
    "Finance review: engineering budget prioritizes AI and data platform work, while marketing budget focuses on product launch campaigns.",
]

POLICIES = [
    {
        "id": "policy_001",
        "title": "Remote Work Policy",
        "content": "Employees may work remotely up to two days per week with manager approval.",
        "category": "HR policy",
        "source": "HR",
    },
    {
        "id": "policy_002",
        "title": "Expense Reimbursement Policy",
        "content": "Expenses must be submitted within 30 days. Items above 5000 require manager approval.",
        "category": "Finance policy",
        "source": "Finance",
    },
    {
        "id": "policy_003",
        "title": "Code Review Standard",
        "content": "Every pull request requires two reviewers. Critical modules require architecture review.",
        "category": "Engineering policy",
        "source": "Engineering",
    },
]

TECH_ARTICLES = [
    {
        "id": "tech_001",
        "title": "Microservice Architecture Guide",
        "content": "Services communicate over gRPC or REST, expose health endpoints, and publish metrics.",
        "category": "Architecture",
        "source": "Engineering",
    },
    {
        "id": "tech_002",
        "title": "CI/CD Pipeline Setup",
        "content": "Commits trigger tests, static analysis, image builds, and staging deployment.",
        "category": "DevOps",
        "source": "Platform",
    },
    {
        "id": "tech_003",
        "title": "API Design Standard",
        "content": "APIs use resource-oriented URLs, standard HTTP status codes, and cursor pagination.",
        "category": "Backend",
        "source": "Architecture",
    },
]


SAMPLE_CODE_FILES = {
    ".env.example": "DATABASE_URL=mysql://root:password@localhost:3306/cloudengine\nREDIS_URL=redis://localhost:6379\nSECRET_KEY=your-secret-key-here\n",
    "requirements.txt": "fastapi>=0.110\nuvicorn>=0.27\nsqlalchemy>=2.0\npydantic>=2.0\nopenai>=1.0\n",
    "Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY src ./src\nCMD [\"uvicorn\", \"src.main:app\", \"--host\", \"0.0.0.0\"]\n",
    "docker-compose.yml": "services:\n  api:\n    build: .\n    ports:\n      - \"8000:8000\"\n    environment:\n      - DATABASE_URL=mysql://root:password@db:3306/cloudengine\n",
    "src/main.py": '"""Application entry point."""\nfrom fastapi import FastAPI\n\napp = FastAPI(title="CloudEngine API")\n\n@app.get("/health")\ndef health_check():\n    return {"status": "ok"}\n',
    "src/config.py": '"""Application configuration."""\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    database_url: str = "mysql://root:password@localhost:3306/cloudengine"\n    secret_key: str = "your-secret-key"\n\nsettings = Settings()\n',
    "src/services/user_service.py": '"""User service."""\n\nclass UserService:\n    def __init__(self, db):\n        self.db = db\n\n    def create_user(self, user):\n        self.db.add(user)\n        self.db.commit()\n        return user\n\n    def get_user(self, user_id):\n        return self.db.query(\"User\").filter(id=user_id).first()\n',
    "src/services/chat_service.py": '"""Support chat service."""\nfrom openai import OpenAI\nfrom src.config import settings\n\nclass ChatService:\n    def __init__(self):\n        self.client = OpenAI(api_key=settings.secret_key)\n\n    def chat(self, message: str) -> str:\n        return f"Demo response for: {message}"\n',
    "src/utils/validators.py": '"""Validation helpers."""\nimport re\nfrom typing import Optional\n\ndef validate_email(email: str) -> bool:\n    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"\n    return bool(re.match(pattern, email))\n\ndef validate_password(password: str) -> Optional[str]:\n    if len(password) < 8:\n        return "Password must be at least 8 characters long."\n    if not re.search(r"[A-Z]", password):\n        return "Password must include an uppercase letter."\n    if not re.search(r"[a-z]", password):\n        return "Password must include a lowercase letter."\n    if not re.search(r"\\d", password):\n        return "Password must include a digit."\n    return None\n\ndef sanitize_input(text: str) -> str:\n    text = re.sub(r"<[^>]*>", "", text)\n    text = re.sub(r"[\\\"\\\';]", "", text)\n    return text.strip()\n',
    "tests/test_chat_service.py": '"""Chat service tests."""\nfrom src.services.chat_service import ChatService\n\ndef test_chat_returns_demo_response():\n    service = ChatService()\n    assert "Demo response" in service.chat("hello")\n',
}

def seed_documents(vector_db: VectorDBEngine) -> int:
    """Load markdown files from data/docs/, split them, and store them."""
    docs_dir = Path(DATA_DIR) / "docs"
    if not docs_dir.exists():
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    total = 0

    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)

        ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": path.name,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        vector_db.add_documents(
            "handbook",
            chunks,
            metadatas=metadatas,
            ids=ids,
        )

        total += len(chunks)

    return total

def is_seeded() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees")
        employee_count = cursor.fetchone()[0]
        conn.close()
    except sqlite3.Error:
        return False
    return employee_count > 0 and os.path.exists(os.path.join(CODE_REPO_DIR, "src", "main.py"))


def reset_sqlite_tables(db: DatabaseEngine) -> None:
    db.init_tables()
    db.connect()
    try:
        cursor = db.conn.cursor()
        for table in ["employees", "departments", "projects", "contracts", "products"]:
            cursor.execute(f"DELETE FROM {table}")
        db.conn.commit()
    finally:
        db.close()


def generate_code_repo() -> int:
    for relative_path, content in SAMPLE_CODE_FILES.items():
        full_path = Path(CODE_REPO_DIR) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    return len(SAMPLE_CODE_FILES)


def seed_all_large(force: bool = False) -> None:
    if is_seeded() and not force:
        print("Demo data already exists. Skipping seed step.")
        return

    print("Preparing English demo data...")

    db = DatabaseEngine()
    reset_sqlite_tables(db)
    db.insert_data("employees", EMPLOYEES)
    db.insert_data("departments", DEPARTMENTS)
    db.insert_data("projects", PROJECTS)
    db.insert_data("contracts", CONTRACTS)
    db.insert_data("products", PRODUCTS)

    vector_db = VectorDBEngine()
    vector_db.add_documents("company_info", COMPANY_DOCS, metadatas=[{"category": "company", "source": "profile"} for _ in COMPANY_DOCS])
    vector_db.add_documents("tech_docs", TECH_DOCS, metadatas=[{"category": "engineering", "source": "technical docs"} for _ in TECH_DOCS])
    vector_db.add_documents("meeting_notes", MEETING_NOTES, metadatas=[{"category": "meeting", "source": "meeting notes"} for _ in MEETING_NOTES])
    chunk_count = seed_documents(vector_db) 
    keyword_engine = KeywordSearchEngine()
    keyword_engine.add_documents("policies", POLICIES)
    keyword_engine.add_documents("tech_articles", TECH_ARTICLES)

    code_count = generate_code_repo()
    print(f"Seed complete: {len(EMPLOYEES)} employees, {len(PROJECTS)} projects, {code_count} code files, {chunk_count} doc chunks.")


def ensure_seed_data() -> None:
    seed_all_large(force=False)


if __name__ == "__main__":
    seed_all_large(force=True)
