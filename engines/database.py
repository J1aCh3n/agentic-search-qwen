from __future__ import annotations

import json
import re
import sqlite3

from config import DB_PATH


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "what",
    "which",
    "who",
    "how",
    "is",
    "are",
    "show",
    "find",
    "tell",
    "me",
    "about",
    "current",
    "latest",
    "recent",
    "data",
    "information",
}


def _extract_keywords(query: str) -> list[str]:
    keywords = re.split(r"[\s,;?!,.]+", query.lower())
    keywords = [item.strip() for item in keywords if item.strip() and item.strip() not in STOP_WORDS]
    return keywords or [query]


class DatabaseEngine:
    def __init__(self) -> None:
        self.db_path = DB_PATH
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def init_tables(self) -> None:
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT,
                position TEXT,
                email TEXT,
                phone TEXT,
                hire_date TEXT,
                salary REAL,
                status TEXT DEFAULT 'active'
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                manager TEXT,
                budget REAL,
                headcount INTEGER,
                location TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT,
                lead TEXT,
                status TEXT,
                start_date TEXT,
                end_date TEXT,
                budget REAL,
                description TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_name TEXT NOT NULL,
                type TEXT,
                amount REAL,
                status TEXT,
                sign_date TEXT,
                expire_date TEXT,
                responsible_person TEXT,
                description TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                price REAL,
                stock INTEGER,
                description TEXT,
                launch_date TEXT
            )
            """
        )
        self.conn.commit()
        self.close()

    def search(self, query: str, table: str | None = None, limit: int = 10) -> str:
        self.connect()
        try:
            cursor = self.conn.cursor()
            keywords = _extract_keywords(query)
            tables = [table] if table in {"employees", "departments", "projects", "contracts", "products"} else [
                "employees",
                "departments",
                "projects",
                "contracts",
                "products",
            ]

            all_results: dict[str, list[dict]] = {}
            for table_name in tables:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
                columns = [desc[0] for desc in cursor.description]
                conditions = []
                params: list[object] = []
                for column in columns:
                    for keyword in keywords:
                        conditions.append(f"{column} LIKE ?")
                        params.append(f"%{keyword}%")

                where_clause = " OR ".join(conditions)
                cursor.execute(f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT ?", [*params, limit])
                rows = [dict(row) for row in cursor.fetchall()]
                if rows:
                    all_results[table_name] = rows

            result: object = all_results.get(table, []) if table else all_results
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
        finally:
            self.close()

    def execute_sql(self, sql: str) -> str:
        self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = [dict(zip(columns, row)) for row in rows]
            return json.dumps(results, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        finally:
            self.close()

    def get_schema(self) -> str:
        self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            schema_info = {}
            for table_name in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                schema_info[table_name] = {"columns": columns, "row_count": count}
            return json.dumps(schema_info, ensure_ascii=False, indent=2)
        finally:
            self.close()

    def insert_data(self, table: str, data: list[dict]) -> int:
        if not data:
            return 0
        self.connect()
        try:
            cursor = self.conn.cursor()
            columns = list(data[0].keys())
            placeholders = ", ".join(["?"] * len(columns))
            col_str = ", ".join(columns)
            count = 0
            for row in data:
                values = [row.get(column) for column in columns]
                cursor.execute(f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})", values)
                count += 1
            self.conn.commit()
            return count
        finally:
            self.close()
