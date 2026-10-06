"""Offline tests for the tools that do not need a model: SQL access and list_sources.

    python -m unittest tests.test_tools
"""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.lc_tools import build_tools
from engines.database import DatabaseEngine


class FakeEngine:
    """Stands in for any engine whose output a test does not look at."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: "{}"


class FakeCodeEngine(FakeEngine):
    def list_files(self, directory="", pattern=None):
        files = ["README.md", "src\\app.py", "src/models/user.py", "tests/test_app.py"]
        return json.dumps({"directory": "/", "total_files": len(files), "files": files})


def tools_with(db=None, code_engine=None):
    tools = build_tools(db or FakeEngine(), FakeEngine(), FakeEngine(), code_engine or FakeEngine(), FakeEngine())
    return {tool.name: tool for tool in tools}


class ReadOnlySqlTests(unittest.TestCase):
    def setUp(self):
        # A path with a space, like the project folder, because the read-only URI must encode it.
        folder = Path(tempfile.mkdtemp()) / "with space"
        folder.mkdir()
        self.db_path = folder / "test.db"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE employees (name TEXT)")
            conn.execute("INSERT INTO employees VALUES ('Ada')")
        self.db = DatabaseEngine()
        self.db.db_path = str(self.db_path)

    def count(self):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT count(*) FROM employees").fetchone()[0]

    def test_the_sql_connection_refuses_writes_whatever_the_text_says(self):
        # Called on the engine directly, past the tool's keyword check, which is not the protection.
        result = json.loads(self.db.execute_sql("DELETE FROM employees"))
        self.assertIn("readonly", result["error"])
        self.assertEqual(self.count(), 1)

    def test_select_queries_still_work(self):
        rows = json.loads(tools_with(db=self.db)["database_sql"].invoke({"sql": "SELECT name FROM employees"}))
        self.assertEqual(rows, [{"name": "Ada"}])

    def test_a_missing_database_is_an_error_not_a_new_empty_file(self):
        self.db.db_path = str(self.db_path.with_name("missing.db"))
        result = json.loads(self.db.execute_sql("SELECT 1"))
        self.assertIn("error", result)
        self.assertFalse(self.db_path.with_name("missing.db").exists())


class ListSourcesTests(unittest.TestCase):
    def test_the_code_repository_is_listed_by_its_top_level_entries(self):
        sources = json.loads(tools_with(code_engine=FakeCodeEngine())["list_sources"].invoke({}))
        self.assertEqual(sources["code_repository"], {"total_files": 4, "top_level": ["README.md", "src", "tests"]})


if __name__ == "__main__":
    unittest.main()
