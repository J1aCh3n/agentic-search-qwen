from __future__ import annotations
import logging
import json
import os
from typing import Optional
from datetime import datetime
from config import LOGS_DIR


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, name: str = "agentic_search"):
        if self._initialized:
            return
        self._initialized = True
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_format)

            log_file = os.path.join(LOGS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)

            json_log_file = os.path.join(LOGS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.json")
            self.json_handler = JsonLogHandler(json_log_file)
            self.json_handler.setLevel(logging.DEBUG)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(self.json_handler)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

    def log_search(self, engine: str, query: str, results_summary: str, round_num: int):
        self.info(
            f"[Round {round_num}] Search: engine={engine}, query={query}",
            extra={
                "search_engine": engine,
                "query": query,
                "results_summary": results_summary,
                "round": round_num
            }
        )

    def log_decision(self, round_num: int, decision: str, reasoning: str):
        self.info(
            f"[Round {round_num}] Decision: {decision}",
            extra={
                "round": round_num,
                "decision": decision,
                "reasoning": reasoning
            }
        )

    def log_final_answer(self, question: str, answer: str, total_rounds: int):
        self.info(
            f"[Final] Answer after {total_rounds} rounds",
            extra={
                "question": question,
                "answer": answer[:200],
                "total_rounds": total_rounds
            }
        )

    def search_logs(self, keyword: str = None, level: str = None,
                    engine: str = None, limit: int = 20) -> str:
        log_file = None
        for f in sorted(os.listdir(LOGS_DIR), reverse=True):
            if f.endswith(".json") and f.startswith(self.name):
                log_file = os.path.join(LOGS_DIR, f)
                break

        if not log_file or not os.path.exists(log_file):
            return json.dumps({"error": "No log file found"}, ensure_ascii=False)

        results = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if keyword and keyword.lower() not in json.dumps(entry, ensure_ascii=False).lower():
                        continue
                    if level and entry.get("level", "").upper() != level.upper():
                        continue
                    if engine and entry.get("search_engine") != engine:
                        continue
                    results.append(entry)
                except (json.JSONDecodeError, AttributeError):
                    continue

        return json.dumps(results[-limit:], ensure_ascii=False, indent=2, default=str)


class JsonLogHandler(logging.Handler):
    def __init__(self, log_file: str):
        super().__init__()
        self.log_file = log_file

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            for key in ["search_engine", "query", "results_summary", "round",
                        "decision", "reasoning", "question", "answer", "total_rounds"]:
                if hasattr(record, key):
                    log_entry[key] = getattr(record, key)

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
