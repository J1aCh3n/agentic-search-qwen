from __future__ import annotations
import json
import os
import re
from typing import List, Dict, Optional
from config import CODE_REPO_DIR


class CodeSearchEngine:
    def __init__(self):
        self.repo_dir = CODE_REPO_DIR

    def search(self, query: str, file_pattern: str = None,
               search_type: str = "content", limit: int = 10) -> str:
        if search_type == "filename":
            return self._search_by_filename(query, limit)
        elif search_type == "content":
            return self._search_by_content(query, file_pattern, limit)
        elif search_type == "symbol":
            return self._search_by_symbol(query, limit)
        else:
            return self._search_by_content(query, file_pattern, limit)

    def _search_by_filename(self, query: str, limit: int) -> str:
        results = []
        keywords = query.lower().split()

        for root, dirs, files in os.walk(self.repo_dir):
            for filename in files:
                if all(kw in filename.lower() for kw in keywords):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, self.repo_dir)
                    try:
                        size = os.path.getsize(filepath)
                        results.append({
                            "file_path": rel_path,
                            "file_name": filename,
                            "size": size,
                            "match_type": "filename"
                        })
                    except OSError:
                        continue

                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break

        return json.dumps({
            "query": query,
            "search_type": "filename",
            "total_found": len(results),
            "results": results
        }, ensure_ascii=False, indent=2)

    def _search_by_content(self, query: str, file_pattern: str, limit: int) -> str:
        results = []
        keywords = query.lower().split()
        code_extensions = {
            ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c",
            ".h", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
            ".sql", ".html", ".css", ".json", ".yaml", ".yml", ".xml",
            ".md", ".txt", ".toml", ".cfg", ".ini", ".env"
        }

        for root, dirs, files in os.walk(self.repo_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in code_extensions:
                    continue

                if file_pattern and file_pattern.lower() not in filename.lower():
                    continue

                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, self.repo_dir)

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    matches = []
                    for i, line in enumerate(lines, 1):
                        line_lower = line.lower()
                        if all(kw in line_lower for kw in keywords):
                            matches.append({
                                "line_number": i,
                                "content": line.rstrip()[:200]
                            })

                    if matches:
                        results.append({
                            "file_path": rel_path,
                            "file_name": filename,
                            "match_count": len(matches),
                            "matches": matches[:5],
                            "match_type": "content"
                        })

                    if len(results) >= limit:
                        break
                except (OSError, UnicodeDecodeError):
                    continue

            if len(results) >= limit:
                break

        return json.dumps({
            "query": query,
            "search_type": "content",
            "total_found": len(results),
            "results": results
        }, ensure_ascii=False, indent=2)

    def _search_by_symbol(self, query: str, limit: int) -> str:
        results = []
        patterns = [
            (r'def\s+' + re.escape(query), "function"),
            (r'class\s+' + re.escape(query), "class"),
            (r'function\s+' + re.escape(query), "function"),
            (r'const\s+' + re.escape(query), "constant"),
            (r'let\s+' + re.escape(query), "variable"),
            (r'var\s+' + re.escape(query), "variable"),
            (r'interface\s+' + re.escape(query), "interface"),
            (r'type\s+' + re.escape(query), "type"),
        ]

        for root, dirs, files in os.walk(self.repo_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in {".py", ".js", ".ts", ".java", ".go", ".rs"}:
                    continue

                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, self.repo_dir)

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    for pattern, symbol_type in patterns:
                        for match in re.finditer(pattern, content):
                            line_num = content[:match.start()].count("\n") + 1
                            line_content = content.split("\n")[line_num - 1].strip()[:200]
                            results.append({
                                "file_path": rel_path,
                                "file_name": filename,
                                "symbol_type": symbol_type,
                                "line_number": line_num,
                                "content": line_content,
                                "match_type": "symbol"
                            })

                    if len(results) >= limit:
                        break
                except (OSError, UnicodeDecodeError):
                    continue

            if len(results) >= limit:
                break

        return json.dumps({
            "query": query,
            "search_type": "symbol",
            "total_found": len(results),
            "results": results[:limit]
        }, ensure_ascii=False, indent=2)

    def read_file(self, file_path: str, start_line: int = 1,
                  end_line: int = None) -> str:
        full_path = os.path.join(self.repo_dir, file_path)
        if not os.path.exists(full_path):
            return json.dumps({"error": f"File not found: {file_path}"}, ensure_ascii=False)

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            start = max(1, start_line) - 1
            end = end_line if end_line else len(lines)

            selected_lines = lines[start:end]
            result_lines = []
            for i, line in enumerate(selected_lines, start=start + 1):
                result_lines.append({"line": i, "content": line.rstrip()})

            return json.dumps({
                "file_path": file_path,
                "total_lines": len(lines),
                "showing": f"{start + 1}-{min(end, len(lines))}",
                "content": result_lines
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def list_files(self, directory: str = "", pattern: str = None) -> str:
        target_dir = os.path.join(self.repo_dir, directory) if directory else self.repo_dir
        if not os.path.exists(target_dir):
            return json.dumps({"error": f"Directory not found: {directory}"}, ensure_ascii=False)

        files = []
        for root, dirs, filenames in os.walk(target_dir):
            for f in filenames:
                rel_path = os.path.relpath(os.path.join(root, f), self.repo_dir)
                if pattern and pattern.lower() not in f.lower():
                    continue
                files.append(rel_path)

        return json.dumps({
            "directory": directory or "/",
            "total_files": len(files),
            "files": files[:50]
        }, ensure_ascii=False, indent=2)
