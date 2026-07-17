from __future__ import annotations

import json
import os

from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, KEYWORD, STORED, TEXT, Schema
from whoosh.index import create_in, exists_in, open_dir
from whoosh.qparser import MultifieldParser, OrGroup

from config import KEYWORD_INDEX_DIR


class KeywordSearchEngine:
    def __init__(self) -> None:
        self.index_dir = KEYWORD_INDEX_DIR
        self.indexes = {}
        self._analyzer = StandardAnalyzer()

    def _get_schema(self) -> Schema:
        return Schema(
            doc_id=ID(stored=True, unique=True),
            title=TEXT(analyzer=self._analyzer, stored=True),
            content=TEXT(analyzer=self._analyzer, stored=True),
            category=KEYWORD(stored=True),
            source=STORED,
        )

    def _get_index(self, index_name: str):
        if index_name in self.indexes:
            return self.indexes[index_name]

        index_path = os.path.join(self.index_dir, index_name)
        os.makedirs(index_path, exist_ok=True)
        ix = open_dir(index_path) if exists_in(index_path) else create_in(index_path, self._get_schema())
        self.indexes[index_name] = ix
        return ix

    def add_documents(self, index_name: str, documents: list[dict]) -> int:
        ix = self._get_index(index_name)
        writer = ix.writer()
        for index, doc in enumerate(documents):
            writer.update_document(
                doc_id=doc.get("id", f"{index_name}_{index}"),
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                category=doc.get("category", ""),
                source=doc.get("source", ""),
            )
        writer.commit()
        return len(documents)

    def search(self, query: str, index_name: str | None = None, fields: list[str] | None = None, limit: int = 10) -> str:
        fields = fields or ["title", "content"]
        if index_name:
            return self._search_single(index_name, query, fields, limit)

        all_results = {}
        for name in self.list_indexes():
            result = json.loads(self._search_single(name, query, fields, limit))
            if result.get("results"):
                all_results[name] = result["results"]
        return json.dumps({"results": all_results}, ensure_ascii=False, indent=2)

    def _search_single(self, index_name: str, query: str, fields: list[str], limit: int) -> str:
        try:
            ix = self._get_index(index_name)
        except Exception as exc:
            return json.dumps({"error": f"Index '{index_name}' not found: {exc}"}, ensure_ascii=False)

        try:
            with ix.searcher() as searcher:
                parser = MultifieldParser(fields, ix.schema, group=OrGroup)
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=limit)
                formatted = [
                    {
                        "doc_id": hit.get("doc_id", ""),
                        "title": hit.get("title", ""),
                        "content": hit.get("content", "")[:500],
                        "category": hit.get("category", ""),
                        "source": hit.get("source", ""),
                        "score": round(hit.score, 4),
                    }
                    for hit in results
                ]
            return json.dumps({"index": index_name, "query": query, "total_found": len(formatted), "results": formatted}, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def list_indexes(self) -> list[str]:
        indexes = []
        if os.path.exists(self.index_dir):
            for name in os.listdir(self.index_dir):
                index_path = os.path.join(self.index_dir, name)
                if os.path.isdir(index_path) and exists_in(index_path):
                    indexes.append(name)
        return indexes

    def get_index_info(self, index_name: str) -> str:
        try:
            ix = self._get_index(index_name)
            return json.dumps({"name": index_name, "document_count": ix.doc_count()}, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def get_all_indexes_info(self) -> str:
        infos = {}
        for name in self.list_indexes():
            infos[name] = json.loads(self.get_index_info(name))
        return json.dumps(infos, ensure_ascii=False, indent=2)
