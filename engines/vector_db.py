from __future__ import annotations

import json
from typing import Any

import chromadb

from config import (
    VECTOR_DB_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    EMBEDDING_MODEL,
    VECTOR_DISTANCE_THRESHOLD,
)
from langchain_openai import OpenAIEmbeddings


class DashScopeEmbeddingFunction:
    """Real embedding model from DashScope (text-embedding-v4)."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            check_embedding_ctx_length=False,
            chunk_size=10
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(input)


class VectorDBEngine:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.embedding_function = DashScopeEmbeddingFunction()
        self.collections: dict[str, Any] = {}

    def _get_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.embedding_function,
            )
        return self.collections[name]

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        collection = self._get_collection(collection_name)
        if ids is None:
            ids = [f"{collection_name}_{index}" for index in range(len(documents))]
        if metadatas is None:
            metadatas = [{"source": collection_name} for _ in documents]

        existing = set(collection.get(include=[])["ids"])
        filtered_docs = []
        filtered_ids = []
        filtered_meta = []
        for doc, doc_id, meta in zip(documents, ids, metadatas):
            if doc_id in existing:
                continue
            filtered_docs.append(doc)
            filtered_ids.append(doc_id)
            filtered_meta.append(meta)

        if filtered_docs:
            collection.add(documents=filtered_docs, ids=filtered_ids, metadatas=filtered_meta)
        return len(filtered_docs)

    def search(
        self,
        query: str,
        collection_name: str | None = None,
        n_results: int = 5,
        where: dict | None = None,
    ) -> str:
        if collection_name:
            collection = self._get_collection(collection_name)
            query_params: dict[str, Any] = {"query_texts": [query], "n_results": n_results}
            if where:
                query_params["where"] = where
            results = collection.query(**query_params)
            return self._format_results(results, collection_name)

        all_results = {}
        for name in self.list_collections():
            collection = self._get_collection(name)
            results = collection.query(query_texts=[query], n_results=n_results)
            parsed = json.loads(self._format_results(results, name))
            if parsed.get("results"):
                all_results[name] = parsed["results"]
        if not all_results:
            return json.dumps({
                "results": {},
                "message": f"No document was within the distance threshold {VECTOR_DISTANCE_THRESHOLD}. The demo dataset does not contain relevant content for this query."
            }, ensure_ascii=False, indent=2)

        return json.dumps({"results": all_results}, ensure_ascii=False, indent=2)

    def _format_results(self, results: dict, collection_name: str) -> str:
        formatted = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            distances = results.get("distances", [[]])[0] if results.get("distances") else [0] * len(docs)
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else [{}] * len(docs)
            ids = results.get("ids", [[]])[0] if results.get("ids") else [""] * len(docs)
            for doc, distance, metadata, doc_id in zip(docs, distances, metadatas, ids):
                if distance > VECTOR_DISTANCE_THRESHOLD:
                    continue
                formatted.append(
                    {
                        "id": doc_id,
                        "content": doc,
                        "distance": round(distance, 4),
                        "metadata": metadata,
                    }
                )
        payload = {"collection": collection_name, "results": formatted}
        if not formatted:
            payload["message"] = (
                f"No document was within the distance threshold {VECTOR_DISTANCE_THRESHOLD}. "
                "The demo dataset does not contain relevant content for this query."
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)


    def list_collections(self) -> list[str]:
        return [collection.name for collection in self.client.list_collections()]

    def get_collection_info(self, collection_name: str) -> str:
        try:
            collection = self._get_collection(collection_name)
            peek = collection.peek(limit=3)
            return json.dumps(
                {
                    "name": collection_name,
                    "document_count": collection.count(),
                    "sample_documents": peek.get("documents", [])[:3],
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def get_all_collections_info(self) -> str:
        infos = {}
        for name in self.list_collections():
            infos[name] = json.loads(self.get_collection_info(name))
        return json.dumps(infos, ensure_ascii=False, indent=2)
