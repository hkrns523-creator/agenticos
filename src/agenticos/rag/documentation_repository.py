"""Retrieves standard operating procedures (SOPs) via similarity/MMR search
over the manual/SOP PDFs. Note this is a *retrieval* step, not a generation
step — no LLM call is involved, so it costs milliseconds, not seconds.

Imports of the RAG stack (langchain-community loaders, HF embeddings,
Chroma) are deferred to `__init__` rather than module level: they're heavy
(sentence-transformers pulls in torch) and unrelated code paths — including
most unit tests, which inject a fake documentation tool instead of a real
one — shouldn't have to pay for loading them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticos.exceptions import ToolExecutionError
from agenticos.logging_config import get_logger

logger = get_logger("rag.documentation_repository")


class DocumentationRepository:
    def __init__(
        self,
        docs_dir: str | Path,
        vector_db_dir: str | Path,
        embedding_model: str,
        top_k: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        search_type: str = "mmr",
        fetch_k: int = 10,
        score_threshold: float | None = None,
    ):
        from agenticos.rag.loader import load_and_chunk_documents
        from agenticos.rag.vector_store import create_vector_store, load_vector_store

        self._top_k = top_k
        self._search_type = search_type
        self._fetch_k = fetch_k
        self._score_threshold = score_threshold
        vector_db_dir = Path(vector_db_dir)

        if not vector_db_dir.exists():
            logger.info("building vector store", extra={"extra_fields": {"docs_dir": str(docs_dir)}})
            chunks = load_and_chunk_documents(docs_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self._db = create_vector_store(chunks, vector_db_dir, embedding_model)
        else:
            logger.info("loading existing vector store", extra={"extra_fields": {"vector_db_dir": str(vector_db_dir)}})
            self._db = load_vector_store(vector_db_dir, embedding_model)

    def get_sop(self, topic: str) -> dict[str, Any]:
        from agenticos.rag.vector_store import search

        try:
            results = search(
                self._db,
                topic,
                top_k=self._top_k,
                search_type=self._search_type,
                fetch_k=self._fetch_k,
                score_threshold=self._score_threshold,
            )
        except Exception as exc:  # Chroma/HF errors aren't a single stable type
            logger.warning("SOP lookup failed", extra={"extra_fields": {"topic": topic, "error": str(exc)}})
            raise ToolExecutionError(f"Documentation lookup failed for topic '{topic}': {exc}") from exc

        matches = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_name", "unknown"),
                "page": doc.metadata.get("page"),
            }
            for doc in results
        ]

        return {
            "topic": topic,
            "documents": [m["content"] for m in matches],  # kept for backward compatibility
            "matches": matches,
            "found": bool(matches),
        }
