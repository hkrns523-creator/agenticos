"""Chroma vector store construction/loading and retrieval, parameterized by
settings instead of hardcoded paths/model names/search strategy."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache
def _embedding(model_name: str) -> HuggingFaceEmbeddings:
    # Cached: loading a sentence-transformers model is expensive (seconds),
    # and every DocumentationRepository instance would otherwise reload it.
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(documents, persist_directory: str | Path, embedding_model: str) -> Chroma:
    return Chroma.from_documents(
        documents=documents,
        embedding=_embedding(embedding_model),
        persist_directory=str(persist_directory),
    )


def load_vector_store(persist_directory: str | Path, embedding_model: str) -> Chroma:
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=_embedding(embedding_model),
    )


def search(
    db: Chroma,
    query: str,
    top_k: int,
    search_type: str = "mmr",
    fetch_k: int = 10,
    score_threshold: float | None = None,
) -> list[Any]:
    """Runs the configured retrieval strategy over the vector store.

    - "similarity": plain nearest-neighbor by cosine distance. Fast, but a
      manual with repetitive phrasing can return several near-duplicate
      chunks from the same page, crowding out other relevant content.
    - "mmr" (Maximal Marginal Relevance, the default): pulls a larger
      `fetch_k` candidate pool, then greedily selects `top_k` results that
      balance relevance against diversity from what's already selected.
      Costs a bit more compute (still milliseconds — no LLM involved) for
      meaningfully less redundant context handed to the Supervisor.

    When `score_threshold` is set, low-confidence similarity matches are
    dropped instead of always returning exactly `top_k` results — better to
    tell the user nothing relevant was found than to hand the Supervisor a
    barely-related chunk it will present as if it answers the question.
    """
    if score_threshold is not None:
        scored = db.similarity_search_with_relevance_scores(query, k=top_k)
        return [doc for doc, score in scored if score >= score_threshold]

    if search_type == "mmr":
        return db.max_marginal_relevance_search(query, k=top_k, fetch_k=max(fetch_k, top_k))

    return db.similarity_search(query, k=top_k)
