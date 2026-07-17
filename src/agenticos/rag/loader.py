"""Loads and chunks source documents (PDF manuals/SOPs) for ingestion into
the vector store.

The original loader indexed one vector per raw PDF *page*. Manual pages
routinely mix several unrelated procedures, so a page-sized chunk dilutes
the embedding and a similarity search over it returns noisy matches. This
splits each page into smaller, overlapping chunks (`RecursiveCharacterTextSplitter`)
before embedding, which is the single biggest lever for retrieval quality
here — it does not require touching the embedding model or the query path.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agenticos.logging_config import get_logger

logger = get_logger("rag.loader")


def load_documents(docs_dir: str | Path):
    """Loads every PDF under `docs_dir` as one Document per page."""
    loader = PyPDFDirectoryLoader(str(docs_dir))
    documents = loader.load()
    logger.info("loaded documents", extra={"extra_fields": {"pages": len(documents), "docs_dir": str(docs_dir)}})
    return documents


def load_and_chunk_documents(docs_dir: str | Path, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Loads every PDF under `docs_dir` and splits it into overlapping,
    citation-friendly chunks. Each chunk keeps its source page's metadata
    (`source`, `page`) plus a `source_name` for display, so retrieval results
    can be traced back to e.g. "Boiler_manual.pdf" instead of an opaque blob
    of text with no provenance.
    """
    documents = load_documents(docs_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for doc in chunks:
        source = doc.metadata.get("source", "")
        doc.metadata["source_name"] = Path(source).name if source else "unknown"

    logger.info(
        "chunked documents",
        extra={
            "extra_fields": {
                "pages": len(documents),
                "chunks": len(chunks),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        },
    )
    return chunks
