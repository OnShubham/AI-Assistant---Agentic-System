"""
app/rag/__init__.py — Public API for the RAG pipeline.

Quick start:
    from app.rag import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.ingest("path/to/document.pdf")
    result = pipeline.query("What is retrieval-augmented generation?")
    print(result["answer"])
"""

from app.rag.pipeline import RAGPipeline
from app.rag.retrival import retrieve_and_answer
from app.rag.vector_store import FAISSVectorStore
from app.rag.embunding import embed_query, embed_documents, get_embedding_model
from app.rag.chunking import split_documents

__all__ = [
    "RAGPipeline",
    "retrieve_and_answer",
    "FAISSVectorStore",
    "embed_query",
    "embed_documents",
    "get_embedding_model",
    "split_documents",
]
