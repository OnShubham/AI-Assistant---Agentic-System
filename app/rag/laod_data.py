"""
laod_data.py — Convenience wrapper kept for backward-compatibility.

Prefer using app.tools.file_tool.pdf_parse() + app.rag.chunking.split_documents()
directly, or the high-level RAGPipeline.ingest() method.
"""

from app.tools.file_tool import pdf_parse
from app.rag.chunking import split_documents


def load_and_split_documents(file_path: str):
    """
    Load a PDF and return chunked LangChain Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    raw_docs = pdf_parse(file_path)
    return split_documents(raw_docs)
