"""
chunking.py — Text chunking utilities.

Splits LangChain Document objects into smaller, overlapping chunks using
RecursiveCharacterTextSplitter. Preserves source metadata for traceability.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.constant.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller, overlapping chunks.

    Uses a hierarchical separator strategy:
      paragraph → newline → sentence → word

    Args:
        documents: Raw documents (typically from file_tool.pdf_parse).

    Returns:
        List of chunked Document objects, each containing a slice of text
        and inherited metadata (source, page number, etc.).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,   # adds 'start_index' to metadata for tracing
    )

    chunks: List[Document] = text_splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} document(s) into {len(chunks)} chunk(s) "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks
