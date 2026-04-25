"""
file_tool.py — Document loading utilities.

Supports PDF files via PyPDFLoader.
Returns a list of LangChain Document objects (with page_content + metadata).
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def pdf_parse(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects,
    one per page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects with page_content and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info(f"Loading PDF: {path.name}")
    loader = PyPDFLoader(str(path))
    documents: List[Document] = loader.load()
    logger.info(f"Loaded {len(documents)} page(s) from '{path.name}'")
    return documents
