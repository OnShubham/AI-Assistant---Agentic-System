"""
embunding.py — Embedding generation with SentenceTransformers.

Provides a singleton model loader and a function to convert text chunks
into dense vector embeddings for FAISS indexing.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from app.constant.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Singleton model (loaded once, reused across calls) ────────────────────────
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the SentenceTransformer model (cached after first call)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of raw strings into embedding vectors.

    Args:
        texts: Plain text strings to embed.

    Returns:
        numpy ndarray of shape (len(texts), embedding_dim).
    """
    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} text(s)...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    logger.info(f"Embedding complete — shape: {embeddings.shape}")
    return embeddings


def embed_documents(chunks: List[Document]) -> tuple[List[Document], np.ndarray]:
    """
    Embed a list of LangChain Document chunks.

    Args:
        chunks: Document chunks from chunking.split_documents().

    Returns:
        Tuple of (chunks, embeddings_array) where embeddings_array[i]
        corresponds to chunks[i].
    """
    texts = [doc.page_content for doc in chunks]
    embeddings = embed_texts(texts)
    return chunks, embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for similarity search.

    Args:
        query: The user's question or search string.

    Returns:
        1-D numpy array (embedding_dim,) — L2-normalised.
    """
    return embed_texts([query])[0]