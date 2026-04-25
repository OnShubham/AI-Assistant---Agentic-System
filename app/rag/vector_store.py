"""
vector_store.py — FAISS-based vector store for document retrieval.

Handles:
  - Building a FAISS index from document embeddings
  - Persisting the index + metadata to disk
  - Loading an existing index from disk
  - Similarity search returning ranked Document objects
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document

from app.constant.config import VECTOR_STORE_DIR, TOP_K_RESULTS

logger = logging.getLogger(__name__)

# File names within VECTOR_STORE_DIR
_INDEX_FILE    = "index.faiss"
_METADATA_FILE = "metadata.pkl"


class FAISSVectorStore:
    """
    Thin wrapper around a flat FAISS index with document metadata storage.

    Using IndexFlatIP (inner-product) which equals cosine similarity when
    vectors are L2-normalised (as done in embunding.embed_texts).
    """

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.documents: List[Document] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Document], embeddings: np.ndarray) -> None:
        """
        Construct the FAISS index from pre-computed embeddings.

        Args:
            chunks:     Document chunks (parallel to embeddings).
            embeddings: float32 array of shape (N, dim).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP — dim={dim}, docs={len(chunks)}")

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.documents = chunks
        logger.info("FAISS index built successfully.")

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, store_dir: Path = VECTOR_STORE_DIR) -> None:
        """Persist the FAISS index and document metadata to disk."""
        if self.index is None:
            raise RuntimeError("Index has not been built yet. Call build() first.")

        store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(store_dir / _INDEX_FILE))

        with open(store_dir / _METADATA_FILE, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Vector store saved to '{store_dir}'")

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, store_dir: Path = VECTOR_STORE_DIR) -> None:
        """Load a previously persisted FAISS index from disk."""
        index_path = store_dir / _INDEX_FILE
        meta_path  = store_dir / _METADATA_FILE

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No saved vector store found at '{store_dir}'. "
                "Run the pipeline with ingest=True first."
            )

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            self.documents = pickle.load(f)

        logger.info(
            f"Loaded vector store — {self.index.ntotal} vectors, "
            f"{len(self.documents)} documents"
        )

    @property
    def is_ready(self) -> bool:
        """True if the index has been built or loaded."""
        return self.index is not None and self.index.ntotal > 0

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_RESULTS,
    ) -> List[Tuple[Document, float]]:
        """
        Find the top-k most similar documents to a query embedding.

        Args:
            query_embedding: 1-D float32 array (embedding_dim,).
            top_k:           Number of results to return.

        Returns:
            List of (Document, score) tuples, sorted by descending similarity.
        """
        if not self.is_ready:
            raise RuntimeError("Vector store is not ready. Build or load it first.")

        query_vec = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)

        results: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS padding for fewer results than top_k
            results.append((self.documents[idx], float(score)))

        logger.info(f"Retrieved {len(results)} result(s) for query.")
        return results
