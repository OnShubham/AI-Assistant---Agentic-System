"""
pipeline.py — End-to-end RAG pipeline orchestrator.

Usage:
    # Ingest once (builds & saves the FAISS index)
    from app.rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.ingest("app/data/AI Engineering Guidebook.pdf")

    # Query (loads saved index, runs retrieval + generation)
    result = pipeline.query("What is a RAG pipeline?")
    print(result["answer"])

Run directly for a quick CLI demo:
    python -m app.rag.pipeline
"""

import logging
import sys
from pathlib import Path
from typing import List

# ── Path bootstrap (allows running as `python pipeline.py` directly) ───────────
# Inserts the project root (two levels up from this file) onto sys.path so that
# absolute imports like `from app.xxx import yyy` resolve correctly regardless
# of which directory the script is invoked from.
_project_root = Path(__file__).resolve().parents[2]  # …/AI Assistant - Agentic System
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# ──────────────────────────────────────────────────────────────────────────────

from app.constant.config import VECTOR_STORE_DIR
from app.tools.file_tool import pdf_parse
from app.rag.chunking import split_documents
from app.rag.embunding import embed_documents
from app.rag.vector_store import FAISSVectorStore
from app.rag.retrival import retrieve_and_answer

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    High-level orchestrator for the full RAG workflow.

    Stages:
      Load PDF → Chunk → Embed → FAISS index → (persist)
      Query    → Embed → Search → LLM generate → Answer
    """

    def __init__(self) -> None:
        self.vector_store = FAISSVectorStore()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, *file_paths: str, force: bool = False) -> None:
        """
        Load, chunk, embed, and index one or more PDF files.

        Args:
            *file_paths: One or more paths to PDF documents.
            force:       If True, re-ingest even if a saved index exists.
        """
        # Skip if already indexed and not forced
        if not force and (VECTOR_STORE_DIR / "index.faiss").exists():
            logger.info(
                "Saved index found. Loading from disk (pass force=True to re-ingest)."
            )
            self.vector_store.load()
            return

        all_documents = []
        for fp in file_paths:
            logger.info(f"── Ingesting: {fp}")
            raw_docs = pdf_parse(fp)
            all_documents.extend(raw_docs)

        if not all_documents:
            raise ValueError("No documents were loaded. Check your file paths.")

        # Chunk
        logger.info("── Chunking documents…")
        chunks = split_documents(all_documents)

        # Embed
        logger.info("── Embedding chunks…")
        chunks, embeddings = embed_documents(chunks)

        # Index
        logger.info("── Building FAISS index…")
        self.vector_store.build(chunks, embeddings)
        self.vector_store.save()

        logger.info(
            f"✅ Ingestion complete — {len(chunks)} chunks indexed from {len(file_paths)} file(s)."
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a natural language question using the indexed documents.

        Args:
            question: The user's question.
            top_k:    Number of context chunks to retrieve.

        Returns:
            Dict with "query", "answer", and "sources".
        """
        # Auto-load index if not yet in memory
        if not self.vector_store.is_ready:
            logger.info("Vector store not loaded — loading from disk…")
            self.vector_store.load()

        return retrieve_and_answer(question, self.vector_store, top_k=top_k)

    def query_interactive(self) -> None:
        """Launch an interactive Q&A loop in the terminal."""
        if not self.vector_store.is_ready:
            self.vector_store.load()

        print("\n" + "═" * 60)
        print("  🤖  RAG Q&A — type 'exit' to quit")
        print("═" * 60 + "\n")

        while True:
            try:
                question = input("❓ Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            if not question:
                continue

            result = self.query(question)

            print(f"\n💡 Answer:\n{result['answer']}\n")
            print("📄 Sources:")
            for i, src in enumerate(result["sources"], 1):
                print(
                    f"  [{i}] {src['source']} | page {src['page']} "
                    f"| score {src['score']:.3f}"
                )
                print(f"       …{src['text_snippet']}…")
            print()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument(
        "--ingest",
        nargs="+",
        metavar="PDF",
        help="PDF file(s) to ingest into the vector store",
    )
    parser.add_argument(
        "--query",
        metavar="QUESTION",
        help="Single question to answer (non-interactive)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive Q&A loop",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if index already exists",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()

    # ── Ingestion ──
    if args.ingest:
        pipeline.ingest(*args.ingest, force=args.force)

    # ── Query modes ──
    if args.query:
        result = pipeline.query(args.query)
        print(f"\n💡 Answer:\n{result['answer']}\n")
        print("📄 Sources:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['source']} | page {src['page']} | score {src['score']:.3f}")
            print(f"       …{src['text_snippet']}…")

    elif args.interactive or not args.ingest:
        pipeline.query_interactive()
