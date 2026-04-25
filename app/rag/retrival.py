"""
retrival.py — Retrieval + LLM generation (the "AG" in RAG).

Given a user query:
  1. Embed the query.
  2. Search the FAISS vector store for relevant document chunks.
  3. Build a grounded prompt from the retrieved context.
  4. Call the configured LLM (Gemini or Groq) to generate an answer.
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from app.constant.config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_PROVIDER,
    GEMINI_MODEL,
    GROQ_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K_RESULTS,
)
from app.rag.embunding import embed_query
from app.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_llm():
    """Instantiate the configured LLM (lazy, so imports only happen once)."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )
    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use 'gemini' or 'groq'.")


_llm = None

def get_llm():
    """Singleton LLM accessor."""
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(query: str, context_chunks: List[Tuple[Document, float]]) -> str:
    """
    Construct a grounded RAG prompt from retrieved context chunks.

    Args:
        query:         The user's question.
        context_chunks: List of (Document, score) from vector_store.search().

    Returns:
        Formatted prompt string ready for the LLM.
    """
    context_blocks = []
    for i, (doc, score) in enumerate(context_chunks, start=1):
        source   = doc.metadata.get("source", "unknown")
        page     = doc.metadata.get("page", "?")
        snippet  = doc.page_content.strip()
        context_blocks.append(
            f"[Context {i}] (source: {source}, page: {page}, score: {score:.3f})\n{snippet}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful and precise AI assistant. Answer the question below
using ONLY the provided context. If the context does not contain enough information,
say "I don't have enough information in the provided documents to answer this."

=== CONTEXT ===
{context_text}

=== QUESTION ===
{query}

=== ANSWER ==="""

    return prompt


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve_and_answer(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int = TOP_K_RESULTS,
) -> dict:
    """
    Full RAG retrieval + generation for a single query.

    Args:
        query:        The user's natural language question.
        vector_store: An initialised (built or loaded) FAISSVectorStore.
        top_k:        Number of context chunks to retrieve.

    Returns:
        Dictionary with keys:
          - "answer"   : The LLM-generated answer string.
          - "sources"  : List of source metadata dicts for each retrieved chunk.
          - "query"    : The original query.
    """
    if not vector_store.is_ready:
        raise RuntimeError("Vector store is not ready. Ingest documents first.")

    # 1. Embed query
    query_vec = embed_query(query)

    # 2. Retrieve top-k relevant chunks
    context_chunks = vector_store.search(query_vec, top_k=top_k)

    if not context_chunks:
        return {
            "query":  query,
            "answer": "No relevant documents found in the knowledge base.",
            "sources": [],
        }

    # 3. Build grounded prompt
    prompt = _build_prompt(query, context_chunks)
    logger.debug(f"Prompt sent to LLM:\n{prompt[:500]}...")

    # 4. Generate answer
    llm = get_llm()
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    # 5. Collect source metadata
    sources = []
    for doc, score in context_chunks:
        sources.append({
            "source":      doc.metadata.get("source", "unknown"),
            "page":        doc.metadata.get("page", None),
            "score":       round(score, 4),
            "text_snippet": doc.page_content[:200].strip(),
        })

    return {
        "query":   query,
        "answer":  answer.strip(),
        "sources": sources,
    }
