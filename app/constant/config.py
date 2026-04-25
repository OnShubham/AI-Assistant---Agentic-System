"""
config.py — Central configuration for the RAG pipeline.
Loads environment variables and exposes typed settings used across all modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Locate .env at project root (two levels up from this file) ──────────────
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

# ── API Keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY:   str = os.getenv("GROQ_API_KEY",   "")

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE:    int = 500
CHUNK_OVERLAP: int = 50

# ── FAISS Vector Store ────────────────────────────────────────────────────────
VECTOR_STORE_DIR: Path = _ROOT / "app" / "data" / "faiss_index"

# ── LLM Settings ─────────────────────────────────────────────────────────────
# "gemini" | "groq"
LLM_PROVIDER:  str = "gemini"
GEMINI_MODEL:  str = "gemini-2.5-flash"
GROQ_MODEL:    str = "llama3-8b-8192"
MAX_TOKENS:    int = 1024
TEMPERATURE:   float = 0.2

# ── Retrieval Settings ────────────────────────────────────────────────────────
TOP_K_RESULTS: int = 5
