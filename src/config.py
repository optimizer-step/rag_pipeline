"""Central configuration for the RAG project."""

from pathlib import Path
import os

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "arogya_shield_policy_handbook.pdf"
)

CHROMA_PATH = PROJECT_ROOT / "chroma_db"


# ChromaDB settings
COLLECTION_NAME = "arogya_shield_policy"


# Embedding model
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "multi-qa-MiniLM-L6-cos-v1"
)





# Chunking settings
CHUNK_SIZE = 350
CHUNK_OVERLAP = 80


# Retrieval settings
RESULTS_PER_QUERY = 8
FINAL_TOP_K = 6
RRF_K = 60  


GENERATION_PROVIDER = os.getenv(
    "GENERATION_PROVIDER",
    "openai",
).strip().lower()

OPENAI_MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna",
).strip()

OLLAMA_MODEL_NAME = os.getenv(
    "OLLAMA_MODEL_NAME",
    "qwen3.5:4b-q4_K_M",
).strip()