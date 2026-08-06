"""Central configuration for the RAG project."""

from pathlib import Path


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


# Local language model
OLLAMA_MODEL_NAME = "qwen3.5:4b-q4_K_M"


# Chunking settings
CHUNK_SIZE = 350
CHUNK_OVERLAP = 80


# Retrieval settings
RESULTS_PER_QUERY = 8
FINAL_TOP_K = 6
RRF_K = 60  