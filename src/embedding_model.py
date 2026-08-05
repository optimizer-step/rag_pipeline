"""Create local embeddings for documents and queries."""

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = "cpu",
) -> SentenceTransformer:
    """Load the SentenceTransformer embedding model."""

    if not model_name.strip():
        raise ValueError(
            "The embedding model name cannot be empty."
        )

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    return model


def create_document_embeddings(
    model: SentenceTransformer,
    chunks: list[dict],
    batch_size: int = 32,
) -> np.ndarray:
    """Convert all document chunks into normalized vectors."""

    if not chunks:
        raise ValueError(
            "The chunks list cannot be empty."
        )

    chunk_texts = [
        chunk["text"].strip()
        for chunk in chunks
    ]

    if any(not text for text in chunk_texts):
        raise ValueError(
            "Every chunk must contain readable text."
        )

    embeddings = model.encode_document(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "The number of embeddings does not "
            "match the number of chunks."
        )

    return embeddings


def create_query_embeddings(
    model: SentenceTransformer,
    queries: list[str],
) -> np.ndarray:
    """Convert one or more search queries into vectors."""

    cleaned_queries = [
        query.strip()
        for query in queries
        if query.strip()
    ]

    if not cleaned_queries:
        raise ValueError(
            "At least one query is required."
        )

    embeddings = model.encode_query(
        cleaned_queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings