"""Store document chunks and embeddings in ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from .config import (
    CHROMA_PATH,
    COLLECTION_NAME,
)


def create_vector_store(
    chroma_path: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
):
    

    chroma_path = Path(chroma_path).resolve()

    client = chromadb.PersistentClient(
        path=str(chroma_path)
    )

    existing_collections = {
        collection.name
        for collection in client.list_collections()
    }

    if (
        reset
        and collection_name in existing_collections
    ):
        client.delete_collection(
            name=collection_name
        )

    collection = client.get_or_create_collection(
        name=collection_name,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )

    return collection


def prepare_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    

    return {
        key: value
        for key, value in metadata.items()
        if isinstance(
            value,
            (str, int, float, bool),
        )
    }


def store_chunks(
    collection,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> int:
    """Store chunk text, metadata and embeddings."""

    if not chunks:
        raise ValueError(
            "The chunks list cannot be empty."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "The number of chunks must match "
            "the number of embedding rows."
        )

    chunk_ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs were detected."
        )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        prepare_metadata(
            chunk["metadata"]
        )
        for chunk in chunks
    ]

    collection.upsert(
        ids=chunk_ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    return collection.count()