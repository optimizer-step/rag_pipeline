"""Retrieve relevant document chunks from ChromaDB."""

from typing import Any

from sentence_transformers import SentenceTransformer

from .config import RESULTS_PER_QUERY
from .embedding_model import create_query_embeddings


def retrieve_for_queries(
    queries: list[str],
    collection,
    embedding_model: SentenceTransformer,
    results_per_query: int = RESULTS_PER_QUERY,
) -> dict[str, Any]:
    

    if not queries:
        raise ValueError(
            "At least one retrieval query is required."
        )

    cleaned_queries = [
        query.strip()
        for query in queries
    ]

    if any(not query for query in cleaned_queries):
        raise ValueError(
            "Retrieval queries cannot be empty."
        )

    if results_per_query <= 0:
        raise ValueError(
            "results_per_query must be greater than zero."
        )

    collection_size = collection.count()

    if collection_size == 0:
        raise ValueError(
            "The Chroma collection is empty. "
            "Build the vector database first."
        )

    query_embeddings = create_query_embeddings(
        model=embedding_model,
        queries=cleaned_queries,
    )

    number_of_results = min(
        results_per_query,
        collection_size,
    )

    query_result = collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=number_of_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return query_result