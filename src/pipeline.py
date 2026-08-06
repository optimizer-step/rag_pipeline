"""Connect all stages of the modular RAG pipeline."""

from typing import Any

from .chunker import (
    chunk_pages,
    create_text_splitter,
)
from .config import PDF_PATH
from .document_loader import load_pdf_pages
from .embedding_model import (
    create_document_embeddings,
    load_embedding_model,
)
from .generator import (
    generate_answer,
    prepare_evidence,
)
from .rank_fusion import reciprocal_rank_fusion
from .retriever import retrieve_for_queries
from .vector_store import (
    create_vector_store,
    store_chunks,
)


def build_database(
    rebuild: bool = False,
):
    """Create or reuse the document vector database."""

    embedding_model = load_embedding_model()

    collection = create_vector_store(
        reset=rebuild
    )

    database_is_empty = collection.count() == 0

    if rebuild or database_is_empty:
        print("Building the vector database...")

        pages = load_pdf_pages(
            PDF_PATH
        )

        splitter = create_text_splitter()

        chunks = chunk_pages(
            pages=pages,
            splitter=splitter,
        )

        embeddings = create_document_embeddings(
            model=embedding_model,
            chunks=chunks,
        )

        stored_count = store_chunks(
            collection=collection,
            chunks=chunks,
            embeddings=embeddings,
        )

        print("Pages loaded:", len(pages))
        print("Chunks created:", len(chunks))
        print("Vectors stored:", stored_count)

    else:
        print(
            "Using the existing Chroma database."
        )
        print(
            "Stored vectors:",
            collection.count(),
        )

    return collection, embedding_model


def run_rag(
    question: str,
    queries: list[str],
    collection,
    embedding_model,
) -> dict[str, Any]:
    """Run retrieval, fusion and answer generation."""

    question = question.strip()

    if not question:
        raise ValueError(
            "The user question cannot be empty."
        )

    if not queries:
        raise ValueError(
            "At least one retrieval query is required."
        )

    query_result = retrieve_for_queries(
        queries=queries,
        collection=collection,
        embedding_model=embedding_model,
    )

    fused_ranking = reciprocal_rank_fusion(
        queries=queries,
        query_result=query_result,
    )

    final_evidence = prepare_evidence(
        fused_ranking=fused_ranking,
    )

    answer = generate_answer(
        question=question,
        final_evidence=final_evidence,
    )

    return {
        "question": question,
        "queries": queries,
        "answer": answer,
        "evidence": final_evidence,
        "fused_ranking": fused_ranking,
    }