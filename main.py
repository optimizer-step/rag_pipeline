"""Run the complete modular RAG pipeline."""

from src.pipeline import build_database, run_rag



REBUILD_DATABASE = False


def main() -> None:
    """Build the database and answer one question."""

    collection, embedding_model = build_database(
        rebuild=REBUILD_DATABASE
    )

    question = (
        "If an advanced life-support ambulance "
        "arrives after 35 minutes, does this count "
        "as a documented service failure, and what "
        "refund can the member receive?"
    )

    queries = [
        question,
        (
            "What is the committed arrival time for "
            "an advanced life-support ambulance?"
        ),
        (
            "What response time is considered a "
            "documented service failure?"
        ),
        (
            "What happens when the response time "
            "exceeds twice the committed limit?"
        ),
        (
            "What refund is provided for a "
            "documented service failure?"
        ),
    ]

    result = run_rag(
        question=question,
        queries=queries,
        collection=collection,
        embedding_model=embedding_model,
    )

    print("\n" + "=" * 80)
    print("USER QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("RETRIEVAL QUERIES")
    print("=" * 80)

    for query_number, query in enumerate(
        result["queries"],
        start=1,
    ):
        print(f"{query_number}. {query}")

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("EVIDENCE USED")
    print("=" * 80)

    for evidence_number, evidence in enumerate(
        result["evidence"],
        start=1,
    ):
        print(
            f"\nEvidence {evidence_number}"
        )
        print(
            "Page:",
            evidence["page_number"],
        )
        print(
            "Chunk ID:",
            evidence["chunk_id"],
        )
        print(
            "RRF score:",
            round(evidence["rrf_score"], 6),
        )
        print(
            "Best distance:",
            round(
                evidence["best_distance"],
                4,
            ),
        )
        print("Text:")
        print(evidence["document"])


if __name__ == "__main__":
    main()