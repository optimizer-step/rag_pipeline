"""Combine multiple retrieval rankings using RRF."""

from typing import Any

from .config import RRF_K


def reciprocal_rank_fusion(
    queries: list[str],
    query_result: dict[str, Any],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse Chroma result lists into one ranking."""

    if not queries:
        raise ValueError(
            "At least one query is required for RRF."
        )

    if rrf_k <= 0:
        raise ValueError(
            "rrf_k must be greater than zero."
        )

    required_keys = {
        "documents",
        "metadatas",
        "distances",
    }

    if not required_keys.issubset(query_result):
        raise ValueError(
            "query_result is missing required fields."
        )

    if len(query_result["documents"]) != len(queries):
        raise ValueError(
            "The number of result groups must match "
            "the number of queries."
        )

    fused_results: dict[
        str,
        dict[str, Any],
    ] = {}

    for query_index, query in enumerate(queries):
        documents = query_result["documents"][
            query_index
        ]
        metadatas = query_result["metadatas"][
            query_index
        ]
        distances = query_result["distances"][
            query_index
        ]

        if not (
            len(documents)
            == len(metadatas)
            == len(distances)
        ):
            raise ValueError(
                "Documents, metadatas and distances "
                "must have matching lengths."
            )

        for rank, (
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):
            chunk_id = metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Every retrieved chunk must have "
                    "a chunk_id."
                )

            distance = float(distance)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "document": document,
                    "metadata": metadata,
                    "rrf_score": 0.0,
                    "best_distance": distance,
                    "matched_queries": [],
                    "appearances": 0,
                }

            fused_results[chunk_id][
                "rrf_score"
            ] += 1 / (rrf_k + rank)

            fused_results[chunk_id][
                "best_distance"
            ] = min(
                fused_results[chunk_id][
                    "best_distance"
                ],
                distance,
            )

            fused_results[chunk_id][
                "appearances"
            ] += 1

            if query not in fused_results[
                chunk_id
            ]["matched_queries"]:
                fused_results[chunk_id][
                    "matched_queries"
                ].append(query)

    fused_ranking = sorted(
        fused_results.values(),
        key=lambda result: (
            -result["rrf_score"],
            result["best_distance"],
        ),
    )

    for fused_rank, result in enumerate(
        fused_ranking,
        start=1,
    ):
        result["fused_rank"] = fused_rank

    return fused_ranking