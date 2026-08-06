"""Prepare retrieved evidence and generate a grounded answer."""

from typing import Any

from ollama import chat

from .config import (
    FINAL_TOP_K,
    OLLAMA_MODEL_NAME,
)


REFUSAL_MESSAGE = (
    "I could not find sufficient information "
    "in the uploaded document."
)


SYSTEM_PROMPT = f"""
You are PolicyGuard, an evidence-first policy assistant.

Answer the user's question using only the supplied evidence.

Rules:
1. Do not use outside knowledge.
2. If the evidence is insufficient, reply exactly:
   "{REFUSAL_MESSAGE}"
3. Cite supporting pages using [Page X].
4. Do not invent page numbers, policy terms,
   calculations or facts.
5. Keep the answer clear and concise.
6. Explain calculations when necessary.
""".strip()


def prepare_evidence(
    fused_ranking: list[dict[str, Any]],
    top_k: int = FINAL_TOP_K,
) -> list[dict[str, Any]]:
    """Select the strongest chunks from the RRF ranking."""

    if not fused_ranking:
        raise ValueError(
            "The fused ranking cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    evidence_count = min(
        top_k,
        len(fused_ranking),
    )

    final_evidence = []

    for result in fused_ranking[:evidence_count]:
        metadata = result["metadata"]

        page_number = metadata.get("page_number")
        chunk_id = metadata.get("chunk_id")

        if page_number is None:
            raise ValueError(
                "Retrieved evidence is missing "
                "a page number."
            )

        if not chunk_id:
            raise ValueError(
                "Retrieved evidence is missing "
                "a chunk ID."
            )

        final_evidence.append(
            {
                "fused_rank": result.get(
                    "fused_rank"
                ),
                "chunk_id": chunk_id,
                "source": metadata.get(
                    "source",
                    "Unknown source",
                ),
                "page_number": int(page_number),
                "document": result["document"],
                "rrf_score": float(
                    result["rrf_score"]
                ),
                "best_distance": float(
                    result["best_distance"]
                ),
                "matched_queries": list(
                    result["matched_queries"]
                ),
            }
        )

    return final_evidence


def build_context(
    final_evidence: list[dict[str, Any]],
) -> str:
    """Convert selected evidence into prompt-ready text."""

    if not final_evidence:
        raise ValueError(
            "At least one evidence chunk is required."
        )

    context_blocks = []

    for evidence_number, evidence in enumerate(
        final_evidence,
        start=1,
    ):
        context_blocks.append(
            f"[EVIDENCE {evidence_number}]\n"
            f"[SOURCE: {evidence['source']} "
            f"| PAGE: {evidence['page_number']} "
            f"| CHUNK: {evidence['chunk_id']}]\n"
            f"{evidence['document']}"
        )

    return "\n\n".join(context_blocks)


def generate_answer(
    question: str,
    final_evidence: list[dict[str, Any]],
    model_name: str = OLLAMA_MODEL_NAME,
) -> str:
    """Generate a citation-backed answer using local Qwen."""

    question = question.strip()

    if not question:
        raise ValueError(
            "The user question cannot be empty."
        )

    context = build_context(
        final_evidence
    )

    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Answer the question using only the retrieved evidence.
Cite every supported claim using [Page X].
""".strip()

    response = chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        think=False,
        options={
            "temperature": 0.1,
        },
    )

    answer = response.message.content.strip()

    if not answer:
        raise ValueError(
            "The local model returned an empty answer."
        )

    return answer