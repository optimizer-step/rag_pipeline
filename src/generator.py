"""Prepare retrieved evidence and generate a grounded answer."""

import os
from typing import Any

from ollama import chat
from openai import OpenAI

from .config import (
    FINAL_TOP_K,
    GENERATION_PROVIDER,
    OLLAMA_MODEL_NAME,
    OPENAI_MODEL_NAME,
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


def build_user_prompt(
    question: str,
    final_evidence: list[dict[str, Any]],
) -> str:
    

    context = build_context(
        final_evidence
    )

    return f"""
USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Answer the question using only the retrieved evidence.
Cite every supported claim using [Page X].
""".strip()


def generate_with_openai(
    user_prompt: str,
    model_name: str = OPENAI_MODEL_NAME,
) -> str:
    

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required when "
            "GENERATION_PROVIDER is set to openai."
        )

    client = OpenAI()

    response = client.responses.create(
        model=model_name,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    return response.output_text.strip()


def generate_with_ollama(
    user_prompt: str,
    model_name: str = OLLAMA_MODEL_NAME,
) -> str:
    """Generate an answer using local Ollama."""

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

    return response.message.content.strip()


def generate_answer(
    question: str,
    final_evidence: list[dict[str, Any]],
) -> str:
    """Generate an answer using the configured provider."""

    question = question.strip()

    if not question:
        raise ValueError(
            "The user question cannot be empty."
        )

    user_prompt = build_user_prompt(
        question=question,
        final_evidence=final_evidence,
    )

    if GENERATION_PROVIDER == "openai":
        answer = generate_with_openai(
            user_prompt
        )
    elif GENERATION_PROVIDER == "ollama":
        answer = generate_with_ollama(
            user_prompt
        )
    else:
        raise ValueError(
            "GENERATION_PROVIDER must be "
            "'openai' or 'ollama'."
        )

    if not answer:
        raise ValueError(
            "The language model returned an empty answer."
        )

    return answer