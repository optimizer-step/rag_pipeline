"""Generate retrieval-query variations."""

import json
import os

from openai import OpenAI

from .config import (
    GENERATION_PROVIDER,
    OPENAI_MODEL_NAME,
)


def clean_queries(
    question: str,
    generated_queries: list[str],
    maximum: int = 5,
) -> list[str]:
    """Normalize, deduplicate, and limit queries."""

    question = question.strip()

    if not question:
        raise ValueError(
            "The user question cannot be empty."
        )

    candidates = [
        question,
        *generated_queries,
    ]

    cleaned = []
    seen = set()

    for candidate in candidates:
        candidate = str(candidate).strip()

        if not candidate:
            continue

        normalized = candidate.casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(candidate)

        if len(cleaned) >= maximum:
            break

    return cleaned


def generate_openai_queries(
    question: str,
) -> list[str]:
    """Generate retrieval queries using OpenAI."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for "
            "OpenAI query generation."
        )

    client = OpenAI()

    prompt = f"""
Create four focused semantic-search queries for the
following insurance-policy question.

Each query should target a distinct fact that may be
needed to answer the question.

Return only a valid JSON array of strings.
Do not include Markdown or explanations.

Question:
{question}
""".strip()

    response = client.responses.create(
        model=OPENAI_MODEL_NAME,
        instructions=(
            "You generate concise retrieval queries "
            "for an insurance-policy search system."
        ),
        input=prompt,
    )

    parsed_queries = json.loads(
        response.output_text
    )

    if not isinstance(parsed_queries, list):
        raise ValueError(
            "Query generation did not return a list."
        )

    return [
        str(query)
        for query in parsed_queries
    ]


def generate_retrieval_queries(
    question: str,
) -> list[str]:
    """Generate queries with an original-query fallback."""

    question = question.strip()

    if not question:
        raise ValueError(
            "The user question cannot be empty."
        )

    if GENERATION_PROVIDER != "openai":
        return [question]

    try:
        generated_queries = generate_openai_queries(
            question
        )

        return clean_queries(
            question=question,
            generated_queries=generated_queries,
        )
    except Exception:
        return [question]