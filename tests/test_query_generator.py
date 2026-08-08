"""Tests for retrieval-query generation."""

import pytest

import src.query_generator as query_generator


def test_clean_queries_includes_original_question():
    result = query_generator.clean_queries(
        question="What is covered?",
        generated_queries=[
            "Coverage details",
            "Policy benefits",
        ],
    )

    assert result == [
        "What is covered?",
        "Coverage details",
        "Policy benefits",
    ]


def test_clean_queries_removes_duplicates():
    result = query_generator.clean_queries(
        question="What is covered?",
        generated_queries=[
            "what is covered?",
            "Coverage details",
            "Coverage details",
        ],
    )

    assert result == [
        "What is covered?",
        "Coverage details",
    ]


def test_clean_queries_removes_empty_values():
    result = query_generator.clean_queries(
        question="What is covered?",
        generated_queries=[
            "",
            "   ",
            "Policy benefits",
        ],
    )

    assert result == [
        "What is covered?",
        "Policy benefits",
    ]


def test_clean_queries_respects_maximum():
    result = query_generator.clean_queries(
        question="Original question",
        generated_queries=[
            "Query one",
            "Query two",
            "Query three",
        ],
        maximum=3,
    )

    assert result == [
        "Original question",
        "Query one",
        "Query two",
    ]


def test_clean_queries_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        query_generator.clean_queries(
            question="   ",
            generated_queries=[],
        )


def test_non_openai_provider_uses_original_question(
    monkeypatch,
):
    monkeypatch.setattr(
        query_generator,
        "GENERATION_PROVIDER",
        "ollama",
    )

    result = (
        query_generator.generate_retrieval_queries(
            "What is the ambulance benefit?"
        )
    )

    assert result == [
        "What is the ambulance benefit?"
    ]


def test_openai_queries_are_cleaned(
    monkeypatch,
):
    monkeypatch.setattr(
        query_generator,
        "GENERATION_PROVIDER",
        "openai",
    )

    monkeypatch.setattr(
        query_generator,
        "generate_openai_queries",
        lambda question: [
            "Ambulance response time",
            "ambulance response time",
            "Service failure refund",
        ],
    )

    result = (
        query_generator.generate_retrieval_queries(
            "What is the ambulance benefit?"
        )
    )

    assert result == [
        "What is the ambulance benefit?",
        "Ambulance response time",
        "Service failure refund",
    ]


def test_openai_failure_uses_original_question(
    monkeypatch,
):
    monkeypatch.setattr(
        query_generator,
        "GENERATION_PROVIDER",
        "openai",
    )

    def raise_error(question):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(
        query_generator,
        "generate_openai_queries",
        raise_error,
    )

    result = (
        query_generator.generate_retrieval_queries(
            "What is covered?"
        )
    )

    assert result == [
        "What is covered?"
    ]