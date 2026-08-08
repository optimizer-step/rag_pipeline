"""Tests for answer generation and evidence formatting."""

import pytest

import src.generator as generator


def sample_evidence():
    """Return one prepared evidence item."""

    return [
        {
            "fused_rank": 1,
            "chunk_id": "page-2-chunk-1",
            "source": "policy.pdf",
            "page_number": 2,
            "document": "Sample policy evidence.",
            "rrf_score": 0.032,
            "best_distance": 0.15,
            "matched_queries": [
                "coverage question",
            ],
        }
    ]


def sample_fused_ranking():
    """Return one fused-ranking result."""

    return [
        {
            "fused_rank": 1,
            "document": "Sample policy evidence.",
            "metadata": {
                "chunk_id": "page-2-chunk-1",
                "source": "policy.pdf",
                "page_number": 2,
            },
            "rrf_score": 0.032,
            "best_distance": 0.15,
            "matched_queries": [
                "coverage question",
            ],
        }
    ]


def test_build_context_contains_metadata():
    context = generator.build_context(
        sample_evidence()
    )

    assert "PAGE: 2" in context
    assert "page-2-chunk-1" in context
    assert "Sample policy evidence." in context


def test_build_context_rejects_empty_evidence():
    with pytest.raises(
        ValueError,
        match="evidence chunk",
    ):
        generator.build_context([])


def test_build_user_prompt_contains_question():
    prompt = generator.build_user_prompt(
        question="What is covered?",
        final_evidence=sample_evidence(),
    )

    assert "What is covered?" in prompt
    assert "Sample policy evidence." in prompt
    assert "[Page X]" in prompt


def test_prepare_evidence_formats_result():
    evidence = generator.prepare_evidence(
        sample_fused_ranking()
    )

    assert len(evidence) == 1
    assert evidence[0]["page_number"] == 2
    assert (
        evidence[0]["chunk_id"]
        == "page-2-chunk-1"
    )
    assert evidence[0]["rrf_score"] == 0.032


def test_prepare_evidence_rejects_empty_ranking():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        generator.prepare_evidence([])


def test_generate_answer_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        generator.generate_answer(
            question="   ",
            final_evidence=sample_evidence(),
        )


def test_generate_answer_uses_openai(
    monkeypatch,
):
    monkeypatch.setattr(
        generator,
        "GENERATION_PROVIDER",
        "openai",
    )

    monkeypatch.setattr(
        generator,
        "generate_with_openai",
        lambda prompt: "Grounded OpenAI answer.",
    )

    result = generator.generate_answer(
        question="What is covered?",
        final_evidence=sample_evidence(),
    )

    assert result == "Grounded OpenAI answer."


def test_generate_answer_uses_ollama(
    monkeypatch,
):
    monkeypatch.setattr(
        generator,
        "GENERATION_PROVIDER",
        "ollama",
    )

    monkeypatch.setattr(
        generator,
        "generate_with_ollama",
        lambda prompt: "Grounded Ollama answer.",
    )

    result = generator.generate_answer(
        question="What is covered?",
        final_evidence=sample_evidence(),
    )

    assert result == "Grounded Ollama answer."


def test_generate_answer_rejects_unknown_provider(
    monkeypatch,
):
    monkeypatch.setattr(
        generator,
        "GENERATION_PROVIDER",
        "unknown",
    )

    with pytest.raises(
        ValueError,
        match="openai.*ollama",
    ):
        generator.generate_answer(
            question="What is covered?",
            final_evidence=sample_evidence(),
        )