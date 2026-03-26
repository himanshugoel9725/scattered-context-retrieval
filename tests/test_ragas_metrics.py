"""Tests for RAGAS evaluation metrics wrapper."""

import math
from unittest.mock import patch, MagicMock

import pytest

from src.evaluation.ragas_metrics import compute_ragas_metrics


@pytest.fixture(autouse=True)
def _clear_ragas_singletons():
    """Reset lazy-initialized globals between tests."""
    import src.evaluation.ragas_metrics as mod
    mod._evaluator_llm = None
    mod._evaluator_embeddings = None
    mod._metrics = None
    yield
    mod._evaluator_llm = None
    mod._evaluator_embeddings = None
    mod._metrics = None


@patch("src.evaluation.ragas_metrics.cached_call")
def test_compute_ragas_metrics_returns_all_four(mock_cached_call):
    """All 4 default metrics are returned with float values."""
    mock_cached_call.side_effect = [0.85, 0.72, 0.91, 0.68]

    result = compute_ragas_metrics(
        question="What is the main character's motivation?",
        answer="The main character is driven by revenge.",
        contexts=["Passage about the character's backstory.", "Passage about conflict."],
        ground_truth="The protagonist seeks revenge for their family's betrayal.",
    )

    assert set(result.keys()) == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    }
    assert result["faithfulness"] == 0.85
    assert result["answer_relevancy"] == 0.72
    assert result["context_precision"] == 0.91
    assert result["context_recall"] == 0.68
    assert mock_cached_call.call_count == 4


@patch("src.evaluation.ragas_metrics.cached_call")
def test_compute_ragas_metrics_subset(mock_cached_call):
    """Only requested metrics are computed."""
    mock_cached_call.return_value = 0.9

    result = compute_ragas_metrics(
        question="q", answer="a", contexts=["c"], ground_truth="g",
        metrics=["faithfulness"],
    )

    assert list(result.keys()) == ["faithfulness"]
    assert mock_cached_call.call_count == 1


@patch("src.evaluation.ragas_metrics.cached_call", side_effect=Exception("API error"))
def test_compute_ragas_metrics_failure_returns_nan(mock_cached_call):
    """Failed metrics return NaN instead of raising."""
    result = compute_ragas_metrics(
        question="q", answer="a", contexts=["c"], ground_truth="g",
        metrics=["faithfulness"],
    )

    assert math.isnan(result["faithfulness"])


@patch("src.evaluation.ragas_metrics.cached_call")
def test_compute_ragas_metrics_cache_keys_differ(mock_cached_call):
    """Each metric uses a unique cache prefix."""
    mock_cached_call.return_value = 0.5

    compute_ragas_metrics(
        question="q", answer="a", contexts=["c"], ground_truth="g",
    )

    prefixes = [call.args[0] for call in mock_cached_call.call_args_list]
    assert len(set(prefixes)) == 4  # All different
    assert all(p.startswith("ragas_") for p in prefixes)
