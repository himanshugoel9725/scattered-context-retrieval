"""Tests for robustness hardening changes.

Tests for:
- Cache key collision fix (full SHA256)
- Entity word-boundary matching
- LLM judge NaN fallback
- Statistical testing utilities
"""

import math

import numpy as np
import pytest


class TestCacheKeyLength:
    """Verify cache keys use full SHA256 (64 hex chars) not truncated 16."""

    def test_full_sha256_key(self):
        from src.utils.cache import make_cache_key
        key = make_cache_key("test", question="hello", answer="world")
        prefix, h = key.split(":", 1)
        assert prefix == "test"
        assert len(h) == 64, f"Expected 64 hex chars, got {len(h)}: {h}"

    def test_keys_differ_for_different_inputs(self):
        from src.utils.cache import make_cache_key
        k1 = make_cache_key("llm", q="What is A?")
        k2 = make_cache_key("llm", q="What is B?")
        assert k1 != k2

    def test_keys_stable_across_calls(self):
        from src.utils.cache import make_cache_key
        k1 = make_cache_key("test", x=1, y=2)
        k2 = make_cache_key("test", x=1, y=2)
        assert k1 == k2


class TestEntityWordBoundary:
    """Verify entity matching uses word boundaries, not substring."""

    def test_cat_does_not_match_category(self):
        from src.retrieval.base import detect_query_entities

        class MockEntityIndex:
            entities = {"cat": {"canonical": "cat", "aliases": []}}

        result = detect_query_entities(
            "What category does this fall into?",
            entity_index=MockEntityIndex(),
            entity_types=()  # skip spaCy NER
        )
        assert "cat" not in result

    def test_art_does_not_match_party(self):
        from src.retrieval.base import detect_query_entities

        class MockEntityIndex:
            entities = {"art": {"canonical": "art", "aliases": []}}

        result = detect_query_entities(
            "The party was held downtown",
            entity_index=MockEntityIndex(),
            entity_types=()
        )
        assert "art" not in result

    def test_exact_word_matches(self):
        from src.retrieval.base import detect_query_entities

        class MockEntityIndex:
            entities = {"Smith": {"canonical": "Smith", "aliases": []}}

        result = detect_query_entities(
            "Tell me about Smith and his work",
            entity_index=MockEntityIndex(),
            entity_types=()
        )
        assert "Smith" in result

    def test_alias_word_boundary(self):
        from src.retrieval.base import detect_query_entities

        class MockEntityIndex:
            entities = {"John Smith": {"canonical": "John Smith",
                                        "aliases": ["JS"]}}

        result = detect_query_entities(
            "What did JS contribute?",
            entity_index=MockEntityIndex(),
            entity_types=()
        )
        assert "John Smith" in result

    def test_alias_no_false_positive(self):
        from src.retrieval.base import detect_query_entities

        class MockEntityIndex:
            entities = {"John Smith": {"canonical": "John Smith",
                                        "aliases": ["JS"]}}

        result = detect_query_entities(
            "What did JSA contribute?",
            entity_index=MockEntityIndex(),
            entity_types=()
        )
        assert "John Smith" not in result


class TestLLMJudgeNaNFallback:
    """Verify LLM judge returns NaN on parse failure, not (3,3,3)."""

    def test_parse_failure_returns_nan(self):
        from unittest.mock import patch
        from src.evaluation.llm_judge import llm_judge_score

        with patch("src.evaluation.llm_judge.generate", return_value="I can't rate this sorry"):
            result = llm_judge_score("q", "a", "ref")
            assert math.isnan(result["completeness"])
            assert math.isnan(result["accuracy"])
            assert math.isnan(result["coherence"])
            assert math.isnan(result["overall"])
            assert result.get("parse_failed") is True

    def test_valid_json_works(self):
        from unittest.mock import patch
        from src.evaluation.llm_judge import llm_judge_score

        mock_resp = '{"completeness": 4, "accuracy": 5, "coherence": 3}'
        with patch("src.evaluation.llm_judge.generate", return_value=mock_resp):
            result = llm_judge_score("q", "a", "ref")
            assert result["completeness"] == 4
            assert result["accuracy"] == 5
            assert result["coherence"] == 3
            assert abs(result["overall"] - 4.0) < 0.01
            assert "parse_failed" not in result


class TestStatistics:
    """Test statistical utilities."""

    def test_bootstrap_same_distribution(self):
        from src.evaluation.statistics import paired_bootstrap_test
        rng = np.random.RandomState(42)
        a = rng.normal(0.5, 0.1, 100).tolist()
        b = rng.normal(0.5, 0.1, 100).tolist()
        result = paired_bootstrap_test(a, b)
        assert result.p_value > 0.05  # should NOT be significant

    def test_bootstrap_different_distributions(self):
        from src.evaluation.statistics import paired_bootstrap_test
        rng = np.random.RandomState(42)
        a = rng.normal(0.7, 0.1, 100).tolist()
        b = rng.normal(0.3, 0.1, 100).tolist()
        result = paired_bootstrap_test(a, b)
        assert result.p_value < 0.05  # should be significant
        assert result.observed_diff > 0

    def test_effect_size_large(self):
        from src.evaluation.statistics import compute_effect_size
        a = [0.8, 0.9, 0.85, 0.87, 0.92]
        b = [0.2, 0.3, 0.25, 0.27, 0.22]
        d = compute_effect_size(a, b)
        assert d > 0.8  # large effect size

    def test_effect_size_nan_with_no_variance(self):
        from src.evaluation.statistics import compute_effect_size
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        d = compute_effect_size(a, b)
        assert math.isnan(d)

    def test_summarize_with_ci(self):
        from src.evaluation.statistics import summarize_with_ci
        scores = [0.5, 0.6, 0.55, 0.52, 0.58, 0.62, 0.48, 0.53, 0.57, 0.61]
        result = summarize_with_ci(scores)
        assert result.n == 10
        assert 0.5 < result.mean < 0.6
        assert result.ci_lower < result.mean < result.ci_upper
        assert result.std > 0

    def test_summarize_with_nans(self):
        from src.evaluation.statistics import summarize_with_ci
        scores = [0.5, float("nan"), 0.6, float("nan"), 0.55]
        result = summarize_with_ci(scores)
        assert result.n == 3  # only non-NaN values

    def test_significance_annotation(self):
        from src.evaluation.statistics import significance_annotation
        assert significance_annotation(0.0001) == "***"
        assert significance_annotation(0.005) == "**"
        assert significance_annotation(0.03) == "*"
        assert significance_annotation(0.1) == "n.s."


class TestDataDedup:
    """Verify data loader deduplication."""

    def test_load_dataset_deduplicates(self):
        from unittest.mock import patch
        from src.data.loaders import load_dataset, Document

        duped_docs = [
            Document(doc_id="doc1", text="hello"),
            Document(doc_id="doc1", text="hello"),
            Document(doc_id="doc2", text="world"),
        ]
        with patch("src.data.loaders.LOADERS", {"test": lambda **kw: duped_docs}):
            result = load_dataset("test")
            assert len(result) == 2
            assert result[0].doc_id == "doc1"
            assert result[1].doc_id == "doc2"
