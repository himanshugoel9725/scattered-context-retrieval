"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    compute_scatter_factor,
    scatter_coverage_at_k,
    retrieval_completeness_at_k,
    information_completeness_score,
)
from src.evaluation.gold_schema import GoldEvidence, QueryMetadata
from src.evaluation.iaa import cohens_kappa
from src.indexing.chunker import ChunkMeta


def _make_chunk(chunk_id, idx, total=10):
    return ChunkMeta(
        chunk_id=chunk_id, doc_id="doc1", text=f"chunk {idx}",
        start_char=idx * 100, end_char=(idx + 1) * 100,
        token_count=50, chunk_index=idx, total_chunks=total,
    )


def test_scatter_factor_single_chunk():
    chunks = {"c0": _make_chunk("c0", 0)}
    sf = compute_scatter_factor("entity", ["c0"], chunks)
    assert sf == 0.0  # Single chunk => no scatter


def test_scatter_factor_multiple_chunks():
    chunks = {f"c{i}": _make_chunk(f"c{i}", i) for i in range(10)}
    sf = compute_scatter_factor("entity", ["c0", "c5", "c9"], chunks)
    assert sf > 0.0  # Spread across document


def test_scatter_coverage():
    chunks = {f"c{i}": _make_chunk(f"c{i}", i) for i in range(10)}
    # Retrieve chunks from deciles 0, 5, 9
    sc = scatter_coverage_at_k(["c0", "c5", "c9"], chunks, n_deciles=10)
    assert sc == 0.3  # 3 out of 10 deciles


def test_retrieval_completeness():
    gold = GoldEvidence(
        query_id="q1", entity_id="e1",
        gold_chunk_ids=["c0", "c1", "c2", "c3"],
    )
    rc = retrieval_completeness_at_k(["c0", "c1", "c5"], gold)
    assert rc == 0.5  # 2 of 4 gold chunks retrieved


def test_ics():
    gold_attrs = {"appearance": "tall", "background": "detective", "personality": "clever"}
    ics = information_completeness_score(["appearance", "personality"], gold_attrs)
    assert abs(ics - 2 / 3) < 0.01


def test_cohens_kappa_perfect():
    k = cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0])
    assert k == 1.0


def test_cohens_kappa_zero():
    # Random-ish agreement
    k = cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1])
    assert k < 0.1  # Should be near 0 or negative
