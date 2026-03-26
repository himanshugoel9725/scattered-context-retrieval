"""Tests for scatter factor computation."""

import pytest
from src.evaluation.metrics import compute_scatter_factor
from src.indexing.chunker import ChunkMeta


def _make_chunk(chunk_id, idx, total):
    return ChunkMeta(
        chunk_id=chunk_id, doc_id="doc1", text=f"text {idx}",
        start_char=idx * 100, end_char=(idx + 1) * 100,
        token_count=50, chunk_index=idx, total_chunks=total,
    )


def test_sf_empty():
    assert compute_scatter_factor("e", [], {}) == 0.0


def test_sf_one_chunk():
    chunks = {"c0": _make_chunk("c0", 0, 10)}
    assert compute_scatter_factor("e", ["c0"], chunks) == 0.0


def test_sf_adjacent_chunks():
    chunks = {f"c{i}": _make_chunk(f"c{i}", i, 100) for i in range(100)}
    sf_adjacent = compute_scatter_factor("e", ["c0", "c1"], chunks)
    sf_spread = compute_scatter_factor("e", ["c0", "c99"], chunks)
    assert sf_spread > sf_adjacent  # More spread = higher SF


def test_sf_monotonic_with_spread():
    total = 20
    chunks = {f"c{i}": _make_chunk(f"c{i}", i, total) for i in range(total)}

    # Low spread: adjacent chunks
    sf_low = compute_scatter_factor("e", ["c0", "c1", "c2"], chunks)
    # High spread: spread across document
    sf_high = compute_scatter_factor("e", ["c0", "c10", "c19"], chunks)

    assert sf_high > sf_low
