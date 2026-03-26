"""Tests for retrieval strategies."""

import pytest
from unittest.mock import MagicMock
from src.indexing.chunker import ChunkMeta
from src.retrieval.base import detect_query_entities, RetrievalResult


def _make_chunk(chunk_id, text, idx=0, total=10):
    return ChunkMeta(
        chunk_id=chunk_id, doc_id="doc1", text=text,
        start_char=0, end_char=len(text), token_count=50,
        chunk_index=idx, total_chunks=total,
    )


def test_detect_query_entities_basic():
    """Test NER detection on a simple query."""
    entities = detect_query_entities("What did Sherlock Holmes do at Baker Street?")
    # spaCy should detect at least Sherlock Holmes
    names_lower = [e.lower() for e in entities]
    assert any("sherlock" in n or "holmes" in n for n in names_lower)


def test_retrieval_result_dataclass():
    r = RetrievalResult(chunk_id="c1", text="hello", score=0.9, source="test")
    assert r.chunk_id == "c1"
    assert r.score == 0.9
    assert r.source == "test"
