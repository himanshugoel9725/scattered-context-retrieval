"""Tests for entity index functionality."""

import pytest
from src.indexing.chunker import Chunker, ChunkMeta
from src.indexing.entity_index import EntityIndex


@pytest.fixture
def sample_text():
    return (
        "Sherlock Holmes lived at 221B Baker Street in London. "
        "He was a brilliant detective who solved many cases. "
        "Dr. Watson was his loyal companion and friend. "
        "Watson documented their adventures in numerous stories. "
        "Holmes often played his violin late at night. "
        "Mrs. Hudson was their long-suffering landlady. "
        "Professor Moriarty was the criminal mastermind who opposed Holmes. "
        "Moriarty controlled a vast criminal network across Europe. "
        "Inspector Lestrade often sought Holmes's help at Scotland Yard. "
        "Holmes used his method of deduction to solve the most baffling mysteries."
    )


@pytest.fixture
def chunks(sample_text):
    chunker = Chunker(chunk_size=50, overlap=10)
    return chunker.chunk_document("test_doc", sample_text)


def test_chunker_basic(sample_text):
    chunker = Chunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_document("test_doc", sample_text)
    assert len(chunks) > 0
    assert all(isinstance(c, ChunkMeta) for c in chunks)
    assert all(c.doc_id == "test_doc" for c in chunks)
    assert chunks[0].chunk_index == 0
    assert all(c.total_chunks == len(chunks) for c in chunks)


def test_entity_index_build(chunks):
    entity_idx = EntityIndex("test_doc")
    entity_idx.build_from_chunks(chunks)
    entities = entity_idx.entities
    assert len(entities) > 0
    # Should find at least Holmes or Watson
    entity_names_lower = set(entities.keys())
    found_known = any("holmes" in n or "watson" in n or "sherlock" in n
                      for n in entity_names_lower)
    assert found_known, f"Expected to find Holmes/Watson in {entity_names_lower}"


def test_entity_chunk_mapping(chunks):
    entity_idx = EntityIndex("test_doc")
    entity_idx.build_from_chunks(chunks)

    for ename, edata in entity_idx.entities.items():
        assert len(edata["chunk_ids"]) > 0
        assert edata["mention_count"] > 0


def test_get_entities_in_chunk(chunks):
    entity_idx = EntityIndex("test_doc")
    entity_idx.build_from_chunks(chunks)

    # At least one chunk should have entities
    found_any = False
    for c in chunks:
        ents = entity_idx.get_entities_in_chunk(c.chunk_id)
        if ents:
            found_any = True
            break
    assert found_any


def test_chunk_position_fraction(sample_text):
    chunker = Chunker(chunk_size=30, overlap=5)
    chunks = chunker.chunk_document("test_doc", sample_text)
    if len(chunks) > 1:
        assert chunks[0].position_fraction == 0.0
        assert chunks[-1].position_fraction == pytest.approx(1.0, abs=0.01)
