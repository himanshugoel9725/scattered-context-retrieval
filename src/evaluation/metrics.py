"""Novel metrics: ICS, Scatter Coverage@K, Retrieval Completeness@K, Scatter Factor."""

import math
from collections import defaultdict

import numpy as np

from src.indexing.chunker import ChunkMeta
from src.evaluation.gold_schema import GoldEvidence


def compute_scatter_factor(entity_name: str, chunk_ids: list[str],
                           chunks: dict[str, ChunkMeta]) -> float:
    """Compute Scatter Factor for an entity.

    SF = (N_chunks × avg_pairwise_distance) / doc_length

    Where distance = |chunk_index_i - chunk_index_j| / total_chunks for the doc.
    Higher SF means more scattered information.
    """
    if len(chunk_ids) < 2:
        return 0.0

    entity_chunks = [chunks[cid] for cid in chunk_ids if cid in chunks]
    if len(entity_chunks) < 2:
        return 0.0

    # Compute average pairwise distance using chunk position fractions
    positions = sorted([c.position_fraction for c in entity_chunks])
    total_pairs = 0
    total_distance = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total_distance += abs(positions[j] - positions[i])
            total_pairs += 1

    avg_distance = total_distance / total_pairs if total_pairs > 0 else 0.0
    n_chunks = len(entity_chunks)

    # Normalize by document size
    total_doc_chunks = entity_chunks[0].total_chunks if entity_chunks else 1
    sf = (n_chunks * avg_distance) / max(total_doc_chunks, 1)
    return sf


def scatter_coverage_at_k(retrieved_chunk_ids: list[str],
                          chunks: dict[str, ChunkMeta],
                          n_deciles: int = 10) -> float:
    """Scatter Coverage@K: fraction of document deciles represented in retrieved chunks.

    Divides document into n_deciles bins, counts how many are covered.
    """
    if not retrieved_chunk_ids:
        return 0.0

    covered_deciles = set()
    for cid in retrieved_chunk_ids:
        chunk = chunks.get(cid)
        if chunk:
            decile = min(int(chunk.position_fraction * n_deciles), n_deciles - 1)
            covered_deciles.add(decile)

    return len(covered_deciles) / n_deciles


def retrieval_completeness_at_k(retrieved_chunk_ids: list[str],
                                gold: GoldEvidence) -> float:
    """Retrieval Completeness@K: fraction of gold-relevant chunks retrieved.

    RC@K = |retrieved ∩ gold| / |gold|
    """
    if not gold.gold_chunk_ids:
        return 0.0

    retrieved_set = set(retrieved_chunk_ids)
    gold_set = set(gold.gold_chunk_ids)
    return len(retrieved_set & gold_set) / len(gold_set)


def information_completeness_score(present_attributes: list[str],
                                   gold_attributes: dict[str, str]) -> float:
    """Information Completeness Score (ICS).

    ICS = count(present_attributes) / count(gold_attributes)

    present_attributes: list of attribute names confirmed present in generated answer.
    gold_attributes: dict of {attribute_name: gold_text_evidence} from gold inventory.
    """
    if not gold_attributes:
        return 0.0

    present_set = set(present_attributes)
    gold_names = set(gold_attributes.keys())
    return len(present_set & gold_names) / len(gold_names)


def compute_all_metrics(retrieved_chunk_ids: list[str],
                        present_attributes: list[str],
                        gold: GoldEvidence,
                        chunks: dict[str, ChunkMeta]) -> dict[str, float]:
    """Compute all novel metrics for one query."""
    return {
        "scatter_coverage_at_k": scatter_coverage_at_k(retrieved_chunk_ids, chunks),
        "retrieval_completeness_at_k": retrieval_completeness_at_k(retrieved_chunk_ids, gold),
        "ics": information_completeness_score(present_attributes, gold.gold_attributes),
    }
