"""Strategy E: Hybrid Entity retrieval (main proposed method).

Score = alpha * semantic + beta * entity_association + gamma * positional_diversity

This is Component 2 of the scatter-aware RAG system.
"""

import numpy as np

from src.indexing.chunker import ChunkMeta
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex, embed_texts
from src.retrieval.base import BaseRetriever, RetrievalResult, detect_query_entities


class HybridEntityRetriever(BaseRetriever):
    """Hybrid scoring: semantic + entity association + positional diversity."""

    def __init__(self, vector_index: VectorIndex, entity_index: EntityIndex,
                 chunks: dict[str, ChunkMeta],
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2,
                 candidate_pool_factor: int = 3):
        self.vector_index = vector_index
        self.entity_index = entity_index
        self.chunks = chunks
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.candidate_pool_factor = candidate_pool_factor

    def retrieve(self, query: str, k: int = 15) -> list[RetrievalResult]:
        # Step 1: Build candidate pool from semantic search + entity chunks
        pool_size = k * self.candidate_pool_factor
        semantic_hits = self.vector_index.search(query, k=pool_size)

        # Detect query entities
        query_entities = detect_query_entities(query, entity_index=self.entity_index)
        query_entity_keys = {e.lower() for e in query_entities}

        # Also get entity chunks
        entity_chunk_ids = set()
        for entity_name in query_entity_keys:
            for cid in self.entity_index.get_chunks_for_entity(entity_name):
                entity_chunk_ids.add(cid)

        # Merge into candidate set
        candidates: dict[str, dict] = {}
        for chunk_id, score in semantic_hits:
            candidates[chunk_id] = {"semantic_score": score}
        for cid in entity_chunk_ids:
            if cid not in candidates:
                candidates[cid] = {"semantic_score": 0.0}

        if not candidates:
            return []

        # Step 2: Compute semantic scores for any candidates missing one
        missing_semantic = [cid for cid, v in candidates.items() if v["semantic_score"] == 0.0]
        if missing_semantic:
            q_emb = embed_texts([query], show_progress=False)
            texts = [self.chunks[cid].text for cid in missing_semantic if cid in self.chunks]
            if texts:
                c_embs = embed_texts(texts, show_progress=False)
                scores = np.dot(c_embs, q_emb.T).flatten()
                for cid, score in zip(missing_semantic, scores):
                    candidates[cid]["semantic_score"] = float(score)

        # Step 3: Compute entity association score
        for cid in candidates:
            ents_in_chunk = set(self.entity_index.get_entities_in_chunk(cid))
            if query_entity_keys:
                overlap = len(ents_in_chunk & query_entity_keys)
                candidates[cid]["entity_score"] = overlap / len(query_entity_keys)
            else:
                candidates[cid]["entity_score"] = 0.0

        # Step 4: Greedy selection with positional diversity
        selected = self._greedy_diverse_select(candidates, k)
        return selected

    def _greedy_diverse_select(self, candidates: dict[str, dict],
                               k: int) -> list[RetrievalResult]:
        """Greedy selection maximizing hybrid score with positional diversity bonus."""
        selected: list[RetrievalResult] = []
        covered_deciles: set[int] = set()
        remaining = set(candidates.keys())

        for _ in range(min(k, len(candidates))):
            best_cid = None
            best_score = -1.0

            for cid in remaining:
                chunk = self.chunks.get(cid)
                if chunk is None:
                    continue

                sem = candidates[cid]["semantic_score"]
                ent = candidates[cid]["entity_score"]

                # Positional diversity: bonus for covering new deciles
                decile = chunk.decile
                div_bonus = 1.0 if decile not in covered_deciles else 0.0

                combined = self.alpha * sem + self.beta * ent + self.gamma * div_bonus

                if combined > best_score:
                    best_score = combined
                    best_cid = cid

            if best_cid is None:
                break

            chunk = self.chunks.get(best_cid)
            covered_deciles.add(chunk.decile if chunk else 0)
            remaining.discard(best_cid)
            selected.append(RetrievalResult(
                chunk_id=best_cid,
                text=chunk.text if chunk else "",
                score=best_score,
                source="hybrid_entity",
                metadata={
                    "semantic": candidates[best_cid]["semantic_score"],
                    "entity": candidates[best_cid]["entity_score"],
                    "decile": chunk.decile if chunk else 0,
                },
            ))

        return selected

    def name(self) -> str:
        return f"HybridEntity(a={self.alpha},b={self.beta},g={self.gamma})"
