"""Strategy B: Entity-Expanded retrieval.

Retrieve top-K by semantics → identify entities in retrieved chunks →
pull additional chunks from entity index for those entities.
"""

from src.indexing.chunker import ChunkMeta
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.base import BaseRetriever, RetrievalResult, detect_query_entities


class EntityExpandedRetriever(BaseRetriever):
    """Semantic retrieval with entity-based expansion."""

    def __init__(self, vector_index: VectorIndex, entity_index: EntityIndex,
                 chunks: dict[str, ChunkMeta], expansion_factor: float = 0.5):
        self.vector_index = vector_index
        self.entity_index = entity_index
        self.chunks = chunks
        self.expansion_factor = expansion_factor  # fraction of K for entity expansion

    def retrieve(self, query: str, k: int = 15) -> list[RetrievalResult]:
        # Step 1: Semantic retrieval
        semantic_k = k
        hits = self.vector_index.search(query, k=semantic_k)
        results = []
        seen = set()

        for chunk_id, score in hits:
            seen.add(chunk_id)
            chunk = self.chunks.get(chunk_id)
            results.append(RetrievalResult(
                chunk_id=chunk_id, text=chunk.text if chunk else "",
                score=score, source="semantic",
            ))

        # Step 2: Identify entities in retrieved chunks
        entities_found = set()
        for r in results:
            ents = self.entity_index.get_entities_in_chunk(r.chunk_id)
            entities_found.update(ents)

        # Also check query for entities
        query_ents = detect_query_entities(query, entity_index=self.entity_index)
        entities_found.update(e.lower() for e in query_ents)

        # Step 3: Pull additional entity chunks
        expansion_budget = max(1, int(k * self.expansion_factor))
        entity_chunks = []
        for entity_name in entities_found:
            for cid in self.entity_index.get_chunks_for_entity(entity_name):
                if cid not in seen:
                    chunk = self.chunks.get(cid)
                    if chunk:
                        entity_chunks.append((cid, chunk))
                        seen.add(cid)

        # Rerank entity chunks by semantic similarity to query
        if entity_chunks:
            from src.indexing.vector_index import embed_texts
            q_emb = embed_texts([query], show_progress=False)
            texts = [c.text for _, c in entity_chunks]
            c_embs = embed_texts(texts, show_progress=False)
            import numpy as np
            scores = np.dot(c_embs, q_emb.T).flatten()
            ranked = sorted(zip(entity_chunks, scores), key=lambda x: x[1], reverse=True)
            for (cid, chunk), score in ranked[:expansion_budget]:
                results.append(RetrievalResult(
                    chunk_id=cid, text=chunk.text,
                    score=float(score), source="entity_expansion",
                ))

        return results[:k]

    def name(self) -> str:
        return "EntityExpanded"
