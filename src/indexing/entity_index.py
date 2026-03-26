"""Entity-to-chunk index built from NER + coreference resolution.

This is Component 1 of the scatter-aware RAG system.
For each entity in a document, maps to all chunk IDs where that entity appears.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import spacy

from src.indexing.chunker import ChunkMeta
from src.utils.config import data_dir

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    """Load spaCy model (cached)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


def _normalize_entity_name(name: str) -> str:
    """Normalize entity name for matching."""
    return name.strip().lower()


class EntityIndex:
    """Maps entities to the chunks where they appear.

    Structure:
        entity_name -> {
            "canonical": str,           # Canonical name form
            "chunk_ids": [str],         # All chunks mentioning this entity
            "mention_count": int,
            "entity_type": str,         # PERSON, ORG, etc.
            "aliases": [str],           # Variant names / coreference aliases
        }
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self._entities: dict[str, dict[str, Any]] = {}
        self._chunk_to_entities: dict[str, list[str]] = defaultdict(list)

    def build_from_chunks(self, chunks: list[ChunkMeta],
                          entity_types: tuple[str, ...] = ("PERSON", "ORG", "GPE", "FAC", "WORK_OF_ART")):
        """Run NER on all chunks and build entity-to-chunk mapping."""
        nlp = _get_nlp()

        for chunk in chunks:
            doc = nlp(chunk.text)
            for ent in doc.ents:
                if ent.label_ not in entity_types:
                    continue
                key = _normalize_entity_name(ent.text)
                if key not in self._entities:
                    self._entities[key] = {
                        "canonical": ent.text,
                        "chunk_ids": [],
                        "mention_count": 0,
                        "entity_type": ent.label_,
                        "aliases": [],
                    }
                if chunk.chunk_id not in self._entities[key]["chunk_ids"]:
                    self._entities[key]["chunk_ids"].append(chunk.chunk_id)
                self._entities[key]["mention_count"] += 1
                self._chunk_to_entities[chunk.chunk_id].append(key)

        logger.info("Built entity index for '%s': %d entities across %d chunks",
                     self.doc_id, len(self._entities), len(chunks))

    def merge_coreferences(self, coref_clusters: dict[str, list[str]]):
        """Merge coreference clusters into the entity index.

        coref_clusters: {canonical_name: [alias1, alias2, ...]}
        """
        for canonical, aliases in coref_clusters.items():
            canon_key = _normalize_entity_name(canonical)
            if canon_key not in self._entities:
                continue
            for alias in aliases:
                alias_key = _normalize_entity_name(alias)
                if alias_key in self._entities and alias_key != canon_key:
                    # Merge alias's chunks into canonical
                    for cid in self._entities[alias_key]["chunk_ids"]:
                        if cid not in self._entities[canon_key]["chunk_ids"]:
                            self._entities[canon_key]["chunk_ids"].append(cid)
                    self._entities[canon_key]["mention_count"] += self._entities[alias_key]["mention_count"]
                    self._entities[canon_key]["aliases"].append(alias)
                    del self._entities[alias_key]

    def get_chunks_for_entity(self, entity_name: str) -> list[str]:
        """Get all chunk IDs associated with an entity."""
        key = _normalize_entity_name(entity_name)
        if key in self._entities:
            return self._entities[key]["chunk_ids"]
        # Fuzzy match: check if the query is a substring of any entity
        for ek, ev in self._entities.items():
            if key in ek or ek in key:
                return ev["chunk_ids"]
        return []

    def get_entities_in_chunk(self, chunk_id: str) -> list[str]:
        """Get all entity names mentioned in a chunk."""
        return self._chunk_to_entities.get(chunk_id, [])

    def get_top_entities(self, n: int = 10) -> list[dict[str, Any]]:
        """Get top-N entities by mention count."""
        sorted_ents = sorted(self._entities.values(),
                             key=lambda e: e["mention_count"], reverse=True)
        return sorted_ents[:n]

    @property
    def entities(self) -> dict[str, dict[str, Any]]:
        return dict(self._entities)

    def save(self, name: str | None = None):
        """Save entity index to disk as JSON."""
        out_dir = data_dir("entity_indices")
        fname = name or self.doc_id
        path = out_dir / f"{fname}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": self.doc_id,
                "entities": self._entities,
                "chunk_to_entities": dict(self._chunk_to_entities),
            }, f, indent=2)
        logger.info("Saved entity index: %s", path)

    @classmethod
    def load(cls, name: str) -> "EntityIndex":
        """Load entity index from disk."""
        idx_dir = data_dir("entity_indices")
        path = idx_dir / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ei = cls(data["doc_id"])
        ei._entities = data["entities"]
        ei._chunk_to_entities = defaultdict(list, data.get("chunk_to_entities", {}))
        logger.info("Loaded entity index: %s (%d entities)", name, len(ei._entities))
        return ei
