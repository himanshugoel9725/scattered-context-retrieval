"""Abstract base retriever interface and query entity detection."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import spacy

from src.indexing.chunker import ChunkMeta

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source: str  # e.g. "semantic", "entity", "bm25"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 15) -> list[RetrievalResult]:
        """Retrieve top-k chunks for a query."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging/reporting."""
        ...


def detect_query_entities(query: str,
                          entity_index=None,
                          entity_types: tuple[str, ...] = ("PERSON", "ORG", "GPE", "FAC", "WORK_OF_ART")
                          ) -> list[str]:
    """Detect named entities in a query using spaCy NER + fuzzy match against entity index.

    Returns list of detected entity names.
    """
    nlp = _get_nlp()
    doc = nlp(query)
    detected = []
    for ent in doc.ents:
        if ent.label_ in entity_types:
            detected.append(ent.text)

    # If entity index is provided, match against known entities using word boundaries
    if entity_index is not None:
        query_lower = query.lower()
        for ename, edata in entity_index.entities.items():
            pattern = re.compile(r'\b' + re.escape(ename) + r'\b', re.IGNORECASE)
            if pattern.search(query_lower):
                canonical = edata.get("canonical", ename)
                if canonical not in detected:
                    detected.append(canonical)
                continue
            for alias in edata.get("aliases", []):
                alias_pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                if alias_pattern.search(query_lower):
                    canonical = edata.get("canonical", ename)
                    if canonical not in detected:
                        detected.append(canonical)
                    break
    return detected
