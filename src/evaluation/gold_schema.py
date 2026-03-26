"""Gold evidence and query metadata schema.

Canonical data structures for gold annotations (Fix #3, #4).
"""

from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path


@dataclass
class QueryMetadata:
    """Query classification metadata (Fix #7)."""
    query_type: str                     # "localized" | "scattered"
    focus_type: str                     # "entity" | "event" | "relation"
    gold_entity: str | None = None
    scatter_category: str | None = None  # progressive_accumulation | distributed_attributes | contradictory_evolution | cross_reference | implicit
    required_chunks_count: int = 1
    domain: str = "narrative"           # narrative | legal | scientific | benchmark

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "focus_type": self.focus_type,
            "gold_entity": self.gold_entity,
            "scatter_category": self.scatter_category,
            "required_chunks_count": self.required_chunks_count,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueryMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GoldEvidence:
    """Gold evidence annotation for a query-entity pair."""
    query_id: str
    entity_id: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_attributes: dict[str, str] = field(default_factory=dict)  # {attribute_name: gold_text_evidence}
    query_metadata: QueryMetadata = field(default_factory=lambda: QueryMetadata("scattered", "entity"))

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "entity_id": self.entity_id,
            "gold_chunk_ids": self.gold_chunk_ids,
            "gold_attributes": self.gold_attributes,
            "query_metadata": self.query_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoldEvidence":
        meta = QueryMetadata.from_dict(d.get("query_metadata", {}))
        return cls(
            query_id=d["query_id"],
            entity_id=d["entity_id"],
            gold_chunk_ids=d.get("gold_chunk_ids", []),
            gold_attributes=d.get("gold_attributes", {}),
            query_metadata=meta,
        )


def load_gold_evidence(path: str | Path) -> list[GoldEvidence]:
    """Load gold evidence from JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(GoldEvidence.from_dict(json.loads(line)))
    return records


def save_gold_evidence(records: list[GoldEvidence], path: str | Path):
    """Save gold evidence to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
