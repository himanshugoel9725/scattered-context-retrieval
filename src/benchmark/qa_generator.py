"""Generate QA pairs for ScatterQA entities.

For each entity × 5 question types = questions.
Uses GPT-4o-mini to generate gold answers from annotated paragraphs.
"""

import hashlib
import logging
from typing import Any

from src.benchmark.entity_annotator import annotate_entity, ATTRIBUTE_LABELS
from src.evaluation.gold_schema import GoldEvidence, QueryMetadata
from src.generation.llm_client import generate
from src.generation.prompts import QA_GENERATION_PROMPT, SCATTER_CATEGORY_PROMPT
from src.indexing.chunker import ChunkMeta

logger = logging.getLogger(__name__)

# 5 question types that map to scatter patterns
QUESTION_TYPES = [
    {
        "type": "comprehensive_description",
        "scatter_category": "distributed_attributes",
        "template": "Provide a comprehensive description of {entity} including their appearance, background, personality, relationships, and development throughout the story.",
    },
    {
        "type": "character_evolution",
        "scatter_category": "progressive_accumulation",
        "template": "How does {entity} change or develop over the course of the narrative?",
    },
    {
        "type": "relationship_analysis",
        "scatter_category": "cross_reference",
        "template": "Describe all significant relationships that {entity} has with other characters and how these relationships evolve.",
    },
    {
        "type": "motivation_analysis",
        "scatter_category": "implicit",
        "template": "What motivates {entity}? Explain their goals, desires, and the reasoning behind their key decisions.",
    },
    {
        "type": "contradiction_check",
        "scatter_category": "contradictory_evolution",
        "template": "Are there any contradictions or inconsistencies in how {entity} is portrayed throughout the text? If so, describe them.",
    },
]


def generate_entity_questions(entity_data: dict[str, Any],
                              chunk_map: dict[str, ChunkMeta],
                              doc_id: str,
                              novel_info: dict[str, str],
                              n_questions: int = 5,
                              model: str = "gpt-4o-mini") -> list[GoldEvidence]:
    """Generate QA pairs for one entity."""
    entity_name = entity_data["canonical"]
    entity_chunks = entity_data["chunk_ids"]

    # Annotate chunks with attribute labels
    annotations = annotate_entity(entity_name, entity_chunks, chunk_map, model=model)

    # Build gold attribute inventory
    gold_attributes = {}
    for attr_name, chunk_ids in annotations.items():
        if chunk_ids:
            # Use first chunk's text as gold evidence for this attribute
            chunk = chunk_map.get(chunk_ids[0])
            if chunk:
                gold_attributes[attr_name] = chunk.text[:500]

    records = []
    for qtype in QUESTION_TYPES[:n_questions]:
        query = qtype["template"].format(entity=entity_name)

        # Generate gold answer from all entity chunks
        context_text = "\n\n---\n\n".join(
            chunk_map[cid].text for cid in entity_chunks[:15]
            if cid in chunk_map
        )

        prompt = QA_GENERATION_PROMPT.format(
            entity=entity_name,
            question_type=qtype["type"],
            passages=context_text,
        )
        gold_answer = generate(prompt, model=model, temperature=0.0, max_tokens=500)

        query_id = hashlib.sha256(f"{doc_id}:{entity_name}:{qtype['type']}".encode()).hexdigest()[:16]

        record = GoldEvidence(
            query_id=query_id,
            entity_id=f"{doc_id}:{entity_name}",
            gold_chunk_ids=entity_chunks,
            gold_attributes=gold_attributes,
            attribute_chunks=annotations,
            query_metadata=QueryMetadata(
                query_type="scattered",
                focus_type="entity",
                gold_entity=entity_name,
                scatter_category=qtype["scatter_category"],
                required_chunks_count=len(entity_chunks),
                domain="benchmark",
            ),
        )
        records.append(record)

    return records
