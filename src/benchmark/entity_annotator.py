"""Entity annotation for ScatterQA: NER + LLM semi-auto annotation.

For each entity: use entity index to find all paragraphs.
Use GPT-4o-mini to classify each paragraph's information type.
"""

import logging
from typing import Any

from src.generation.llm_client import generate
from src.generation.prompts import PARAGRAPH_ATTRIBUTE_CLASSIFIER
from src.indexing.chunker import ChunkMeta

logger = logging.getLogger(__name__)

ATTRIBUTE_LABELS = [
    "appearance",
    "background",
    "personality",
    "relationships",
    "arc",
]


def classify_paragraph_attribute(entity_name: str, paragraph_text: str,
                                 model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Classify what type of information a paragraph provides about an entity.

    Returns:
        {"label": str, "confidence": str, "evidence": str}
    """
    prompt = PARAGRAPH_ATTRIBUTE_CLASSIFIER.format(
        entity=entity_name,
        paragraph=paragraph_text,
    )
    response = generate(prompt, model=model, temperature=0.0, max_tokens=100)
    response = response.strip().lower()

    # Parse: first line should be the label
    label = "unknown"
    for attr in ATTRIBUTE_LABELS:
        if attr in response:
            label = attr
            break

    return {"label": label, "raw_response": response}


def annotate_entity(entity_name: str, chunk_ids: list[str],
                    chunk_map: dict[str, ChunkMeta],
                    model: str = "gpt-4o-mini") -> dict[str, list[str]]:
    """Annotate all chunks for an entity with attribute labels.

    Returns: {attribute_label: [chunk_id, ...]}
    """
    annotations: dict[str, list[str]] = {a: [] for a in ATTRIBUTE_LABELS}

    for cid in chunk_ids:
        chunk = chunk_map.get(cid)
        if chunk is None:
            continue
        result = classify_paragraph_attribute(entity_name, chunk.text, model=model)
        label = result["label"]
        if label in annotations:
            annotations[label].append(cid)

    return annotations
