#!/usr/bin/env python3
"""Re-annotate ScatterQA gold evidence with per-attribute chunk mappings.

For each unique entity in gold_evidence.jsonl:
  1. Re-chunk the source novel (deterministic — reproduces original chunk_ids)
  2. Call annotate_entity() to classify each chunk by attribute type
  3. Store the full {attribute: [chunk_ids]} mapping into attribute_chunks

LLM calls are disk-cached, so re-runs cost $0 if the ScatterQA build ran previously.
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.entity_annotator import annotate_entity, ATTRIBUTE_LABELS
from src.data.gutenberg import download_gutenberg_text
from src.data.processors import clean_text
from src.evaluation.gold_schema import load_gold_evidence, save_gold_evidence
from src.indexing.chunker import Chunker
from src.utils.config import data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    input_path = data_dir("scatterqa") / "gold_evidence.jsonl"
    output_path = data_dir("scatterqa") / "gold_evidence_annotated.jsonl"

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load all records
    records = load_gold_evidence(input_path)
    logger.info("Loaded %d records from %s", len(records), input_path)

    # Group by entity_id (all 5 question types share the same entity/chunks)
    entity_records: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(records):
        entity_records[r.entity_id].append(idx)

    logger.info("Found %d unique entities", len(entity_records))

    # Cache: book_id → chunk_map (avoid re-chunking the same novel)
    book_chunk_maps: dict[int, dict] = {}
    chunker = Chunker(chunk_size=512, overlap=128)

    # Stats
    total_entities = len(entity_records)
    stats_rows = []

    for ent_idx, (entity_id, record_indices) in enumerate(entity_records.items(), 1):
        # Parse entity_id: "gutenberg_{book_id}:{entity_name}"
        parts = entity_id.split(":", 1)
        if len(parts) != 2 or not parts[0].startswith("gutenberg_"):
            logger.warning("Skipping unrecognized entity_id format: %s", entity_id)
            continue

        doc_id = parts[0]
        entity_name = parts[1]
        book_id = int(doc_id.replace("gutenberg_", ""))

        logger.info("[%d/%d] Annotating entity: %s (book %d, %d records)",
                    ent_idx, total_entities, entity_name, book_id, len(record_indices))

        # Get or build chunk_map for this book
        if book_id not in book_chunk_maps:
            try:
                text = download_gutenberg_text(book_id)
                text = clean_text(text)
                chunks = chunker.chunk_document(doc_id, text)
                book_chunk_maps[book_id] = {c.chunk_id: c for c in chunks}
                logger.info("  Chunked book %d: %d chunks", book_id, len(chunks))
            except Exception as e:
                logger.error("  Failed to chunk book %d: %s", book_id, e)
                continue

        chunk_map = book_chunk_maps[book_id]

        # Get gold_chunk_ids from the first record (all records for same entity share them)
        gold_chunk_ids = records[record_indices[0]].gold_chunk_ids

        # Filter to chunks that exist in the chunk_map
        valid_chunk_ids = [cid for cid in gold_chunk_ids if cid in chunk_map]
        if len(valid_chunk_ids) < len(gold_chunk_ids):
            logger.warning("  %d/%d gold chunks not found in chunk_map (book %d)",
                           len(gold_chunk_ids) - len(valid_chunk_ids),
                           len(gold_chunk_ids), book_id)

        # Run annotation
        annotations = annotate_entity(entity_name, valid_chunk_ids, chunk_map)

        # Count unclassified chunks (in gold but not in any attribute bucket)
        classified_ids = set()
        for chunk_ids in annotations.values():
            classified_ids.update(chunk_ids)
        unclassified = [cid for cid in valid_chunk_ids if cid not in classified_ids]
        if unclassified:
            annotations["unknown"] = unclassified

        # Log per-entity stats
        attr_counts = {k: len(v) for k, v in annotations.items()}
        total_classified = sum(attr_counts.values())
        logger.info("  %s: %d total → %s",
                    entity_name, total_classified,
                    ", ".join(f"{k}={v}" for k, v in attr_counts.items()))
        stats_rows.append({
            "entity_id": entity_id,
            "total_gold_chunks": len(gold_chunk_ids),
            "valid_chunks": len(valid_chunk_ids),
            **attr_counts,
        })

        # Apply annotations to all records for this entity
        for ridx in record_indices:
            records[ridx].attribute_chunks = annotations

    # Save annotated output
    save_gold_evidence(records, output_path)
    logger.info("Saved %d annotated records to %s", len(records), output_path)

    # Re-split dev/test (20/80) — same split logic as scatterqa_builder
    n_dev = max(1, len(records) // 5)
    dev_records = records[:n_dev]
    test_records = records[n_dev:]
    save_gold_evidence(dev_records, data_dir("scatterqa") / "dev_annotated.jsonl")
    save_gold_evidence(test_records, data_dir("scatterqa") / "test_annotated.jsonl")
    logger.info("Split: %d dev, %d test", len(dev_records), len(test_records))

    # Verification checks
    logger.info("--- Verification ---")
    n_empty = sum(1 for r in records if not r.attribute_chunks)
    logger.info("Records with empty attribute_chunks: %d / %d", n_empty, len(records))

    missing_attrs = 0
    for r in records:
        for attr in ATTRIBUTE_LABELS:
            if attr not in r.attribute_chunks:
                missing_attrs += 1
                break
    logger.info("Records missing ≥1 standard attribute key: %d / %d", missing_attrs, len(records))

    subset_violations = 0
    for r in records:
        gold_set = set(r.gold_chunk_ids)
        for attr, cids in r.attribute_chunks.items():
            if not set(cids).issubset(gold_set):
                subset_violations += 1
                break
    logger.info("Records with attribute chunks NOT subset of gold: %d", subset_violations)

    # Summary table
    logger.info("--- Per-Entity Summary ---")
    logger.info("%-40s %6s %6s %6s %6s %6s %6s %6s",
                "Entity", "Total", "appear", "backgr", "person", "relat", "arc", "unknwn")
    for row in stats_rows:
        logger.info("%-40s %6d %6d %6d %6d %6d %6d %6d",
                    row["entity_id"][:40],
                    row["valid_chunks"],
                    row.get("appearance", 0),
                    row.get("background", 0),
                    row.get("personality", 0),
                    row.get("relationships", 0),
                    row.get("arc", 0),
                    row.get("unknown", 0))


if __name__ == "__main__":
    main()
