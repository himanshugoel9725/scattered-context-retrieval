#!/usr/bin/env python3
"""Clean the ScatterQA benchmark.

Applies three transformations to gold_evidence.jsonl and
gold_evidence_annotated.jsonl:

  1. Remove Yellow Wallpaper (gutenberg_1952) — all 50 records
  2. Merge name-variant duplicates — union gold_chunk_ids from the variant
     into the canonical entity's records, then drop the variant's records
  3. Remove non-person entities — GPE/FAC/LOC entities that aren't characters

Outputs:
  data/scatterqa/gold_evidence_cleaned.jsonl
  data/scatterqa/gold_evidence_annotated_cleaned.jsonl

Usage:
  .venv/bin/python scripts/clean_scatterqa.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.gold_schema import load_gold_evidence, save_gold_evidence
from src.utils.config import data_dir

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Novel to remove entirely
REMOVE_NOVEL_PREFIXES = [
    "gutenberg_1952",  # The Yellow Wallpaper — too short, too many NER artifacts
]

# Non-person entities to remove (GPE / FAC / LOC — not characters)
REMOVE_ENTITY_IDS = {
    "gutenberg_1661:London",       # GPE — setting, not a character
    "gutenberg_1661:Baker Street", # FAC — address, not a character
    "gutenberg_84:Geneva",         # GPE — city in Frankenstein
}

# Name-variant pairs: (variant_entity_id, canonical_entity_id)
# variant's chunk_ids are merged into canonical; variant records are dropped.
NAME_VARIANT_PAIRS = [
    ("gutenberg_1342:Lizzy",           "gutenberg_1342:Elizabeth"),    # Pride and Prejudice
    ("gutenberg_1661:Sherlock Holmes", "gutenberg_1661:Holmes"),       # Sherlock Holmes
    ("gutenberg_174:Harry",            "gutenberg_174:Henry"),         # Dorian Gray (Lord Henry Wotton)
    ("gutenberg_2554:Rodya",           "gutenberg_2554:Raskolnikov"),  # Crime & Punishment
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def clean_records(records):
    """Apply all three cleaning passes to a list of GoldEvidence records.

    Returns (cleaned_records, stats_dict).
    """
    stats = {
        "initial": len(records),
        "removed_novel": 0,
        "removed_nonperson": 0,
        "merged_variants": 0,
        "removed_variants": 0,
        "final": 0,
    }

    # -----------------------------------------------------------------------
    # Pass 1: Remove Yellow Wallpaper (and any other removed-novel prefix)
    # -----------------------------------------------------------------------
    filtered = []
    for r in records:
        if any(r.entity_id.startswith(prefix + ":") for prefix in REMOVE_NOVEL_PREFIXES):
            stats["removed_novel"] += 1
        else:
            filtered.append(r)
    logger.info("Pass 1 (remove novels): %d → %d  (dropped %d)",
                stats["initial"], len(filtered), stats["removed_novel"])

    # -----------------------------------------------------------------------
    # Pass 2: Remove non-person entities
    # -----------------------------------------------------------------------
    kept = []
    for r in filtered:
        if r.entity_id in REMOVE_ENTITY_IDS:
            stats["removed_nonperson"] += 1
        else:
            kept.append(r)
    logger.info("Pass 2 (remove non-person entities): %d → %d  (dropped %d)",
                len(filtered), len(kept), stats["removed_nonperson"])

    # -----------------------------------------------------------------------
    # Pass 3: Merge name variants into canonical, then drop variants
    # -----------------------------------------------------------------------
    # Build a map from entity_id → list of records
    entity_map: dict[str, list] = {}
    for r in kept:
        entity_map.setdefault(r.entity_id, []).append(r)

    variant_entity_ids_to_drop = set()
    merges_done = 0

    for variant_id, canonical_id in NAME_VARIANT_PAIRS:
        if variant_id not in entity_map:
            logger.info("  Name-variant pair not found in data (skipping): %s → %s",
                        variant_id, canonical_id)
            continue
        if canonical_id not in entity_map:
            logger.warning("  Canonical entity missing for variant %s; skipping merge",
                           variant_id)
            continue

        # Collect the union of all chunk_ids mentioned by the variant
        variant_all_chunks: set[str] = set()
        variant_all_attr_chunks: dict[str, set[str]] = {}
        for vr in entity_map[variant_id]:
            variant_all_chunks.update(vr.gold_chunk_ids)
            for attr, cids in vr.attribute_chunks.items():
                variant_all_attr_chunks.setdefault(attr, set()).update(cids)

        # Merge into each of the canonical's records
        for cr in entity_map[canonical_id]:
            merged_chunks = sorted(
                set(cr.gold_chunk_ids) | variant_all_chunks,
                key=lambda cid: int(cid.rsplit("_c", 1)[1]) if "_c" in cid else 0,
            )
            cr.gold_chunk_ids = merged_chunks
            cr.query_metadata.required_chunks_count = len(merged_chunks)

            for attr, cids in variant_all_attr_chunks.items():
                if attr in cr.attribute_chunks:
                    cr.attribute_chunks[attr] = sorted(
                        set(cr.attribute_chunks[attr]) | cids,
                        key=lambda cid: int(cid.rsplit("_c", 1)[1]) if "_c" in cid else 0,
                    )
                # Don't add new attribute keys — canonical's gold_attributes is authoritative

        variant_entity_ids_to_drop.add(variant_id)
        merges_done += 1
        logger.info("  Merged %s → %s  (added %d unique chunks to canonical)",
                    variant_id, canonical_id, len(variant_all_chunks))

    stats["merged_variants"] = merges_done

    # Drop variant records
    final = []
    for r in kept:
        if r.entity_id in variant_entity_ids_to_drop:
            stats["removed_variants"] += 1
        else:
            final.append(r)

    logger.info("Pass 3 (merge/drop variants): %d → %d  (dropped %d variant records)",
                len(kept), len(final), stats["removed_variants"])

    stats["final"] = len(final)

    # -----------------------------------------------------------------------
    # Sanity: report remaining novel prefixes
    # -----------------------------------------------------------------------
    novels = {r.entity_id.split(":")[0] for r in final}
    entities = {r.entity_id for r in final}
    unique_entities = len({eid for eid in entities})
    logger.info("Remaining: %d records | %d unique entities | %d novels",
                len(final), unique_entities // 5,  # each entity has 5 records
                len(novels))
    logger.info("Novels: %s", sorted(novels))

    return final, stats


def main():
    scatterqa_dir = data_dir("scatterqa")

    input_pairs = [
        ("gold_evidence.jsonl",            "gold_evidence_cleaned.jsonl"),
        ("gold_evidence_annotated.jsonl",   "gold_evidence_annotated_cleaned.jsonl"),
    ]

    for src_name, dst_name in input_pairs:
        src_path = scatterqa_dir / src_name
        dst_path = scatterqa_dir / dst_name

        if not src_path.exists():
            logger.warning("Source file not found, skipping: %s", src_path)
            continue

        logger.info("=== Processing %s ===", src_name)
        records = load_gold_evidence(src_path)
        logger.info("Loaded %d records from %s", len(records), src_name)

        cleaned, stats = clean_records(records)
        save_gold_evidence(cleaned, dst_path)

        logger.info("Written %d records to %s", len(cleaned), dst_path)
        logger.info("Summary: %d initial → %d final  "
                    "(−%d novel, −%d non-person, merged %d variant pairs [−%d records])",
                    stats["initial"], stats["final"],
                    stats["removed_novel"], stats["removed_nonperson"],
                    stats["merged_variants"], stats["removed_variants"])
        print()

    # Final quick check on the primary cleaned file
    primary = scatterqa_dir / "gold_evidence_cleaned.jsonl"
    if primary.exists():
        from src.evaluation.gold_schema import load_gold_evidence as lge
        check = lge(primary)
        novels = sorted({r.entity_id.split(":")[0] for r in check})
        entities = sorted({r.entity_id for r in check})
        print(f"\n✓ gold_evidence_cleaned.jsonl: {len(check)} records")
        print(f"  {len(entities)} unique entity_ids across {len(novels)} novels")
        print(f"  Novels: {novels}")
        # Verify none of the bad entity_ids survived
        bad = [r.entity_id for r in check
               if r.entity_id in REMOVE_ENTITY_IDS
               or any(r.entity_id.startswith(p + ":") for p in REMOVE_NOVEL_PREFIXES)
               or r.entity_id in {v for v, _ in NAME_VARIANT_PAIRS}]
        if bad:
            print(f"  ⚠ STILL PRESENT (should have been removed): {bad}")
        else:
            print(f"  ✓ No blacklisted entity_ids remain")


if __name__ == "__main__":
    main()
