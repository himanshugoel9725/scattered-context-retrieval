#!/usr/bin/env python3
"""Run Information Completeness Score (ICS) evaluation on exp2_1 ScatterQA results.

ICS measures what fraction of gold attributes are present in a generated answer,
using an LLM judge per attribute.  This complements Attribute Recall (retrieval-side)
by measuring the generation side.

Inputs:
  results/exp2_1/exp2_1_results.json   — exp2_1 strategy comparison results
  data/scatterqa/gold_evidence_cleaned.jsonl  — cleaned gold evidence

Outputs:
  results/exp2_1/exp2_1_ics_scores.json  — per-record ICS per strategy
  (prints per-strategy mean ICS to stdout)

Estimated cost: ~$0.33 at gpt-4.1-nano rates
  (5 attributes × 515 records × 6 strategies × ~$0.00010/call)

Usage:
  .venv/bin/python scripts/run_ics_eval.py [--dry-run]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.gold_alignment import score_ics
from src.evaluation.gold_schema import load_gold_evidence
from src.utils.config import data_dir, results_dir

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES = ["BM25", "StandardSemantic", "EntityExpanded",
              "EntityFirst", "Iterative", "HybridEntity"]
MODEL = "gpt-4.1-nano"


def build_gold_index(gold_path: Path) -> dict[str, object]:
    """Build entity_id → GoldEvidence mapping from JSONL."""
    records = load_gold_evidence(gold_path)
    index: dict[str, object] = {}
    for r in records:
        # entity_id is not unique (5 records per entity, one per scatter_category)
        # Use query_id as the unique key when available, and build entity_id lookup too
        index[r.query_id] = r
    logger.info("Gold index built: %d records (keyed by query_id)", len(index))
    return index


def build_entity_gold_index(gold_path: Path) -> dict[str, list]:
    """Build entity_id → [GoldEvidence] mapping (5 records per entity)."""
    records = load_gold_evidence(gold_path)
    index: dict[str, list] = {}
    for r in records:
        index.setdefault(r.entity_id, []).append(r)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print cost estimate without making LLM calls")
    args = parser.parse_args()

    # Paths
    exp2_1_path = results_dir("exp2_1") / "exp2_1_results.json"
    if not exp2_1_path.exists():
        logger.error("Exp 2.1 results not found: %s", exp2_1_path)
        sys.exit(1)

    gold_dir = data_dir("scatterqa")
    gold_path = gold_dir / "gold_evidence_cleaned.jsonl"
    if not gold_path.exists():
        gold_path = gold_dir / "gold_evidence.jsonl"
    logger.info("Using gold evidence: %s", gold_path)

    # Load data
    with open(exp2_1_path) as f:
        exp2_data = json.load(f)

    scatterqa_entries = exp2_data.get("scatterqa", [])
    if not scatterqa_entries:
        logger.error("No 'scatterqa' entries in exp2_1 results")
        sys.exit(1)
    logger.info("Loaded %d ScatterQA entries from exp2_1", len(scatterqa_entries))

    # Build gold index keyed by entity_id
    entity_gold_index = build_entity_gold_index(gold_path)

    # Cost estimate
    n_records = len(scatterqa_entries)
    n_strategies = len(STRATEGIES)
    n_attributes = 5  # standard ScatterQA attribute set
    est_calls = n_records * n_strategies * n_attributes
    est_cost = est_calls * 0.00015
    logger.info("Estimated LLM calls: %d  (~$%.2f)", est_calls, est_cost)

    if args.dry_run:
        logger.info("Dry run — exiting without making LLM calls")
        return

    # Match exp2_1 entries to gold evidence by entity_id (from metadata)
    # exp2_1 ScatterQA entries carry metadata.entity_id in the QA pair
    # However, after the refactor the entry dict has: query, reference, doc_id, dataset,
    # detected_entity, {strategy}_answer, etc.
    # gold_attributes live in GoldEvidence; we match by doc_id + detected_entity.

    out_path = results_dir("exp2_1") / "exp2_1_ics_scores.json"
    # Resume support: load existing scores
    if out_path.exists():
        with open(out_path) as f:
            ics_scores = json.load(f)
        logger.info("Resuming: %d records already scored", len(ics_scores))
    else:
        ics_scores = {}

    per_strategy_ics: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    total = len(scatterqa_entries)

    for idx, entry in enumerate(scatterqa_entries):
        record_key = f"{entry.get('doc_id', '')}::{entry.get('query', '')[:60]}"

        if record_key in ics_scores:
            # Re-use cached
            for strat in STRATEGIES:
                cached = ics_scores[record_key].get(strat, {})
                if "ics" in cached:
                    per_strategy_ics[strat].append(cached["ics"])
            continue

        # Find a matching GoldEvidence record
        doc_id = entry.get("doc_id", "")
        detected_entity = entry.get("detected_entity")
        entity_id = f"{doc_id}:{detected_entity}" if detected_entity else None

        gold_list = entity_gold_index.get(entity_id, []) if entity_id else []
        if not gold_list:
            # Try all entities from this doc as fallback
            gold_list = [r for eid, recs in entity_gold_index.items()
                         if eid.startswith(doc_id + ":")
                         for r in recs]

        if not gold_list:
            logger.debug("No gold record for entry %d (doc_id=%s, entity=%s)",
                         idx, doc_id, detected_entity)
            continue

        # Use the first matching gold record (they share the same gold_attributes)
        gold = gold_list[0]

        entry_scores: dict[str, dict] = {}
        for strat in STRATEGIES:
            answer = entry.get(f"{strat}_answer", "")
            if not answer:
                continue
            try:
                result = score_ics(answer, gold, model=MODEL)
                entry_scores[strat] = result
                per_strategy_ics[strat].append(result["ics"])
            except Exception as e:
                logger.warning("ICS failed for strat=%s idx=%d: %s", strat, idx, e)

        ics_scores[record_key] = entry_scores

        if (idx + 1) % 20 == 0 or idx < 3:
            logger.info("Progress: %d / %d records scored", idx + 1, total)

        # Save incrementally every 50 records
        if (idx + 1) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(ics_scores, f, indent=2)
            logger.info("Checkpoint saved (%d records)", len(ics_scores))

    # Final save
    with open(out_path, "w") as f:
        json.dump(ics_scores, f, indent=2)
    logger.info("ICS scores saved to %s", out_path)

    # Print summary
    print("\n=== ICS Summary (ScatterQA subset of exp2_1) ===")
    print(f"{'Strategy':<20}  {'Mean ICS':>9}  {'N':>5}")
    print("-" * 40)
    for strat in STRATEGIES:
        scores = per_strategy_ics[strat]
        if scores:
            print(f"{strat:<20}  {np.mean(scores):>9.4f}  {len(scores):>5}")
        else:
            print(f"{strat:<20}  {'N/A':>9}  {'0':>5}")


if __name__ == "__main__":
    main()
