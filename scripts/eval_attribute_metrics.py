#!/usr/bin/env python3
"""Evaluate attribute-level retrieval metrics on annotated ScatterQA.

Two phases:
  Phase 1 (fast, no LLM): Attribute Recall/Precision/F1@15, Scatter Coverage@15
  Phase 2 (LLM-cached):   ROUGE-L via cached synthesis

Run: .venv/bin/python scripts/eval_attribute_metrics.py [--skip-rouge]
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.entity_annotator import ATTRIBUTE_LABELS
from src.data.gutenberg import download_gutenberg_text
from src.data.processors import clean_text
from src.evaluation.gold_schema import load_gold_evidence
from src.evaluation.metrics import scatter_coverage_at_k
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.entity_expanded import EntityExpandedRetriever
from src.retrieval.entity_first import EntityFirstRetriever
from src.retrieval.hybrid_entity import HybridEntityRetriever
from src.retrieval.iterative import IterativeRetriever
from src.retrieval.standard_semantic import StandardSemanticRetriever
from src.utils.config import data_dir, results_dir

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Exact templates from src/data/loaders.py _SCATTER_QUESTION_TEMPLATES
QUESTION_TEMPLATES = {
    "distributed_attributes": "Provide a comprehensive description of {entity} including their appearance, background, personality, relationships, and development throughout the story.",
    "progressive_accumulation": "How does {entity} change or develop over the course of the narrative?",
    "cross_reference": "Describe all significant relationships that {entity} has with other characters and how these relationships evolve.",
    "implicit": "What motivates {entity}? Explain their goals, desires, and the reasoning behind their key decisions.",
    "contradictory_evolution": "Are there any contradictions or inconsistencies in how {entity} is portrayed throughout the text? If so, describe them.",
}

K = 15
STRATEGY_NAMES = ["BM25", "StandardSemantic", "EntityExpanded",
                  "EntityFirst", "Iterative", "HybridEntity"]


def compute_attribute_metrics(retrieved_ids: list[str],
                              attribute_chunks: dict[str, list[str]]) -> dict:
    """Compute attribute-level recall, precision, F1."""
    retrieved_set = set(retrieved_ids[:K])

    # Macro recall over 5 standard attributes
    recalls = []
    for attr in ATTRIBUTE_LABELS:
        gold_attr = attribute_chunks.get(attr, [])
        if not gold_attr:
            continue  # skip attributes with 0 gold chunks
        recall_a = len(retrieved_set & set(gold_attr)) / len(gold_attr)
        recalls.append(recall_a)

    attr_recall = float(np.mean(recalls)) if recalls else 0.0

    # Precision: fraction of retrieved that are in ANY gold bucket (including unknown)
    all_gold = set()
    for cids in attribute_chunks.values():
        all_gold.update(cids)
    n_retrieved = len(retrieved_set)
    attr_precision = len(retrieved_set & all_gold) / n_retrieved if n_retrieved > 0 else 0.0

    # F1
    if attr_recall + attr_precision > 0:
        attr_f1 = 2 * attr_recall * attr_precision / (attr_recall + attr_precision)
    else:
        attr_f1 = 0.0

    return {
        "attr_recall": attr_recall,
        "attr_precision": attr_precision,
        "attr_f1": attr_f1,
    }


def build_book_indices(records, chunker):
    """Pre-build all book indices once upfront. Returns book_cache dict."""
    book_ids_needed = {}
    for rec in records:
        parts = rec.entity_id.split(":", 1)
        doc_id = parts[0]
        book_id = int(doc_id.replace("gutenberg_", ""))
        book_ids_needed[book_id] = doc_id

    book_cache = {}
    for book_id, doc_id in book_ids_needed.items():
        logger.info("Building indices for book %d...", book_id)
        text = download_gutenberg_text(book_id)
        text = clean_text(text)
        chunks = chunker.chunk_document(doc_id, text)
        chunk_map = {c.chunk_id: c for c in chunks}
        vector_idx = VectorIndex()
        vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
        entity_idx = EntityIndex(doc_id)
        entity_idx.build_from_chunks(chunks)
        book_cache[book_id] = (chunks, chunk_map, vector_idx, entity_idx)
        logger.info("  Book %d: %d chunks, indices ready", book_id, len(chunks))

    return book_cache


def run_retrieval_metrics(records, book_cache):
    """Phase 1: compute retrieval-only metrics (no LLM calls)."""
    all_results = []
    t_start = time.time()

    for i, rec in enumerate(records):
        parts = rec.entity_id.split(":", 1)
        doc_id = parts[0]
        entity_name = parts[1]
        book_id = int(doc_id.replace("gutenberg_", ""))

        chunks_list, chunk_map, vector_idx, entity_idx = book_cache[book_id]

        cat = rec.query_metadata.scatter_category
        entity = rec.query_metadata.gold_entity
        query = QUESTION_TEMPLATES.get(cat, "Describe {entity} in detail.").format(entity=entity)

        entry = {
            "query_id": rec.query_id,
            "entity_id": rec.entity_id,
            "scatter_category": cat,
            "entity_name": entity_name,
        }

        for sname in STRATEGY_NAMES:
            if sname == "BM25":
                retriever = BM25Retriever(chunks_list)
            elif sname == "StandardSemantic":
                retriever = StandardSemanticRetriever(vector_idx, chunk_map)
            elif sname == "EntityExpanded":
                retriever = EntityExpandedRetriever(vector_idx, entity_idx, chunk_map)
            elif sname == "EntityFirst":
                retriever = EntityFirstRetriever(vector_idx, entity_idx, chunk_map)
            elif sname == "Iterative":
                retriever = IterativeRetriever(vector_idx, entity_idx, chunk_map)
            elif sname == "HybridEntity":
                retriever = HybridEntityRetriever(vector_idx, entity_idx, chunk_map)

            retrieved = retriever.retrieve(query, k=K)
            chunk_ids = [r.chunk_id for r in retrieved]

            am = compute_attribute_metrics(chunk_ids, rec.attribute_chunks)
            sc = scatter_coverage_at_k(chunk_ids, chunk_map)

            entry[f"{sname}_attr_recall"] = am["attr_recall"]
            entry[f"{sname}_attr_precision"] = am["attr_precision"]
            entry[f"{sname}_attr_f1"] = am["attr_f1"]
            entry[f"{sname}_scatter_coverage"] = sc
            entry[f"{sname}_chunk_ids"] = chunk_ids

        all_results.append(entry)
        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t_start
            logger.info("[%d/%d] retrieval done (%.1fs elapsed)", i + 1, len(records), elapsed)

    return all_results


def run_rouge_phase(all_results, records, book_cache, checkpoint_path: Path | None = None):
    """Phase 2: compute ROUGE-L via cached synthesis."""
    from rouge_score import rouge_scorer
    from src.generation.synthesizer import synthesize, to_synthesis_chunks

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    t_start = time.time()

    # Resume: find first record that hasn't been ROUGE-scored yet
    start_idx = 0
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            cached = json.load(f)
        # Merge cached scores back into all_results
        cached_by_qid = {r["query_id"]: r for r in cached}
        for entry, rec in zip(all_results, records):
            if rec.query_id in cached_by_qid:
                for sname in STRATEGY_NAMES:
                    key = f"{sname}_rougeL"
                    if key in cached_by_qid[rec.query_id]:
                        entry[key] = cached_by_qid[rec.query_id][key]
        # Find first unscored entry
        for i, entry in enumerate(all_results):
            if f"BM25_rougeL" not in entry:
                start_idx = i
                break
        else:
            start_idx = len(all_results)  # fully done
        logger.info("Resuming Phase 2 from record %d / %d", start_idx, len(records))

    for i, (entry, rec) in enumerate(zip(all_results, records)):
        if i < start_idx:
            continue
        parts = rec.entity_id.split(":", 1)
        doc_id = parts[0]
        book_id = int(doc_id.replace("gutenberg_", ""))
        chunks_list, chunk_map, vector_idx, entity_idx = book_cache[book_id]

        cat = rec.query_metadata.scatter_category
        entity = rec.query_metadata.gold_entity
        query = QUESTION_TEMPLATES.get(cat, "Describe {entity} in detail.").format(entity=entity)

        answer_parts = [f"{attr}: {evidence}" for attr, evidence in rec.gold_attributes.items()]
        reference = " | ".join(answer_parts) if answer_parts else ""

        for sname in STRATEGY_NAMES:
            chunk_ids = entry[f"{sname}_chunk_ids"]
            # Rebuild minimal retrieval results for synthesis
            from src.retrieval.base import RetrievalResult
            fake_results = []
            for cid in chunk_ids:
                meta = chunk_map.get(cid)
                fake_results.append(RetrievalResult(
                    chunk_id=cid,
                    text=meta.text if meta else "",
                    score=0.0,
                    source=sname,
                ))

            syn_chunks = to_synthesis_chunks(fake_results[:K], chunk_map, doc_id)
            result = synthesize(query, syn_chunks, entity=entity, model="gpt-4.1-nano")
            answer = result["answer"]

            scores = scorer.score(reference, answer)
            entry[f"{sname}_rougeL"] = scores["rougeL"].fmeasure

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t_start
            logger.info("[%d/%d] ROUGE done (%.1fs elapsed)", i + 1, len(records), elapsed)
            # Checkpoint: save current results so we can resume on crash
            if checkpoint_path:
                save_results_cp = []
                for r in all_results:
                    if f"BM25_rougeL" in r:  # only save ROUGE-completed records
                        entry_cp = {k: v for k, v in r.items() if not k.endswith("_chunk_ids")}
                        save_results_cp.append(entry_cp)
                with open(checkpoint_path, "w") as f:
                    json.dump(save_results_cp, f)

    return all_results


def print_results(all_results, include_rouge=True):
    """Print aggregate results table."""
    metrics = ["attr_recall", "attr_precision", "attr_f1", "scatter_coverage"]
    headers = ["AttrRec@15", "AttrPre@15", "AttrF1@15", "ScatCov@15"]
    if include_rouge:
        metrics.append("rougeL")
        headers.append("ROUGE-L")

    logger.info("")
    logger.info("=" * 90)
    logger.info("RESULTS: Mean over %d records (K=%d)", len(all_results), K)
    logger.info("=" * 90)
    header = f"{'Strategy':<20} " + " ".join(f"{h:>10}" for h in headers)
    logger.info(header)
    logger.info("-" * 90)

    summary = {}
    for sname in STRATEGY_NAMES:
        summary[sname] = {}
        vals_str = []
        for metric in metrics:
            key = f"{sname}_{metric}"
            values = [r[key] for r in all_results if key in r]
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                summary[sname][metric] = mean_val
                summary[sname][f"{metric}_std"] = std_val
                vals_str.append(f"{mean_val:>10.4f}")
            else:
                vals_str.append(f"{'N/A':>10}")
        logger.info(f"{sname:<20} " + " ".join(vals_str))

    logger.info("=" * 90)

    # Per scatter_category breakdown
    categories = sorted(set(r["scatter_category"] for r in all_results))
    logger.info("")
    logger.info("Per scatter_category breakdown (Attribute F1@15):")
    logger.info(f"{'Category':<30} " + " ".join(f"{s:<18}" for s in STRATEGY_NAMES))
    for cat in categories:
        cat_records = [r for r in all_results if r["scatter_category"] == cat]
        vals = []
        for sname in STRATEGY_NAMES:
            f1s = [r[f"{sname}_attr_f1"] for r in cat_records]
            vals.append(f"{np.mean(f1s):.4f}")
        logger.info(f"{cat:<30} " + " ".join(f"{v:<18}" for v in vals))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rouge", action="store_true",
                        help="Skip ROUGE-L (avoids LLM synthesis calls)")
    args = parser.parse_args()

    _clean = data_dir("scatterqa") / "gold_evidence_annotated_cleaned.jsonl"
    gold_path = _clean if _clean.exists() else data_dir("scatterqa") / "gold_evidence_annotated.jsonl"
    if not gold_path.exists():
        logger.error("Annotated file not found: %s", gold_path)
        sys.exit(1)

    records = load_gold_evidence(gold_path)
    logger.info("Loaded %d annotated records", len(records))

    out_dir = results_dir("attr_eval")
    chunker = Chunker(chunk_size=512, overlap=128)

    # Phase 0: Build all indices upfront
    logger.info("=== Phase 0: Building indices ===")
    book_cache = build_book_indices(records, chunker)

    phase1_checkpoint = out_dir / "phase1_checkpoint.json"
    rouge_checkpoint = out_dir / "phase2_rouge_checkpoint.json"

    # Phase 1: Retrieval metrics (fast) — resume from checkpoint if available
    if phase1_checkpoint.exists():
        logger.info("=== Phase 1: Loading from checkpoint ===")
        with open(phase1_checkpoint) as f:
            all_results = json.load(f)
        logger.info("Loaded %d Phase 1 records from checkpoint", len(all_results))
    else:
        logger.info("=== Phase 1: Retrieval metrics ===")
        all_results = run_retrieval_metrics(records, book_cache)
        with open(phase1_checkpoint, "w") as f:
            json.dump(all_results, f)
        logger.info("Phase 1 checkpoint saved (%d records)", len(all_results))

    # Phase 2: ROUGE-L (optional, uses LLM)
    include_rouge = not args.skip_rouge
    if include_rouge:
        logger.info("=== Phase 2: ROUGE-L ===")
        all_results = run_rouge_phase(all_results, records, book_cache,
                                      checkpoint_path=rouge_checkpoint)

    # Print and save
    summary = print_results(all_results, include_rouge=include_rouge)

    save_results = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if not k.endswith("_chunk_ids")}
        save_results.append(entry)

    with open(out_dir / "attribute_metrics.json", "w") as f:
        json.dump({"summary": summary, "per_record": save_results}, f, indent=2, default=str)
    logger.info("Saved detailed results to %s", out_dir / "attribute_metrics.json")

    metrics = ["attr_recall", "attr_precision", "attr_f1", "scatter_coverage"]
    headers = ["AttrRec@15", "AttrPre@15", "AttrF1@15", "ScatCov@15"]
    if include_rouge:
        metrics.append("rougeL")
        headers.append("ROUGE-L")
    header = f"{'Strategy':<20} " + " ".join(f"{h:>10}" for h in headers)

    with open(out_dir / "summary_table.txt", "w") as f:
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for sname in STRATEGY_NAMES:
            s = summary[sname]
            vals = " ".join(f"{s.get(m, 0):>10.4f}" for m in metrics)
            f.write(f"{sname:<20} {vals}\n")

    logger.info("Done!")


if __name__ == "__main__":
    main()
