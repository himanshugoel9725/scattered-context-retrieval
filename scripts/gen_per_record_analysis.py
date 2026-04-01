#!/usr/bin/env python3
"""
Generate per-record analysis for all 600 ScatterQA records:
- Question text
- Question type
- Gold attribute chunks (per-attribute breakdown)
- Retrieved chunks (HybridEntity, StandardSemantic)
- Per-attribute overlap

Outputs:
- results/attr_eval/per_record_analysis.jsonl (machine-readable)
- results/attr_eval/per_record_analysis.txt (human-readable)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from collections import defaultdict

from src.evaluation.gold_schema import load_gold_evidence
from src.indexing.chunker import Chunker
from src.indexing.vector_index import VectorIndex
from src.indexing.entity_index import EntityIndex
from src.retrieval.hybrid_entity import HybridEntityRetriever
from src.retrieval.standard_semantic import StandardSemanticRetriever
from src.data.gutenberg import download_gutenberg_text
from src.data.processors import clean_text

# Question templates (copied from eval script)
QUESTION_TEMPLATES = {
    "distributed_attributes": "Provide a comprehensive description of {entity} including their appearance, background, personality, relationships, and development throughout the story.",
    "progressive_accumulation": "How does {entity} change or develop over the course of the narrative?",
    "cross_reference": "Describe all significant relationships that {entity} has with other characters and how these relationships evolve.",
    "implicit": "What motivates {entity}? Explain their goals, desires, and the reasoning behind their key decisions.",
    "contradictory_evolution": "Are there any contradictions or inconsistencies in how {entity} is portrayed throughout the text? If so, describe them.",
}
ATTRIBUTE_LABELS = ["appearance", "background", "personality", "relationships", "arc"]
K = 15

DATA_PATH = Path("data/scatterqa/gold_evidence_annotated.jsonl")
OUT_JSONL = Path("results/attr_eval/per_record_analysis.jsonl")
OUT_TXT = Path("results/attr_eval/per_record_analysis.txt")

# ---
def build_book_indices(records, chunker):
    book_ids_needed = {}
    for rec in records:
        doc_id = rec.entity_id.split(":", 1)[0]
        book_id = int(doc_id.replace("gutenberg_", ""))
        book_ids_needed[book_id] = doc_id
    book_cache = {}
    for book_id, doc_id in book_ids_needed.items():
        text = download_gutenberg_text(book_id)
        text = clean_text(text)
        chunks = chunker.chunk_document(doc_id, text)
        chunk_map = {c.chunk_id: c for c in chunks}
        vector_idx = VectorIndex()
        vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
        entity_idx = EntityIndex(doc_id)
        entity_idx.build_from_chunks(chunks)
        book_cache[book_id] = (chunks, chunk_map, vector_idx, entity_idx)
    return book_cache

# ---
def short_id(cid):
    # Strip gutenberg_NNNN_ prefix for readability
    return cid.split("_c")[-1] if "_c" in cid else cid

def per_attr_overlap(retrieved, attr_chunks):
    overlap = {}
    retrieved_set = set(retrieved)
    for attr in ATTRIBUTE_LABELS:
        gold = attr_chunks.get(attr, [])
        hits = list(retrieved_set & set(gold))
        overlap[attr] = hits
    return overlap

def main():
    print("[INFO] Starting per-record analysis script...")
    print(f"[INFO] Loading gold evidence from {DATA_PATH} ...")
    records = load_gold_evidence(DATA_PATH)
    print(f"[INFO] Loaded {len(records)} records.")
    chunker = Chunker(chunk_size=512, overlap=128)
    print("[INFO] Building book indices...")
    book_cache = build_book_indices(records, chunker)
    print("[INFO] Book indices built for books:", list(book_cache.keys()))
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl = open(OUT_JSONL, "w")
    out_txt = open(OUT_TXT, "w")
    for idx, rec in enumerate(records, 1):
        try:
            print(f"[INFO] Processing record {idx}/{len(records)}: {rec.query_id}")
            doc_id = rec.entity_id.split(":", 1)[0]
            entity_name = rec.entity_id.split(":", 1)[1]
            book_id = int(doc_id.replace("gutenberg_", ""))
            cat = rec.query_metadata.scatter_category
            entity = rec.query_metadata.gold_entity
            query = QUESTION_TEMPLATES.get(cat, "Describe {entity} in detail.").format(entity=entity)
            chunks_list, chunk_map, vector_idx, entity_idx = book_cache[book_id]
            # Retrieval
            print(f"[DEBUG] Running HybridEntityRetriever for record {idx}")
            hybrid = HybridEntityRetriever(vector_idx, entity_idx, chunk_map)
            hybrid_chunks = [r.chunk_id for r in hybrid.retrieve(query, k=K)]
            print(f"[DEBUG] Running StandardSemanticRetriever for record {idx}")
            semantic = StandardSemanticRetriever(vector_idx, chunk_map)
            semantic_chunks = [r.chunk_id for r in semantic.retrieve(query, k=K)]
            # Overlap
            gold_attr_chunks = rec.attribute_chunks
            hybrid_overlap = per_attr_overlap(hybrid_chunks, gold_attr_chunks)
            semantic_overlap = per_attr_overlap(semantic_chunks, gold_attr_chunks)
            # JSONL output
            out_jsonl.write(json.dumps({
                "query_id": rec.query_id,
                "entity_id": rec.entity_id,
                "scatter_category": cat,
                "question": query,
                "gold_attribute_chunks": gold_attr_chunks,
                "retrieved_chunks": {
                    "HybridEntity": hybrid_chunks,
                    "StandardSemantic": semantic_chunks
                },
                "overlap": {
                    "HybridEntity": hybrid_overlap,
                    "StandardSemantic": semantic_overlap
                }
            }) + "\n")
            # TXT output
            out_txt.write(f"=== Record {idx} ===\n")
            out_txt.write(f"Question: {query}\n")
            out_txt.write(f"Question type: {cat}\n")
            out_txt.write(f"Entity: {entity_name} (Book: {doc_id})\n\n")
            out_txt.write(f"Gold attribute chunks:\n")
            for attr in ATTRIBUTE_LABELS + ["unknown"]:
                gold = gold_attr_chunks.get(attr, [])
                if gold:
                    out_txt.write(f"  {attr} ({len(gold)}): [{', '.join(short_id(cid) for cid in gold)}]\n")
            out_txt.write("\n")
            for strat, chunks, overlap in [
                ("HybridEntity", hybrid_chunks, hybrid_overlap),
                ("StandardSemantic", semantic_chunks, semantic_overlap)
            ]:
                out_txt.write(f"Retrieved chunks ({strat}@{K}): [{', '.join(short_id(cid) for cid in chunks)}]\n")
                out_txt.write(f"  Per-attribute overlap:\n")
                total_hits = 0
                for attr in ATTRIBUTE_LABELS:
                    gold = gold_attr_chunks.get(attr, [])
                    hits = overlap[attr]
                    total_hits += len(hits)
                    out_txt.write(f"    {attr}: {len(hits)}/{len(gold)} [{', '.join(short_id(cid) for cid in hits)}]\n")
                out_txt.write(f"  Total overlap with gold: {total_hits}/{K}\n\n")
            out_txt.write("\n")
        except Exception as e:
            print(f"[ERROR] Exception in record {idx} ({rec.query_id}): {e}")
            import traceback
            traceback.print_exc()
            break
    out_jsonl.close()
    out_txt.close()
    print(f"[INFO] Done. Wrote {idx} records to {OUT_JSONL} and {OUT_TXT}")

if __name__ == "__main__":
    main()
