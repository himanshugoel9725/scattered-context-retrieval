"""Validate scatter factor formula against human judgment.

Presents entities with their chunk positions and asks a human to rate
"scatteredness" on a 1-5 scale. Computes Spearman correlation between
SF formula and human ratings.

Usage:
    PYTHONPATH=. python scripts/validate_scatter_factor.py [--n 30]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.evaluation.metrics import compute_scatter_factor
from src.utils.config import results_dir

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate scatter factor formula")
    parser.add_argument("--n", type=int, default=30, help="Number of entities to validate")
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = results_dir("validation")
    ratings_path = out_dir / "scatter_factor_human_ratings.json"
    results_path = out_dir / "scatter_factor_validation_results.json"

    # Load dataset and build entity indices
    logger.info("Loading dataset and building entity indices...")
    docs = load_dataset("narrativeqa", max_docs=30)
    chunker = Chunker(chunk_size=512, overlap=128)

    all_entities = []  # (entity_name, chunk_ids, chunks_dict, sf_value, doc_id, positions)

    for doc in docs:
        text = clean_text(doc.text)
        chunks = chunker.chunk_document(doc.doc_id, text)
        if len(chunks) < 5:
            continue

        chunk_map = {c.chunk_id: c for c in chunks}
        entity_idx = EntityIndex()
        entity_idx.build_from_chunks(chunks)

        for ename, edata in entity_idx.entities.items():
            cids = edata.get("chunk_ids", [])
            if len(cids) < 2:
                continue
            sf = compute_scatter_factor(ename, cids, chunk_map)
            positions = sorted([chunk_map[cid].position_fraction for cid in cids if cid in chunk_map])
            all_entities.append({
                "entity": ename,
                "doc_id": doc.doc_id,
                "n_chunks": len(cids),
                "total_doc_chunks": len(chunks),
                "positions": [round(p, 3) for p in positions],
                "scatter_factor": round(sf, 4),
            })

    if len(all_entities) < args.n:
        logger.warning("Only %d entities available, using all", len(all_entities))
        args.n = len(all_entities)

    # Sample stratified: 1/3 low SF, 1/3 mid, 1/3 high
    sorted_ents = sorted(all_entities, key=lambda e: e["scatter_factor"])
    third = len(sorted_ents) // 3
    low = sorted_ents[:third]
    mid = sorted_ents[third:2*third]
    high = sorted_ents[2*third:]

    rng = np.random.RandomState(42)
    n_per = args.n // 3
    sample = []
    for group in [low, mid, high]:
        if len(group) <= n_per:
            sample.extend(group)
        else:
            idxs = rng.choice(len(group), size=n_per, replace=False)
            sample.extend(group[i] for i in idxs)
    rng.shuffle(sample)  # randomize presentation order

    # Load existing ratings if resuming
    existing_ratings = {}
    if args.resume and ratings_path.exists():
        with open(ratings_path) as f:
            existing_ratings = json.load(f)

    # Interactive rating
    print("\n" + "=" * 60)
    print("SCATTER FACTOR VALIDATION")
    print("Rate how scattered each entity's information is (1-5)")
    print("  1 = All info in one place (adjacent chunks)")
    print("  2 = Mostly clustered, minor spread")
    print("  3 = Moderately spread across document")
    print("  4 = Widely spread, info in many sections")
    print("  5 = Maximally dispersed across the entire document")
    print("Type 'q' to save and quit early")
    print("=" * 60)

    ratings = dict(existing_ratings)

    for i, ent in enumerate(sample):
        key = f"{ent['doc_id']}:{ent['entity']}"
        if key in ratings:
            continue

        pos_str = ", ".join(f"{p:.0%}" for p in ent["positions"])
        print(f"\n[{i+1}/{len(sample)}] Entity: \"{ent['entity']}\"")
        print(f"  Document chunks: {ent['total_doc_chunks']}")
        print(f"  Entity appears in {ent['n_chunks']} chunks")
        print(f"  Chunk positions (% through doc): [{pos_str}]")
        print(f"  Visual: ", end="")

        # Simple ASCII visualization of chunk positions
        bar_len = 50
        bar = ["."] * bar_len
        for p in ent["positions"]:
            idx = min(int(p * bar_len), bar_len - 1)
            bar[idx] = "#"
        print("[" + "".join(bar) + "]")

        while True:
            choice = input("  Scatteredness rating (1-5, q=quit): ").strip().lower()
            if choice == "q":
                break
            try:
                rating = int(choice)
                if 1 <= rating <= 5:
                    break
            except ValueError:
                pass
            print("  Invalid input. Enter 1-5 or 'q'.")

        if choice == "q":
            break
        ratings[key] = {"rating": rating, **ent}

        # Save progress
        with open(ratings_path, "w") as f:
            json.dump(ratings, f, indent=2)

    # Compute correlation
    rated_items = [v for v in ratings.values() if isinstance(v, dict) and "rating" in v]
    if len(rated_items) < 5:
        print(f"\nOnly {len(rated_items)} ratings. Need at least 5 for correlation.")
        return

    sf_values = [item["scatter_factor"] for item in rated_items]
    human_ratings = [item["rating"] for item in rated_items]

    spearman_r, spearman_p = stats.spearmanr(sf_values, human_ratings)
    pearson_r, pearson_p = stats.pearsonr(sf_values, human_ratings)

    report = {
        "n_rated": len(rated_items),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 6),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 6),
        "passes_threshold": spearman_r >= 0.6 and spearman_p < 0.05,
        "sf_range": [round(min(sf_values), 4), round(max(sf_values), 4)],
        "rating_distribution": {str(r): human_ratings.count(r) for r in range(1, 6)},
    }

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(rated_items)} entities rated)")
    print(f"  Spearman r:    {spearman_r:.3f}  (p={spearman_p:.4f})")
    print(f"  Pearson r:     {pearson_r:.3f}  (p={pearson_p:.4f})")
    print(f"  SF range:      [{min(sf_values):.4f}, {max(sf_values):.4f}]")
    print(f"  Threshold met: {'YES' if report['passes_threshold'] else 'NO'} (Spearman >= 0.6, p < 0.05)")
    print(f"{'=' * 60}")

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
