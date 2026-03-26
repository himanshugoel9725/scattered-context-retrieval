"""Validate the LLM query classifier against human labels.

Samples queries, gets LLM predictions, then presents an interactive CLI
for a human annotator. Reports accuracy and Cohen's kappa.

Usage:
    PYTHONPATH=. python scripts/validate_classification.py [--n 50]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.loaders import load_dataset
from src.utils.config import results_dir
from experiments.phase1.exp1_2_rag_failure import classify_query

logger = logging.getLogger(__name__)


def _cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two lists of binary labels."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Build confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    p_o = np.trace(matrix) / n  # observed agreement
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = np.sum(row_sums * col_sums)  # expected agreement

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def load_gold_labels(path: Path) -> dict[str, str]:
    """Load previously saved gold labels."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Validate query classifier with human labels")
    parser.add_argument("--n", type=int, default=50, help="Number of queries to validate")
    parser.add_argument("--dataset", default="narrativeqa", help="Dataset to sample from")
    parser.add_argument("--resume", action="store_true", help="Resume from saved progress")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = results_dir("validation")
    gold_path = out_dir / "classification_gold_labels.json"
    results_path = out_dir / "classification_validation_results.json"

    # Load dataset and sample queries
    docs = load_dataset(args.dataset, max_docs=100)
    all_queries = []
    for doc in docs:
        for qa in doc.questions:
            all_queries.append({"question": qa.question, "doc_id": doc.doc_id})

    if len(all_queries) < args.n:
        logger.warning("Only %d queries available, using all", len(all_queries))
        args.n = len(all_queries)

    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_queries), size=args.n, replace=False)
    sample = [all_queries[i] for i in indices]

    # Get LLM predictions
    logger.info("Classifying %d queries with LLM...", args.n)
    llm_labels = {}
    for item in sample:
        q = item["question"]
        llm_labels[q] = classify_query(q)

    # Load existing gold labels if resuming
    gold_labels = load_gold_labels(gold_path) if args.resume else {}

    # Interactive annotation
    print("\n" + "=" * 60)
    print("QUERY CLASSIFICATION VALIDATION")
    print("Label each query as 'l' (localized) or 's' (scattered)")
    print("Type 'q' to save and quit early")
    print("=" * 60 + "\n")

    for i, item in enumerate(sample):
        q = item["question"]
        if q in gold_labels:
            continue  # already labeled

        print(f"\n[{i+1}/{args.n}] Question: {q}")
        print(f"  LLM predicted: {llm_labels[q]}")

        while True:
            choice = input("  Your label (l=localized, s=scattered, q=quit): ").strip().lower()
            if choice in ("l", "s", "q"):
                break
            print("  Invalid input. Enter 'l', 's', or 'q'.")

        if choice == "q":
            break
        gold_labels[q] = "localized" if choice == "l" else "scattered"

        # Save progress after each label
        with open(gold_path, "w") as f:
            json.dump(gold_labels, f, indent=2)

    # Compute metrics on labeled subset
    common_keys = [q for q in gold_labels if q in llm_labels]
    if len(common_keys) < 5:
        print(f"\nOnly {len(common_keys)} labels collected. Need at least 5 for meaningful metrics.")
        return

    llm_list = [llm_labels[q] for q in common_keys]
    gold_list = [gold_labels[q] for q in common_keys]

    accuracy = sum(a == b for a, b in zip(llm_list, gold_list)) / len(common_keys)
    kappa = _cohens_kappa(llm_list, gold_list)

    # Breakdown
    n_loc = sum(1 for g in gold_list if g == "localized")
    n_scat = sum(1 for g in gold_list if g == "scattered")
    loc_correct = sum(1 for l, g in zip(llm_list, gold_list) if g == "localized" and l == g)
    scat_correct = sum(1 for l, g in zip(llm_list, gold_list) if g == "scattered" and l == g)

    report = {
        "n_labeled": len(common_keys),
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(kappa, 4),
        "n_localized": n_loc,
        "n_scattered": n_scat,
        "localized_accuracy": round(loc_correct / max(n_loc, 1), 4),
        "scattered_accuracy": round(scat_correct / max(n_scat, 1), 4),
        "passes_threshold": accuracy >= 0.85 and kappa >= 0.6,
    }

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(common_keys)} queries labeled)")
    print(f"  Accuracy:        {accuracy:.1%}")
    print(f"  Cohen's kappa:   {kappa:.3f}")
    print(f"  Localized:       {loc_correct}/{n_loc} correct ({loc_correct/max(n_loc,1):.1%})")
    print(f"  Scattered:       {scat_correct}/{n_scat} correct ({scat_correct/max(n_scat,1):.1%})")
    print(f"  Threshold met:   {'YES' if report['passes_threshold'] else 'NO'} (>=85% acc, >=0.6 kappa)")
    print(f"{'=' * 60}")

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
