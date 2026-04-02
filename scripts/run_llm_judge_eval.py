#!/usr/bin/env python3
"""LLM-as-judge evaluation on exp2_1 strategy comparison results.

Scores each (query, strategy_answer) pair on completeness, accuracy, and
coherence (1–5) using GPT-4o.  Produces:
  - results/exp2_1/exp2_1_judge_scores.json  (per-record augmented results)
  - results/exp2_1/exp2_1_judge_summary.json (per-strategy / per-dataset means)
  - results/figures/figure14_judge_scores.{png,pdf}

Resume-safe: if exp2_1_judge_scores.json already exists, already-judged
(record_idx, strategy) pairs are skipped.

Run:
    .venv/bin/python scripts/run_llm_judge_eval.py
    .venv/bin/python scripts/run_llm_judge_eval.py --workers 12
    .venv/bin/python scripts/run_llm_judge_eval.py --dry-run  # estimate cost, no API calls
"""

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.llm_judge import llm_judge_score
from src.utils.config import results_dir
from src.utils.plotting import create_figure, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

STRATEGIES = [
    "BM25",
    "StandardSemantic",
    "EntityExpanded",
    "EntityFirst",
    "Iterative",
    "HybridEntity",
]

# Default judge model — gpt-4o-mini has 200K TPM on Tier 1 vs 30K for gpt-4o.
# Use --model gpt-4o for publication-quality scoring if you have higher-tier access.
JUDGE_MODEL = "gpt-4.1-nano"

# ─── Cost estimation ─────────────────────────────────────────────────────────

def estimate_cost(records_by_dataset: dict) -> float:
    """Rough cost estimate based on token counts."""
    total_records = sum(len(v) for v in records_by_dataset.values())
    n_calls = total_records * len(STRATEGIES)
    # Judge prompt: ~200 tokens; reference: ~20 tokens; generated answer: ~500 tokens
    # Response: ~200 tokens
    input_tokens = n_calls * 720
    output_tokens = n_calls * 200
    cost = (input_tokens / 1_000_000) * 2.50 + (output_tokens / 1_000_000) * 10.00
    logger.info(
        "Estimate: %d records × %d strategies = %d judge calls → ~$%.2f",
        total_records, len(STRATEGIES), n_calls, cost,
    )
    return cost


# ─── Core evaluation ─────────────────────────────────────────────────────────

def _judge_task(args: tuple) -> tuple:
    """Worker function: judge one (record, strategy) pair.

    Returns (dataset, record_idx, strategy, judge_result).
    """
    dataset, record_idx, strategy, query, generated_answer, reference, model, stagger = args
    if stagger > 0:
        time.sleep(stagger)
    result = llm_judge_score(
        query=query,
        generated_answer=generated_answer,
        reference_answer=reference,
        model=model,
        temperature=0.0,
    )
    return dataset, record_idx, strategy, result


def run_judge_eval(
    records_by_dataset: dict,
    existing_scores: dict,
    max_workers: int,
    model: str = JUDGE_MODEL,
) -> dict:
    """Run judge evaluation with resumption support.

    existing_scores: dict mapping dataset → list of per-record dicts (may be
    partially filled with {Strategy}_judge_* keys).
    Returns the fully-populated scores structure.
    """
    # Build task list — skip already-judged pairs
    tasks: list[tuple] = []
    for dataset, records in records_by_dataset.items():
        for i, rec in enumerate(records):
            dataset_scores = existing_scores.get(dataset, []) if existing_scores else []
            existing_rec = dataset_scores[i] if i < len(dataset_scores) else {}
            for strategy in STRATEGIES:
                done_key = f"{strategy}_judge_overall"
                if done_key in existing_rec and not math.isnan(
                    existing_rec[done_key]
                ):
                    continue  # already judged successfully
                tasks.append((
                    dataset, i, strategy,
                    rec["query"],
                    rec.get(f"{strategy}_answer", ""),
                    rec.get("reference", ""),
                    model,
                    0,  # stagger: will be set below
                ))

    total = len(tasks)
    logger.info("Tasks to judge: %d (skipping %d already done)",
                total, len(STRATEGIES) * sum(len(v) for v in records_by_dataset.values()) - total)

    if total == 0:
        logger.info("Nothing to do — all records already judged.")
        return existing_scores

    # Build output structure (start from existing or copy records)
    scores: dict = {}
    for dataset, records in records_by_dataset.items():
        scores[dataset] = []
        for i, rec in enumerate(records):
            base = dict(rec)  # copy all original fields
            if existing_scores and i < len(existing_scores.get(dataset, [])):
                # Merge existing judge scores
                base.update(existing_scores[dataset][i])
            scores[dataset].append(base)

    completed = 0
    start = time.perf_counter()

    # Stagger task submission so workers don't all fire simultaneously at t=0
    stagger_step = 0.5  # seconds between worker launches
    tasks_staggered = [
        (*t[:-1], stagger_step * (i % max_workers))
        for i, t in enumerate(tasks)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_task, t): t for t in tasks_staggered}
        for future in as_completed(futures):
            dataset, record_idx, strategy, result = future.result()
            rec = scores[dataset][record_idx]
            rec[f"{strategy}_judge_completeness"] = result.get("completeness", math.nan)
            rec[f"{strategy}_judge_accuracy"] = result.get("accuracy", math.nan)
            rec[f"{strategy}_judge_coherence"] = result.get("coherence", math.nan)
            rec[f"{strategy}_judge_overall"] = result.get("overall", math.nan)
            # Don't store reasoning in per-record (verbose); drop it
            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.perf_counter() - start
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] %.0f%% — %.1f calls/s — ~%.0fs remaining",
                    completed, total, 100 * completed / total, rate, remaining,
                )

    return scores


# ─── Summary statistics ───────────────────────────────────────────────────────

def compute_summary(scores: dict) -> dict:
    """Compute per-strategy and per-dataset mean scores."""
    dims = ["completeness", "accuracy", "coherence", "overall"]

    # Per-strategy (across all datasets)
    strategy_summary: dict = {}
    for strategy in STRATEGIES:
        vals: dict = {d: [] for d in dims}
        for records in scores.values():
            for rec in records:
                for dim in dims:
                    v = rec.get(f"{strategy}_judge_{dim}", math.nan)
                    if not math.isnan(v):
                        vals[dim].append(v)
        strategy_summary[strategy] = {
            dim: {
                "mean": float(np.mean(vals[dim])) if vals[dim] else math.nan,
                "std": float(np.std(vals[dim])) if vals[dim] else math.nan,
                "n": len(vals[dim]),
            }
            for dim in dims
        }

    # Per-dataset × per-strategy
    dataset_summary: dict = {}
    for dataset, records in scores.items():
        dataset_summary[dataset] = {}
        for strategy in STRATEGIES:
            vals = {d: [] for d in dims}
            for rec in records:
                for dim in dims:
                    v = rec.get(f"{strategy}_judge_{dim}", math.nan)
                    if not math.isnan(v):
                        vals[dim].append(v)
            dataset_summary[dataset][strategy] = {
                dim: float(np.mean(vals[dim])) if vals[dim] else math.nan
                for dim in dims
            }

    return {"by_strategy": strategy_summary, "by_dataset": dataset_summary}


# ─── Figure generation ────────────────────────────────────────────────────────

def plot_judge_figure(summary: dict, out_dir: Path) -> None:
    """Figure 14: grouped bar — completeness / accuracy / coherence per strategy."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    by_strategy = summary["by_strategy"]
    strategies = STRATEGIES
    dims = ["completeness", "accuracy", "coherence"]
    dim_labels = ["Completeness", "Accuracy", "Coherence"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(strategies))
    bar_w = 0.24
    offsets = [-bar_w, 0, bar_w]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (dim, label, color) in enumerate(zip(dims, dim_labels, colors)):
        means = [by_strategy[s][dim]["mean"] for s in strategies]
        stds = [by_strategy[s][dim]["std"] for s in strategies]
        ax.bar(
            x + offsets[i], means, bar_w,
            label=label, color=color, alpha=0.85,
            yerr=stds, capsize=3, error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("LLM Judge Score (1–5)")
    ax.set_title("LLM-as-Judge Evaluation: Completeness / Accuracy / Coherence by Strategy")
    ax.set_ylim(1, 5)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, out_dir / "figure14_judge_scores.pdf")
    logger.info("Saved figure14_judge_scores to results/figures/ and %s", out_dir)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent judge calls (default: 3)")
    parser.add_argument("--model", default=JUDGE_MODEL,
                        help=f"Judge model (default: {JUDGE_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost and exit without making API calls")
    args = parser.parse_args()

    out_dir = results_dir("exp2_1")
    scores_path = out_dir / "exp2_1_judge_scores.json"
    summary_path = out_dir / "exp2_1_judge_summary.json"

    # Load exp2_1 source results
    results_path = out_dir / "exp2_1_results.json"
    if not results_path.exists():
        logger.error("exp2_1_results.json not found at %s", results_path)
        sys.exit(1)

    with open(results_path) as f:
        records_by_dataset: dict = json.load(f)

    estimate_cost(records_by_dataset)

    if args.dry_run:
        logger.info("--dry-run: exiting without API calls.")
        return

    # Load existing partial results if present
    existing_scores: dict = {}
    if scores_path.exists():
        with open(scores_path) as f:
            existing_scores = json.load(f)
        logger.info("Resuming from existing %s", scores_path)

    logger.info("Judge model: %s, workers: %d", args.model, args.workers)

    # Run evaluation
    scores = run_judge_eval(records_by_dataset, existing_scores, args.workers, model=args.model)

    # Save per-record results
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2, allow_nan=False,
                  default=lambda x: None if math.isnan(x) else x)
    logger.info("Saved per-record judge scores → %s", scores_path)

    # Save summary
    summary = compute_summary(scores)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary → %s", summary_path)

    # Print strategy table
    print("\n── LLM Judge Summary (mean ± std across all datasets) ──")
    print(f"{'Strategy':<22} {'Completeness':>13} {'Accuracy':>10} {'Coherence':>10} {'Overall':>9}")
    print("─" * 70)
    for strategy in STRATEGIES:
        s = summary["by_strategy"][strategy]
        print(
            f"{strategy:<22}"
            f"  {s['completeness']['mean']:5.3f}±{s['completeness']['std']:.3f}"
            f"  {s['accuracy']['mean']:5.3f}±{s['accuracy']['std']:.3f}"
            f"  {s['coherence']['mean']:5.3f}±{s['coherence']['std']:.3f}"
            f"  {s['overall']['mean']:5.3f}"
        )

    # Generate figure
    plot_judge_figure(summary, out_dir)


if __name__ == "__main__":
    main()
