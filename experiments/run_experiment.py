"""CLI experiment runner.

Usage: python -m experiments.run_experiment --exp 1.1
"""

import argparse
import logging
import os
import random
import sys

import numpy as np

EXPERIMENTS = {
    "1.1": ("experiments.phase1.exp1_1_scatter_factor", "Scatter Factor Measurement"),
    "1.2": ("experiments.phase1.exp1_2_rag_failure", "Standard RAG Failure Analysis"),
    "1.3": ("experiments.phase1.exp1_3_completeness_audit", "Chunk Retrieval Completeness Audit"),
    "2.1": ("experiments.phase2.exp2_1_strategy_comparison", "Strategy Comparison (Main Result)"),
    "2.2": ("experiments.phase2.exp2_2_ablation", "Ablation Study"),
    "2.3": ("experiments.phase2.exp2_3_chunk_count", "Chunk Count Analysis"),
    "2.4": ("experiments.phase2.exp2_4_llm_comparison", "LLM Backbone Comparison"),
    "2.5": ("experiments.phase2.exp2_5_ordering", "Ordering Strategy Comparison"),
    "3.1": ("experiments.phase3.exp3_1_scatter_taxonomy", "Scatter Pattern Taxonomy"),
    "3.2": ("experiments.phase3.exp3_2_cross_domain", "Cross-Domain Transfer"),
    "3.3": ("experiments.phase3.exp3_3_error_analysis", "Error Analysis"),
    "3.4": ("experiments.phase3.exp3_4_long_context", "Long-Context Comparison"),
}


def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run experiments for entity-centric scatter-aware RAG")
    parser.add_argument("--exp", required=True, help="Experiment ID (e.g., 1.1, 2.1)")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--multi-seed", action="store_true", help="Run with all seeds from config")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.list:
        for eid, (mod, desc) in sorted(EXPERIMENTS.items()):
            print(f"  {eid:5s}  {desc}")
        return

    if args.exp not in EXPERIMENTS:
        print(f"Unknown experiment: {args.exp}")
        print("Available:", ", ".join(sorted(EXPERIMENTS.keys())))
        sys.exit(1)

    mod_path, desc = EXPERIMENTS[args.exp]
    logging.info("Running Experiment %s: %s", args.exp, desc)

    from src.utils.config import get_experiments_config
    exp_config = get_experiments_config()
    global_config = exp_config.get("global", {})

    if args.multi_seed:
        seeds = global_config.get("seeds", [42])
    else:
        seeds = [args.seed if args.seed is not None
                 else global_config.get("random_seed", 42)]

    import importlib
    mod = importlib.import_module(mod_path)
    all_results = []

    for seed in seeds:
        _set_seed(seed)
        logging.info("Running with seed %d (seed %d/%d)", seed,
                     seeds.index(seed) + 1, len(seeds))
        results = mod.run()
        all_results.append({"seed": seed, "results": results})

    if len(seeds) > 1:
        logging.info("Experiment %s complete with %d seeds.", args.exp, len(seeds))
    else:
        logging.info("Experiment %s complete.", args.exp)
    return all_results if len(seeds) > 1 else all_results[0]["results"]


if __name__ == "__main__":
    main()
