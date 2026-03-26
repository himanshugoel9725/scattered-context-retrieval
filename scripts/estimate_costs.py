"""Estimate API costs before running experiments."""

import argparse
import logging

from src.utils.config import get_experiments_config

logger = logging.getLogger(__name__)

# Cost per 1M tokens (input / output)
COST_TABLE = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_experiment_cost(exp_id: str, config: dict) -> dict:
    """Estimate cost for a single experiment."""
    estimates = {}

    if exp_id == "1.1":
        # No LLM calls, only local NER
        estimates = {"total": 0.0, "note": "Local compute only (NER)"}
    elif exp_id == "1.2":
        n_queries = config.get("n_queries", 500)
        # Classification + generation + evaluation
        input_tokens = n_queries * 2000  # avg context
        output_tokens = n_queries * 300
        cost = (input_tokens / 1e6) * COST_TABLE["gpt-4o-mini"][0] + \
               (output_tokens / 1e6) * COST_TABLE["gpt-4o-mini"][1]
        estimates = {"total": cost * 3, "note": f"{n_queries} queries × 2 strategies"}
    elif exp_id == "2.1":
        n_total = 1200  # 300 × 4 datasets
        input_tokens = n_total * 3000
        output_tokens = n_total * 300
        cost = (input_tokens / 1e6) * COST_TABLE["gpt-4o-mini"][0] + \
               (output_tokens / 1e6) * COST_TABLE["gpt-4o-mini"][1]
        estimates = {"total": cost * 6, "note": "1200 queries × 6 strategies"}
    elif exp_id == "2.4":
        n_queries = 200
        # Multiple LLMs at different costs
        total = 0
        for model, (inp, out) in COST_TABLE.items():
            cost = (n_queries * 3000 / 1e6) * inp + (n_queries * 300 / 1e6) * out
            total += cost
        estimates = {"total": total, "note": f"{n_queries} queries × {len(COST_TABLE)} models"}
    elif exp_id == "3.4":
        n_queries = 100
        # Full context = lots of input tokens
        full_input = n_queries * 50000  # ~50K tokens per doc
        cost_4o = (full_input / 1e6) * COST_TABLE["gpt-4o"][0]
        cost_mini = (full_input / 1e6) * COST_TABLE["gpt-4o-mini"][0]
        estimates = {"total": cost_4o + cost_mini, "note": "Full-context comparison"}
    else:
        n_queries = config.get("n_queries", 200)
        input_tokens = n_queries * 3000
        output_tokens = n_queries * 300
        cost = (input_tokens / 1e6) * COST_TABLE["gpt-4o-mini"][0] + \
               (output_tokens / 1e6) * COST_TABLE["gpt-4o-mini"][1]
        estimates = {"total": cost, "note": "Standard estimate"}

    return estimates


def main():
    logging.basicConfig(level=logging.INFO)
    config = get_experiments_config()

    total = 0.0
    print("\n=== API Cost Estimates ===\n")
    for phase_key in ["phase1", "phase2", "phase3"]:
        phase_config = config.get(phase_key, {})
        for exp_key, exp_config in phase_config.items():
            exp_id = exp_key.replace("exp", "").replace("_", ".")
            est = estimate_experiment_cost(exp_id, exp_config)
            print(f"  Exp {exp_id:5s}: ${est['total']:8.4f}  ({est['note']})")
            total += est["total"]

    print(f"\n  {'TOTAL':>10s}: ${total:8.4f}")
    print(f"\n  Budget remaining: ${100 - total:.2f} (of $100)")


if __name__ == "__main__":
    main()
